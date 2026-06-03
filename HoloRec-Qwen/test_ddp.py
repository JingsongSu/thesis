import argparse
import json
import os
from typing import Dict, List, Optional

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)

from peft import PeftModel

from utils import *
from collator import TestCollator
from evaluate import get_topk_results, get_metrics_results


def normalize_code_text(x):
    if x is None:
        return None
    x = str(x)
    x = x.split("Response:")[-1]
    return x.strip().replace(" ", "")


def flatten_tokenizer_ids(encoded):
    ids = encoded.get("input_ids", encoded)
    if isinstance(ids, torch.Tensor):
        ids = ids.tolist()
    if len(ids) > 0 and isinstance(ids[0], list):
        ids = ids[0]
    return [int(x) for x in ids]


def token_to_id(tokenizer, token):
    token_id = tokenizer.convert_tokens_to_ids(token)

    if token_id is None:
        encoded = tokenizer(token, add_special_tokens=False)
        ids = flatten_tokenizer_ids(encoded)
        if len(ids) == 1:
            return int(ids[0])
        raise ValueError(f"Token {token!r} is not a single token: {ids}")

    unk_id = getattr(tokenizer, "unk_token_id", None)
    if unk_id is not None and int(token_id) == int(unk_id):
        encoded = tokenizer(token, add_special_tokens=False)
        ids = flatten_tokenizer_ids(encoded)
        if len(ids) == 1:
            return int(ids[0])
        raise ValueError(
            f"Token {token!r} maps to unk or multiple ids: {ids}. "
            "Please make sure tokenizer has loaded the saved HoloRec code tokens."
        )

    return int(token_id)


class Trie:
    def __init__(self):
        self.root: Dict[int, Dict] = {}

    def insert(self, ids: List[int]):
        node = self.root
        for x in ids:
            node = node.setdefault(int(x), {})

    def get(self, prefix: List[int]) -> List[int]:
        node = self.root
        for x in prefix:
            x = int(x)
            if x not in node:
                return []
            node = node[x]
        return list(node.keys())

    def __len__(self):
        def count_nodes(node):
            total = 1
            for child in node.values():
                total += count_nodes(child)
            return total

        return count_nodes(self.root)


def setup_distributed():
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size > 1
    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    if ddp:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")

    return ddp, rank, local_rank, world_size


def cleanup_distributed(ddp):
    if ddp and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


def to_device(inputs, device):
    return {k: v.to(device) for k, v in inputs.items()}


def infer_fine_code_length(test_data) -> int:
    if hasattr(test_data, "indices") and len(test_data.indices) > 0:
        first_key = next(iter(test_data.indices.keys()))
        return len(test_data.indices[first_key])
    raise ValueError("Cannot infer fine code length from test dataset.")


def build_fine_trie_from_dataset(test_data, tokenizer) -> Trie:
    trie = Trie()

    if not hasattr(test_data, "indices"):
        raise ValueError("test_data has no `indices`; cannot build fine trie.")

    for _, seq in test_data.indices.items():
        ids = [token_to_id(tokenizer, tok) for tok in seq]
        trie.insert(ids)

    return trie


def build_fine_all_items(test_data):
    if hasattr(test_data, "get_all_items"):
        return set(normalize_code_text(x) for x in test_data.get_all_items())

    if hasattr(test_data, "indices"):
        return set(normalize_code_text("".join(v)) for v in test_data.indices.values())

    return None


def build_coarse_position_token_ids(test_data, tokenizer) -> List[List[int]]:
    if not hasattr(test_data, "coarse_indices") or test_data.coarse_indices is None:
        raise ValueError(
            "Interleaved inference requires coarse_indices. "
            "Please pass --coarse_index_file, e.g. --coarse_index_file .tw8.json"
        )

    pos_to_ids = {}
    for _, seq in test_data.coarse_indices.items():
        for pos, tok in enumerate(seq):
            tok_id = token_to_id(tokenizer, tok)
            pos_to_ids.setdefault(pos, set()).add(tok_id)

    if len(pos_to_ids) == 0:
        raise ValueError("coarse position token ids are empty.")

    max_pos = max(pos_to_ids.keys())
    return [sorted(list(pos_to_ids.get(i, set()))) for i in range(max_pos + 1)]


def decode_fine_from_interleaved_ids(
    tokenizer,
    generated_ids: List[int],
    eos_token_id: Optional[int],
    pad_token_id: Optional[int],
):
    fine_ids = []

    for pos, token_id in enumerate(generated_ids):
        token_id = int(token_id)

        if eos_token_id is not None and token_id == int(eos_token_id):
            break
        if pad_token_id is not None and token_id == int(pad_token_id):
            break

        # generated sequence is: coarse_1, fine_1, coarse_2, fine_2, ...
        if pos % 2 == 1:
            fine_ids.append(token_id)

    fine_tokens = tokenizer.convert_ids_to_tokens(fine_ids)
    fine_tokens = [
        tok for tok in fine_tokens
        if tok not in set(tokenizer.all_special_tokens)
    ]

    return normalize_code_text("".join(fine_tokens))


@torch.no_grad()
def constrained_interleaved_generate(
    model,
    tokenizer,
    inputs,
    fine_trie: Trie,
    fine_code_len: int,
    coarse_position_token_ids: List[List[int]],
    num_beams: int,
):
    prompt_len = inputs["input_ids"].size(1)
    device = inputs["input_ids"].device

    eos_id = tokenizer.eos_token_id
    pad_id = tokenizer.pad_token_id

    if eos_id is None:
        raise ValueError("tokenizer.eos_token_id is None; cannot stop generation safely.")

    max_new_tokens = fine_code_len * 2 + 1

    def prefix_allowed_tokens_fn(batch_id, sentence):
        gen_pos = len(sentence) - prompt_len

        if gen_pos < 0:
            return [int(eos_id)]

        if gen_pos >= fine_code_len * 2:
            return [int(eos_id)]

        # even generated positions are coarse tokens
        if gen_pos % 2 == 0:
            coarse_pos = min(gen_pos // 2, len(coarse_position_token_ids) - 1)
            allowed = coarse_position_token_ids[coarse_pos]
            return allowed if len(allowed) > 0 else [int(eos_id)]

        # odd generated positions are fine tokens, constrained by fine trie
        generated = sentence[prompt_len:].detach().cpu().tolist()
        fine_prefix = []

        for pos, token_id in enumerate(generated):
            if pos >= gen_pos:
                break
            if pos % 2 == 1:
                fine_prefix.append(int(token_id))

        allowed = fine_trie.get(fine_prefix)
        return allowed if len(allowed) > 0 else [int(eos_id)]

    output = model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_new_tokens=max_new_tokens,
        prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
        num_beams=num_beams,
        num_return_sequences=num_beams,
        output_scores=True,
        return_dict_in_generate=True,
        early_stopping=True,
        do_sample=False,
        pad_token_id=pad_id,
        eos_token_id=eos_id,
        use_cache=True,
    )

    output_ids = output.sequences
    generated_ids = output_ids[:, prompt_len:]

    if hasattr(output, "sequences_scores") and output.sequences_scores is not None:
        scores = output.sequences_scores.detach().cpu()
    else:
        scores = torch.zeros(output_ids.size(0), dtype=torch.float)

    decoded = []
    for row in generated_ids:
        decoded.append(
            decode_fine_from_interleaved_ids(
                tokenizer=tokenizer,
                generated_ids=row.detach().cpu().tolist(),
                eos_token_id=eos_id,
                pad_token_id=pad_id,
            )
        )

    return decoded, scores


def make_quant_config(args, compute_dtype):
    if args.load_in_4bit:
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=compute_dtype,
        )

    if args.load_in_8bit:
        return BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_threshold=6.0,
            llm_int8_has_fp16_weight=False,
        )

    return None


def load_model_and_tokenizer(args, device, local_rank):
    tokenizer_path = args.ckpt_path if args.ckpt_path else args.base_model

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
        model_max_length=args.model_max_length,
        padding_side="left",
        trust_remote_code=True,
        use_fast=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)

    compute_dtype = torch.float32
    if args.bf16:
        compute_dtype = torch.bfloat16
    elif args.fp16:
        compute_dtype = torch.float16

    quant_config = make_quant_config(args, compute_dtype)

    model_kwargs = dict(
        torch_dtype=compute_dtype if (args.fp16 or args.bf16) else None,
        trust_remote_code=True,
    )

    if quant_config is not None:
        model_kwargs["quantization_config"] = quant_config
        model_kwargs["device_map"] = {"": local_rank}
    else:
        model_kwargs["device_map"] = {"": local_rank}

    if args.attn_implementation:
        model_kwargs["attn_implementation"] = args.attn_implementation

    if args.lora:
        base = AutoModelForCausalLM.from_pretrained(
            args.base_model,
            **model_kwargs,
        )
        base.resize_token_embeddings(len(tokenizer))
        model = PeftModel.from_pretrained(base, args.ckpt_path)
    else:
        load_path = args.ckpt_path if args.ckpt_path else args.base_model
        model = AutoModelForCausalLM.from_pretrained(
            load_path,
            **model_kwargs,
        )
        model.resize_token_embeddings(len(tokenizer))

    if hasattr(model.config, "use_cache"):
        model.config.use_cache = True

    model.eval()
    return model, tokenizer


def parse_prompt_ids(args):
    if args.test_prompt_ids == "all":
        if args.test_task.lower() == "seqrec":
            return range(len(all_prompt["seqrec"]))
        elif args.test_task.lower() == "itemsearch":
            return range(len(all_prompt["itemsearch"]))
        elif args.test_task.lower() == "fusionseqrec":
            return range(len(all_prompt["fusionseqrec"]))
        else:
            raise NotImplementedError

    return [int(x) for x in args.test_prompt_ids.split(",") if x.strip()]


def gather_objects(obj, ddp, world_size):
    if not ddp:
        return [obj]

    gathered = [None for _ in range(world_size)]
    dist.all_gather_object(gathered, obj)
    return gathered


def main(args):
    ddp, rank, local_rank, world_size = setup_distributed()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    set_seed(args.seed)

    model, tokenizer = load_model_and_tokenizer(args, device, local_rank)

    test_data = load_test_dataset(args)

    # If tokenizer was loaded from base_model instead of ckpt_path, add missing code tokens.
    if hasattr(test_data, "get_new_tokens"):
        added = tokenizer.add_tokens(test_data.get_new_tokens())
        if added > 0:
            model.resize_token_embeddings(len(tokenizer))
            if rank == 0:
                print(f"Warning: tokenizer missed {added} code tokens; added them at test time.")

    fine_code_len = infer_fine_code_length(test_data)
    fine_trie = build_fine_trie_from_dataset(test_data, tokenizer)
    coarse_position_token_ids = build_coarse_position_token_ids(test_data, tokenizer)
    fine_all_items = build_fine_all_items(test_data) if args.filter_items else None

    if rank == 0:
        print("fine_code_len:", fine_code_len)
        print("coarse positions:", len(coarse_position_token_ids))
        print("fine trie nodes:", len(fine_trie))
        print("num_beams:", args.num_beams)

    prompt_ids = list(parse_prompt_ids(args))
    metrics = [x.strip() for x in args.metrics.split(",") if x.strip()]

    all_prompt_results = {}

    for prompt_id in prompt_ids:
        if hasattr(test_data, "set_prompt"):
            test_data.set_prompt(prompt_id)

        sampler = DistributedSampler(
            test_data,
            num_replicas=world_size,
            rank=rank,
            shuffle=False,
            drop_last=False,
        ) if ddp else None

        collator = TestCollator(args, tokenizer)

        test_loader = DataLoader(
            test_data,
            batch_size=args.test_batch_size,
            sampler=sampler,
            shuffle=False if sampler is not None else False,
            collate_fn=collator,
            num_workers=args.num_workers,
            pin_memory=True,
        )

        local_topk_results = []
        local_top1_pairs = []

        for step, batch in enumerate(test_loader):
            inputs, targets, coarse_targets = batch
            inputs = to_device(inputs, device)

            predictions, scores = constrained_interleaved_generate(
                model=model,
                tokenizer=tokenizer,
                inputs=inputs,
                fine_trie=fine_trie,
                fine_code_len=fine_code_len,
                coarse_position_token_ids=coarse_position_token_ids,
                num_beams=args.num_beams,
            )

            topk_results, top1_pairs = get_topk_results(
                predictions=predictions,
                scores=scores,
                fine_targets=targets,
                coarse_targets=coarse_targets,
                k=args.num_beams,
                all_items=fine_all_items,
            )

            local_topk_results.extend(topk_results)
            local_top1_pairs.extend(top1_pairs)

            if rank == 0 and step % args.logging_step == 0:
                print(f"prompt {prompt_id} step {step}")

        gathered = gather_objects(
            {
                "topk_results": local_topk_results,
                "top1_pairs": local_top1_pairs,
            },
            ddp=ddp,
            world_size=world_size,
        )

        if rank == 0:
            merged_topk = []
            merged_pairs = []

            for part in gathered:
                merged_topk.extend(part["topk_results"])
                merged_pairs.extend(part["top1_pairs"])

            metric_values = get_metrics_results(merged_topk, metrics)
            denom = max(1, len(merged_topk))
            metric_values = {k: float(v) / denom for k, v in metric_values.items()}

            all_prompt_results[str(prompt_id)] = {
                "metrics": metric_values,
                "num_examples": len(merged_topk),
                "top1_pairs": merged_pairs[:100],
            }

            print("prompt_id:", prompt_id)
            print(json.dumps(metric_values, indent=2, ensure_ascii=False))

    if rank == 0:
        ensure_dir(os.path.dirname(args.results_file) or ".")
        with open(args.results_file, "w", encoding="utf-8") as f:
            json.dump(all_prompt_results, f, indent=2, ensure_ascii=False)

        print("Save results to:", args.results_file)

    cleanup_distributed(ddp)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fast HoloRec-Qwen DDP Test")

    parser = parse_global_args(parser)
    parser = parse_dataset_args(parser)
    parser = parse_test_args(parser)
    parser = parse_train_args(parser)

    parser.add_argument("--coarse_index_file", type=str, default=".tw8.json")

    parser.add_argument("--load_in_4bit", action="store_true", default=False)
    parser.add_argument("--load_in_8bit", action="store_true", default=False)

    parser.add_argument(
        "--attn_implementation",
        type=str,
        default="flash_attention_2",
    )

    parser.add_argument("--num_workers", type=int, default=2)

    args = parser.parse_args()

    if args.load_in_4bit and args.load_in_8bit:
        raise ValueError("Choose only one of --load_in_4bit and --load_in_8bit.")

    main(args)
