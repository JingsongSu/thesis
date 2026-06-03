import argparse
import json
import os
from typing import Dict, List, Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

from utils import *
from collator import TestCollator
from prompt import all_prompt
from evaluate import get_topk_results, get_metrics_results
from generation_trie import Trie, build_trie_from_token_sequences


def build_model_and_tokenizer(args, device_map):
    """
    Load tokenizer from ckpt_path because ckpt_path contains added fine/coarse code tokens.
    Load base model + LoRA adapter when args.lora=True.
    """

    tokenizer = AutoTokenizer.from_pretrained(
        args.ckpt_path,
        trust_remote_code=True,
        use_fast=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
    tokenizer.padding_side = "left"

    if getattr(args, "use_bf16", False):
        dtype = torch.bfloat16
    elif getattr(args, "use_fp16", False):
        dtype = torch.float16
    else:
        dtype = torch.bfloat16

    if args.lora:
        base = AutoModelForCausalLM.from_pretrained(
            args.base_model,
            device_map=device_map,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )

        base.resize_token_embeddings(len(tokenizer))

        model = PeftModel.from_pretrained(
            base,
            args.ckpt_path,
            device_map=device_map,
            torch_dtype=dtype,
        )
    else:
        if getattr(args, "load_in_8bit", False):
            quant_config = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False,
            )

            model = AutoModelForCausalLM.from_pretrained(
                args.ckpt_path,
                device_map=device_map,
                quantization_config=quant_config,
                torch_dtype=dtype,
                low_cpu_mem_usage=True,
                trust_remote_code=True,
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                args.ckpt_path,
                device_map=device_map,
                torch_dtype=dtype,
                low_cpu_mem_usage=True,
                trust_remote_code=True,
            )

    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False

    model.eval()
    return tokenizer, model


def gather_object_list(local_obj, world_size):
    gathered = [None for _ in range(world_size)]
    dist.all_gather_object(gathered, local_obj)

    merged = []

    for part in gathered:
        if part is None:
            continue
        if isinstance(part, list):
            merged.extend(part)
        else:
            merged.append(part)

    return merged


def normalize_code_text(x):
    if x is None:
        return None
    return str(x).strip().replace(" ", "")


def code_tokens_to_text(tokens: List[str]) -> str:
    return normalize_code_text("".join(tokens))


def token_to_id(tokenizer, token: str) -> int:
    token_id = tokenizer.convert_tokens_to_ids(token)

    if token_id is None:
        encoded = tokenizer(token, add_special_tokens=False).get("input_ids", [])

        if len(encoded) == 1:
            return int(encoded[0])

        raise ValueError(
            f"Token {token!r} is not a single tokenizer id: {encoded}. "
            f"Please ensure fine/coarse code tokens have been added to tokenizer."
        )

    unk_id = getattr(tokenizer, "unk_token_id", None)

    if unk_id is not None and token_id == unk_id:
        encoded = tokenizer(token, add_special_tokens=False).get("input_ids", [])

        if len(encoded) == 1:
            return int(encoded[0])

        raise ValueError(
            f"Token {token!r} is mapped to unk or multiple ids: {encoded}. "
            f"Please ensure fine/coarse code tokens have been added to tokenizer."
        )

    return int(token_id)


def infer_fine_code_length(test_data) -> int:
    if hasattr(test_data, "indices") and len(test_data.indices) > 0:
        first_key = next(iter(test_data.indices.keys()))
        return len(test_data.indices[first_key])

    raise ValueError("Cannot infer fine code length from test dataset.")


def build_fine_trie_from_dataset(test_data, tokenizer) -> Trie:
    if not hasattr(test_data, "indices"):
        raise ValueError("test_data must have `indices` for fine trie construction.")

    fine_token_sequences = list(test_data.indices.values())
    return build_trie_from_token_sequences(fine_token_sequences, tokenizer)


def build_fine_all_items(test_data) -> set:
    if not hasattr(test_data, "indices"):
        return set()

    return set(code_tokens_to_text(v) for v in test_data.indices.values())


def build_coarse_position_token_ids(test_data, tokenizer) -> List[List[int]]:
    """
    Build allowed coarse token ids for each coarse-code position.

    Example:
        --coarse_index_file .tw8.json

    This means coarse_position_token_ids[pos] contains all possible .tw8.json
    tokens at position pos.
    """

    if not hasattr(test_data, "coarse_indices") or test_data.coarse_indices is None:
        raise ValueError(
            "Latent interleaved inference requires coarse_indices. "
            "Please pass --coarse_index_file, for example --coarse_index_file .tw8.json"
        )

    pos_to_ids = {}

    for _, seq in test_data.coarse_indices.items():
        for pos, tok in enumerate(seq):
            tok_id = token_to_id(tokenizer, tok)
            pos_to_ids.setdefault(pos, set()).add(tok_id)

    if len(pos_to_ids) == 0:
        raise ValueError("coarse position token ids are empty.")

    max_pos = max(pos_to_ids.keys())

    return [
        sorted(list(pos_to_ids.get(i, set())))
        for i in range(max_pos + 1)
    ]


def get_model_input_embeddings(model):
    if hasattr(model, "get_input_embeddings"):
        emb = model.get_input_embeddings()
        if emb is not None:
            return emb

    if hasattr(model, "base_model") and hasattr(model.base_model, "get_input_embeddings"):
        emb = model.base_model.get_input_embeddings()
        if emb is not None:
            return emb

    if hasattr(model, "model") and hasattr(model.model, "embed_tokens"):
        return model.model.embed_tokens

    raise ValueError("Cannot locate model input embeddings.")


def get_model_output_embeddings(model):
    if hasattr(model, "get_output_embeddings"):
        emb = model.get_output_embeddings()
        if emb is not None:
            return emb

    if hasattr(model, "lm_head"):
        return model.lm_head

    if hasattr(model, "base_model") and hasattr(model.base_model, "get_output_embeddings"):
        emb = model.base_model.get_output_embeddings()
        if emb is not None:
            return emb

    if hasattr(model, "base_model") and hasattr(model.base_model, "lm_head"):
        return model.base_model.lm_head

    if hasattr(model, "model") and hasattr(model.model, "lm_head"):
        return model.model.lm_head

    raise ValueError("Cannot locate model output embeddings / lm_head.")


def unwrap_peft_module(module):
    """
    PEFT may wrap embed_tokens / lm_head as ModulesToSaveWrapper.

    The real trainable module is usually:
        modules_to_save["default"]
    """

    if module is None:
        return None

    if hasattr(module, "weight"):
        return module

    if hasattr(module, "modules_to_save"):
        modules_to_save = module.modules_to_save

        candidate_names = []

        active_adapter = getattr(module, "active_adapter", None)
        if isinstance(active_adapter, str):
            candidate_names.append(active_adapter)
        elif isinstance(active_adapter, (list, tuple)):
            candidate_names.extend(list(active_adapter))

        active_adapters = getattr(module, "active_adapters", None)
        if isinstance(active_adapters, str):
            candidate_names.append(active_adapters)
        elif isinstance(active_adapters, (list, tuple)):
            candidate_names.extend(list(active_adapters))

        candidate_names.append("default")

        for name in candidate_names:
            if name in modules_to_save and hasattr(modules_to_save[name], "weight"):
                return modules_to_save[name]

        for real_module in modules_to_save.values():
            if hasattr(real_module, "weight"):
                return real_module

    if hasattr(module, "original_module"):
        original = module.original_module
        if hasattr(original, "weight"):
            return original

    if hasattr(module, "module"):
        return unwrap_peft_module(module.module)

    return module


def get_input_embedding_module(model):
    return unwrap_peft_module(get_model_input_embeddings(model))


def get_input_embedding_weight(model):
    emb = get_input_embedding_module(model)
    if not hasattr(emb, "weight"):
        raise AttributeError(f"Input embedding module has no weight: {type(emb)}")
    return emb.weight


def embed_input_ids(model, input_ids):
    """
    Use real embedding module forward instead of F.embedding.
    This is safer for PEFT modules_to_save.
    """
    emb = get_input_embedding_module(model)
    return emb(input_ids)


@torch.no_grad()
def predict_soft_coarse_embedding(
    next_logits: torch.Tensor,
    embedding_weight: torch.Tensor,
    coarse_token_ids: List[int],
    temperature: float = 1.0,
    use_coarse_score: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    next_logits:
        [N, vocab_size], logits for predicted coarse token.

    coarse_emb:
        [N, hidden_size], weighted sum of coarse token embeddings.
    """

    device = next_logits.device
    coarse_ids = torch.tensor(coarse_token_ids, dtype=torch.long, device=device)

    selected_logits = next_logits.index_select(1, coarse_ids).float()

    temp = max(float(temperature), 1e-6)
    coarse_log_probs = F.log_softmax(selected_logits / temp, dim=-1)
    coarse_probs = torch.exp(coarse_log_probs).to(dtype=embedding_weight.dtype)

    selected_emb = embedding_weight.index_select(0, coarse_ids)
    coarse_emb = torch.matmul(coarse_probs, selected_emb)

    if use_coarse_score:
        coarse_scores = coarse_log_probs.max(dim=-1).values.to(dtype=next_logits.dtype)
    else:
        coarse_scores = torch.zeros(
            next_logits.size(0),
            dtype=next_logits.dtype,
            device=device,
        )

    return coarse_emb, coarse_scores


@torch.no_grad()
def model_next_logits_no_cache(
    model,
    inputs_embeds,
    attention_mask,
):
    """
    Qwen3-safe forward.

    We do not pass past_key_values.
    This avoids old tuple cache vs new Cache object mismatch.
    """

    outputs = model(
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        use_cache=False,
        return_dict=True,
    )

    return outputs.logits[:, -1, :]


@torch.no_grad()
def batch_generate_latent_interleaved_qwen(
    model,
    tokenizer,
    inputs,
    fine_trie: Trie,
    fine_code_len: int,
    coarse_position_token_ids: List[List[int]],
    num_beams: int,
    coarse_temperature: float = 1.0,
    use_coarse_score_in_rank: bool = False,
):
    """
    Latent coarse-to-fine inference.

    This corresponds to the modified training wrapper:

        training:
            context
            -> predicted coarse logits
            -> soft coarse embedding
            -> fine logits
            -> teacher forcing gold fine for next step

        inference:
            context
            -> predicted coarse logits
            -> soft coarse embedding
            -> fine logits
            -> beam search predicted fine for next step

    Output:
        only fine tokens are decoded.
    """

    device = inputs["input_ids"].device
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    batch_size = input_ids.size(0)

    embedding_weight = get_input_embedding_weight(model)
    hidden_size = embedding_weight.size(-1)

    prompt_embeds = embed_input_ids(model, input_ids)

    beams_by_sample = []

    for b in range(batch_size):
        beams_by_sample.append([
            {
                "fine_token_ids": [],
                "score": 0.0,
                "extra_embeds": torch.empty(
                    0,
                    hidden_size,
                    dtype=prompt_embeds.dtype,
                    device=device,
                ),
            }
        ])

    trie_cache: Dict[tuple, List[int]] = {}

    for step_idx in range(fine_code_len):
        flat_beams = []
        flat_sample_ids = []

        for sample_id in range(batch_size):
            for beam in beams_by_sample[sample_id]:
                flat_beams.append(beam)
                flat_sample_ids.append(sample_id)

        if len(flat_beams) == 0:
            break

        full_embeds_list = []
        full_attention_list = []

        for flat_idx, beam in enumerate(flat_beams):
            sample_id = flat_sample_ids[flat_idx]

            sample_prompt_embeds = prompt_embeds[sample_id]
            sample_prompt_mask = attention_mask[sample_id]

            extra_embeds = beam["extra_embeds"]

            if extra_embeds.size(0) > 0:
                full_embeds = torch.cat(
                    [sample_prompt_embeds, extra_embeds],
                    dim=0,
                )

                extra_mask = torch.ones(
                    extra_embeds.size(0),
                    dtype=sample_prompt_mask.dtype,
                    device=device,
                )

                full_attention = torch.cat(
                    [sample_prompt_mask, extra_mask],
                    dim=0,
                )
            else:
                full_embeds = sample_prompt_embeds
                full_attention = sample_prompt_mask

            full_embeds_list.append(full_embeds)
            full_attention_list.append(full_attention)

        full_inputs_embeds = torch.stack(full_embeds_list, dim=0)
        full_attention_mask = torch.stack(full_attention_list, dim=0)

        # 1. Predict coarse logits from current context.
        coarse_next_logits = model_next_logits_no_cache(
            model=model,
            inputs_embeds=full_inputs_embeds,
            attention_mask=full_attention_mask,
        )

        coarse_pos = min(step_idx, len(coarse_position_token_ids) - 1)
        coarse_ids_this_pos = coarse_position_token_ids[coarse_pos]

        if len(coarse_ids_this_pos) == 0:
            raise ValueError(f"No coarse candidate tokens at position {coarse_pos}.")

        # 2. Convert predicted coarse distribution to soft latent embedding.
        coarse_emb, coarse_scores = predict_soft_coarse_embedding(
            next_logits=coarse_next_logits,
            embedding_weight=embedding_weight,
            coarse_token_ids=coarse_ids_this_pos,
            temperature=coarse_temperature,
            use_coarse_score=use_coarse_score_in_rank,
        )

        # 3. Append latent coarse embedding and predict fine logits.
        inputs_after_coarse = torch.cat(
            [full_inputs_embeds, coarse_emb.unsqueeze(1)],
            dim=1,
        )

        ones = torch.ones(
            full_attention_mask.size(0),
            1,
            dtype=full_attention_mask.dtype,
            device=device,
        )

        attention_after_coarse = torch.cat(
            [full_attention_mask, ones],
            dim=1,
        )

        fine_logits = model_next_logits_no_cache(
            model=model,
            inputs_embeds=inputs_after_coarse,
            attention_mask=attention_after_coarse,
        )

        fine_log_probs = F.log_softmax(fine_logits.float(), dim=-1)

        candidate_sample_ids = []
        candidate_scores = []
        candidate_fine_prefixes = []
        candidate_extra_embeds = []

        per_beam_expand = max(1, num_beams)

        # 4. Fine-only trie constrained beam search.
        for flat_idx, beam in enumerate(flat_beams):
            fine_prefix = beam["fine_token_ids"]
            prefix_key = tuple(fine_prefix)

            if prefix_key not in trie_cache:
                trie_cache[prefix_key] = fine_trie.get(fine_prefix)

            allowed = trie_cache[prefix_key]

            if allowed is None or len(allowed) == 0:
                continue

            allowed_tensor = torch.tensor(
                allowed,
                dtype=torch.long,
                device=device,
            )

            allowed_scores = fine_log_probs[flat_idx].index_select(
                0,
                allowed_tensor,
            )

            topk = min(per_beam_expand, allowed_tensor.size(0))
            topk_scores, topk_pos = torch.topk(allowed_scores, k=topk)
            topk_ids = allowed_tensor[topk_pos]

            base_score = float(beam["score"])
            coarse_score = float(coarse_scores[flat_idx].item())
            sample_id = flat_sample_ids[flat_idx]

            for j in range(topk):
                fine_token_id = int(topk_ids[j].item())
                fine_logprob = float(topk_scores[j].item())

                fine_emb = embedding_weight[fine_token_id].unsqueeze(0)

                new_extra_embeds = torch.cat(
                    [
                        beam["extra_embeds"],
                        coarse_emb[flat_idx].unsqueeze(0),
                        fine_emb,
                    ],
                    dim=0,
                )

                candidate_sample_ids.append(sample_id)
                candidate_scores.append(base_score + coarse_score + fine_logprob)
                candidate_fine_prefixes.append(
                    beam["fine_token_ids"] + [fine_token_id]
                )
                candidate_extra_embeds.append(new_extra_embeds)

        if len(candidate_scores) == 0:
            break

        # 5. Keep top beams for each sample.
        candidates_grouped = [[] for _ in range(batch_size)]

        for cand_idx, sample_id in enumerate(candidate_sample_ids):
            candidates_grouped[sample_id].append(cand_idx)

        new_beams_by_sample = [[] for _ in range(batch_size)]

        for sample_id in range(batch_size):
            cand_indices = candidates_grouped[sample_id]

            if len(cand_indices) == 0:
                continue

            cand_indices.sort(
                key=lambda i: candidate_scores[i],
                reverse=True,
            )

            dedup = []
            seen = set()

            for cand_idx in cand_indices:
                prefix_key = tuple(candidate_fine_prefixes[cand_idx])

                if prefix_key in seen:
                    continue

                seen.add(prefix_key)

                dedup.append({
                    "fine_token_ids": candidate_fine_prefixes[cand_idx],
                    "score": float(candidate_scores[cand_idx]),
                    "extra_embeds": candidate_extra_embeds[cand_idx],
                })

                if len(dedup) >= num_beams:
                    break

            new_beams_by_sample[sample_id] = dedup

        beams_by_sample = new_beams_by_sample

    decoded = []
    scores = []

    for sample_id in range(batch_size):
        beams = beams_by_sample[sample_id]

        if len(beams) == 0:
            beams = [{
                "fine_token_ids": [],
                "score": -1e9,
            }]

        while len(beams) < num_beams:
            beams.append({
                "fine_token_ids": list(beams[-1]["fine_token_ids"]),
                "score": float(beams[-1]["score"]),
            })

        for beam in beams[:num_beams]:
            text = tokenizer.decode(
                beam["fine_token_ids"],
                skip_special_tokens=True,
            )
            decoded.append(normalize_code_text(text))
            scores.append(float(beam["score"]))

    return decoded, scores


def run_vanilla_generate(
    model,
    tokenizer,
    inputs,
    base_prefix_fn,
    prompt_len,
    args,
):
    """
    Fallback vanilla generate path.
    Only used when passing --no_interleaved_inference.

    This is not recommended for the latent coarse-to-fine checkpoint.
    """

    def prefix_fn(batch_id, sentence):
        try:
            return base_prefix_fn(batch_id, sentence, prompt_len=prompt_len)
        except TypeError:
            return base_prefix_fn(batch_id, sentence)

    num_beams = args.num_beams

    while True:
        try:
            output = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=getattr(args, "max_new_tokens", 10),
                prefix_allowed_tokens_fn=prefix_fn,
                num_beams=num_beams,
                num_return_sequences=num_beams,
                output_scores=True,
                return_dict_in_generate=True,
                early_stopping=True,
            )
            break

        except torch.cuda.OutOfMemoryError:
            print("Out of memory!")
            num_beams -= 1
            print("Beam:", num_beams)

            if num_beams <= 0:
                raise RuntimeError("num_beams reduced to 0 due to OOM.")

        except Exception as e:
            raise RuntimeError(e)

    output_ids = output["sequences"]
    scores = output["sequences_scores"]

    gen_ids = output_ids[:, prompt_len:]
    decoded = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
    decoded = [normalize_code_text(x) for x in decoded]

    return decoded, scores


def parse_prompt_ids(args):
    if args.test_prompt_ids == "all":
        if args.test_task.lower() == "seqrec":
            return range(len(all_prompt["seqrec"]))
        elif args.test_task.lower() == "itemsearch":
            return range(len(all_prompt["itemsearch"]))
        elif args.test_task.lower() == "fusionseqrec":
            return range(len(all_prompt["fusionseqrec"]))
        else:
            raise ValueError(f"Unknown test_task: {args.test_task}")

    return [int(_) for _ in args.test_prompt_ids.split(",")]


def test_ddp(args):
    set_seed(args.seed)

    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK") or 0)

    torch.cuda.set_device(local_rank)

    if local_rank == 0:
        print(vars(args))

    dist.init_process_group(
        backend="nccl",
        world_size=world_size,
        rank=local_rank,
    )

    device_map = {"": local_rank}
    device = torch.device("cuda", local_rank)

    tokenizer, model = build_model_and_tokenizer(args, device_map=device_map)
    model.eval()

    prompt_ids = parse_prompt_ids(args)

    test_data = load_test_dataset(args)

    ddp_sampler = DistributedSampler(
        test_data,
        num_replicas=world_size,
        rank=local_rank,
        drop_last=True,
        shuffle=False,
    )

    collator = TestCollator(args, tokenizer)

    fine_all_items = build_fine_all_items(test_data)
    base_prefix_fn = test_data.get_prefix_allowed_tokens_fn(tokenizer)

    fine_trie = None
    coarse_position_token_ids = None
    fine_code_len = None

    if getattr(args, "interleaved_inference", True):
        fine_trie = build_fine_trie_from_dataset(test_data, tokenizer)
        fine_code_len = infer_fine_code_length(test_data)
        coarse_position_token_ids = build_coarse_position_token_ids(test_data, tokenizer)

        if local_rank == 0:
            print("[latent interleaved] enabled")
            print("[latent interleaved] fine_code_len:", fine_code_len)
            print("[latent interleaved] coarse positions:", len(coarse_position_token_ids))
            print("[latent interleaved] fine trie size:", len(fine_trie))
    else:
        if local_rank == 0:
            print("[latent interleaved] disabled, using vanilla generate()")

    test_loader = DataLoader(
        test_data,
        batch_size=args.test_batch_size,
        collate_fn=collator,
        sampler=ddp_sampler,
        num_workers=2,
        pin_memory=True,
    )

    if local_rank == 0:
        print("data num:", len(test_data))

    metrics = args.metrics.split(",")

    all_prompt_results = []
    all_pairs_rank0 = []
    all_txt_lines_rank0 = []

    amp_dtype = None

    if getattr(args, "use_bf16", False):
        amp_dtype = torch.bfloat16
    elif getattr(args, "use_fp16", False):
        amp_dtype = torch.float16

    with torch.no_grad():
        for prompt_id in prompt_ids:
            if local_rank == 0:
                print("Start prompt:", prompt_id)

            test_loader.dataset.set_prompt(prompt_id)

            metrics_results = {}
            total = 0
            local_pairs = []
            local_txt_lines = []

            pbar = tqdm(test_loader, disable=(local_rank != 0))

            for step, batch in enumerate(pbar):
                inputs = batch[0].to(device)
                fine_targets = batch[1]
                coarse_targets = batch[2]

                bs = len(fine_targets)
                prompt_len = inputs["input_ids"].shape[1]

                if getattr(args, "interleaved_inference", True):
                    if amp_dtype is not None:
                        with torch.cuda.amp.autocast(dtype=amp_dtype):
                            decoded, scores = batch_generate_latent_interleaved_qwen(
                                model=model,
                                tokenizer=tokenizer,
                                inputs=inputs,
                                fine_trie=fine_trie,
                                fine_code_len=fine_code_len,
                                coarse_position_token_ids=coarse_position_token_ids,
                                num_beams=args.num_beams,
                                coarse_temperature=args.interleaved_temperature,
                                use_coarse_score_in_rank=args.use_coarse_score_in_rank,
                            )
                    else:
                        decoded, scores = batch_generate_latent_interleaved_qwen(
                            model=model,
                            tokenizer=tokenizer,
                            inputs=inputs,
                            fine_trie=fine_trie,
                            fine_code_len=fine_code_len,
                            coarse_position_token_ids=coarse_position_token_ids,
                            num_beams=args.num_beams,
                            coarse_temperature=args.interleaved_temperature,
                            use_coarse_score_in_rank=args.use_coarse_score_in_rank,
                        )
                else:
                    decoded, scores = run_vanilla_generate(
                        model=model,
                        tokenizer=tokenizer,
                        inputs=inputs,
                        base_prefix_fn=base_prefix_fn,
                        prompt_len=prompt_len,
                        args=args,
                    )

                if local_rank == 0 and step < 3:
                    empty_cnt = sum(1 for x in decoded if x is None or x.strip() == "")
                    print(
                        f"[debug] empty decoded: {empty_cnt}/{len(decoded)} ; "
                        f"sample0={repr(decoded[0]) if len(decoded) > 0 else None}"
                    )

                if getattr(args, "eval_coarse_as_correct", False):
                    eval_coarse_targets = coarse_targets
                    eval_all_items = test_data.get_all_items() if args.filter_items else None
                else:
                    eval_coarse_targets = [None for _ in fine_targets]
                    eval_all_items = fine_all_items if args.filter_items else None

                topk_res, top1_pairs = get_topk_results(
                    decoded,
                    scores,
                    fine_targets,
                    eval_coarse_targets,
                    args.num_beams,
                    all_items=eval_all_items,
                )

                if args.save_simple_results:
                    for pair_idx, pair in enumerate(top1_pairs):
                        fine_t = normalize_code_text(fine_targets[pair_idx])
                        coarse_t = normalize_code_text(coarse_targets[pair_idx])
                        pred = normalize_code_text(pair["pred"])

                        local_txt_lines.append(f"{fine_t} | {coarse_t}")
                        local_txt_lines.append(f"{pred}")

                local_pairs.extend(top1_pairs)

                bs_gather_list = [None for _ in range(world_size)]
                dist.all_gather_object(obj=bs, object_list=bs_gather_list)
                total += sum(bs_gather_list)

                res_gather_list = [None for _ in range(world_size)]
                dist.all_gather_object(obj=topk_res, object_list=res_gather_list)

                if local_rank == 0:
                    all_device_topk_res = []

                    for ga_res in res_gather_list:
                        all_device_topk_res += ga_res

                    batch_metrics_res = get_metrics_results(
                        all_device_topk_res,
                        metrics,
                    )

                    for m, res in batch_metrics_res.items():
                        metrics_results[m] = metrics_results.get(m, 0) + res

                    if total > 0:
                        temp = {
                            m: metrics_results[m] / total
                            for m in metrics_results
                        }
                        pbar.set_postfix(temp)

                    if (step + 1) % 50 == 0 and total > 0:
                        temp = {
                            m: metrics_results[m] / total
                            for m in metrics_results
                        }
                        print(temp)

                dist.barrier()

            gathered_pairs = gather_object_list(local_pairs, world_size)
            gathered_txt_lines = gather_object_list(local_txt_lines, world_size)

            if local_rank == 0:
                all_pairs_rank0.extend(gathered_pairs)

                if args.save_simple_results:
                    all_txt_lines_rank0.extend(gathered_txt_lines)

                for m in metrics_results:
                    metrics_results[m] = metrics_results[m] / total if total > 0 else 0.0

                all_prompt_results.append(metrics_results)

                print("======================================================")
                print("Prompt {} results: ".format(prompt_id), metrics_results)
                print("======================================================")
                print("")

            dist.barrier()

    dist.barrier()

    if local_rank == 0:
        mean_results = {}
        min_results = {}
        max_results = {}

        for m in metrics:
            all_res = [_[m] for _ in all_prompt_results]
            mean_results[m] = sum(all_res) / len(all_res) if len(all_res) > 0 else 0.0
            min_results[m] = min(all_res) if len(all_res) > 0 else 0.0
            max_results[m] = max(all_res) if len(all_res) > 0 else 0.0

        print("======================================================")
        print("Mean results:", mean_results)
        print("Min results:", min_results)
        print("Max results:", max_results)
        print("======================================================")

        save_data = {
            "test_prompt_ids": args.test_prompt_ids,
            "interleaved_inference": args.interleaved_inference,
            "interleaved_temperature": args.interleaved_temperature,
            "use_coarse_score_in_rank": args.use_coarse_score_in_rank,
            "eval_coarse_as_correct": args.eval_coarse_as_correct,
            "mean_results": mean_results,
            "min_results": min_results,
            "max_results": max_results,
            "all_prompt_results": all_prompt_results,
        }

        results_dir = os.path.dirname(args.results_file)

        if results_dir != "":
            os.makedirs(results_dir, exist_ok=True)

        with open(args.results_file, "w", encoding="utf-8") as f:
            json.dump(save_data, f, indent=4, ensure_ascii=False)

        print("Save file:", args.results_file)

        if args.save_pairs_json:
            pairs_dir = os.path.dirname(args.pairs_json_file)

            if pairs_dir != "":
                os.makedirs(pairs_dir, exist_ok=True)

            with open(args.pairs_json_file, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "ckpt_path": args.ckpt_path,
                        "test_prompt_ids": args.test_prompt_ids,
                        "metrics": args.metrics,
                        "interleaved_inference": args.interleaved_inference,
                        "num_pairs": len(all_pairs_rank0),
                        "pairs": all_pairs_rank0,
                    },
                    f,
                    ensure_ascii=False,
                    indent=2,
                )

            print("Save pairs json to:", args.pairs_json_file)

        if args.save_simple_results:
            txt_dir = os.path.dirname(args.simple_results_file)

            if txt_dir != "":
                os.makedirs(txt_dir, exist_ok=True)

            with open(args.simple_results_file, "w", encoding="utf-8") as f:
                for line in all_txt_lines_rank0:
                    f.write(line + "\n")

            print("Save simple txt to:", args.simple_results_file)

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="HoloRec-Qwen latent coarse-to-fine interleaved inference"
    )

    parser = parse_global_args(parser)
    parser = parse_dataset_args(parser)
    parser = parse_test_args(parser)

    parser.add_argument(
        "--coarse_index_file",
        type=str,
        default=".tw8.json",
        help="Coarse code index file, e.g. .tw8.json",
    )

    parser.add_argument(
        "--interleaved_inference",
        dest="interleaved_inference",
        action="store_true",
        default=True,
        help="Enable latent coarse-to-fine interleaved inference. Default: enabled.",
    )

    parser.add_argument(
        "--no_interleaved_inference",
        dest="interleaved_inference",
        action="store_false",
        help="Disable interleaved inference and use original generate() path.",
    )

    parser.add_argument(
        "--interleaved_temperature",
        type=float,
        default=0.8,
        help="Temperature for predicted coarse soft distribution.",
    )

    parser.add_argument(
        "--use_coarse_score_in_rank",
        action="store_true",
        help=(
            "Add coarse max log-prob to beam score. "
            "Default off: coarse only acts as latent reasoning signal."
        ),
    )

    parser.add_argument(
        "--eval_coarse_as_correct",
        action="store_true",
        help=(
            "Whether to treat coarse target as correct during evaluation. "
            "For fine-only latent decoding, default False is recommended."
        ),
    )

    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=10,
        help="Only used by vanilla generate() path.",
    )

    parser.add_argument(
        "--load_in_8bit",
        action="store_true",
        default=False,
        help="Load non-LoRA checkpoint with 8bit quantization.",
    )

    parser.add_argument(
        "--use_fp16",
        action="store_true",
        help="Use fp16 autocast in inference.",
    )

    parser.add_argument(
        "--use_bf16",
        action="store_true",
        help="Use bf16 autocast in inference.",
    )

    parser.add_argument(
        "--save_pairs_json",
        action="store_true",
    )

    parser.add_argument(
        "--pairs_json_file",
        type=str,
        default="./results/pairs.json",
    )

    parser.add_argument(
        "--save_simple_results",
        action="store_true",
    )

    parser.add_argument(
        "--simple_results_file",
        type=str,
        default="./results/simple_results.txt",
    )

    args = parser.parse_args()
    test_ddp(args)
