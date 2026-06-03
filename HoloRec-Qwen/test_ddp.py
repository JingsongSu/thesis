import argparse
import json
import os
from typing import Dict, List, Sequence, Tuple

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

from utils import *
from collator import TestCollator
from prompt import all_prompt
from evaluate import get_topk_results, get_metrics_results
from generation_trie import Trie, build_trie_from_token_sequences


def build_model_and_tokenizer(args, device_map):
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

    model_kwargs = dict(
        device_map=device_map,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )

    if getattr(args, "attn_implementation", None):
        model_kwargs["attn_implementation"] = args.attn_implementation

    if args.lora:
        base = AutoModelForCausalLM.from_pretrained(
            args.base_model,
            **model_kwargs,
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
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False,
            )
        model = AutoModelForCausalLM.from_pretrained(
            args.ckpt_path,
            **model_kwargs,
        )

    if hasattr(model.config, "use_cache"):
        model.config.use_cache = True

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
        raise ValueError(f"Token {token!r} is not a single tokenizer id: {encoded}")

    unk_id = getattr(tokenizer, "unk_token_id", None)
    if unk_id is not None and token_id == unk_id:
        encoded = tokenizer(token, add_special_tokens=False).get("input_ids", [])
        if len(encoded) == 1:
            return int(encoded[0])
        raise ValueError(f"Token {token!r} is mapped to unk or multiple ids: {encoded}")

    return int(token_id)


def infer_fine_code_length(test_data) -> int:
    if hasattr(test_data, "indices") and len(test_data.indices) > 0:
        first_key = next(iter(test_data.indices.keys()))
        return len(test_data.indices[first_key])

    raise ValueError("Cannot infer fine code length from test dataset.")


def build_fine_trie_from_dataset(test_data, tokenizer) -> Trie:
    if not hasattr(test_data, "indices"):
        raise ValueError("test_data must have `indices` for fine trie construction.")

    return build_trie_from_token_sequences(
        list(test_data.indices.values()),
        tokenizer,
    )


def build_fine_all_items(test_data) -> set:
    if not hasattr(test_data, "indices"):
        return set()

    return set(code_tokens_to_text(v) for v in test_data.indices.values())


def build_coarse_position_token_ids(test_data, tokenizer) -> List[List[int]]:
    if not hasattr(test_data, "coarse_indices") or test_data.coarse_indices is None:
        raise ValueError(
            "Latent interleaved inference requires --coarse_index_file, for example .tw8.json"
        )

    pos_to_ids = {}

    for _, seq in test_data.coarse_indices.items():
        for pos, tok in enumerate(seq):
            pos_to_ids.setdefault(pos, set()).add(token_to_id(tokenizer, tok))

    if len(pos_to_ids) == 0:
        raise ValueError("coarse position token ids are empty.")

    return [
        sorted(list(pos_to_ids.get(i, set())))
        for i in range(max(pos_to_ids.keys()) + 1)
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


def _as_legacy_cache(past_key_values):
    if past_key_values is None:
        return None
    if hasattr(past_key_values, "to_legacy_cache"):
        return past_key_values.to_legacy_cache()
    return past_key_values


def _slice_past(past_key_values, index: int):
    past_key_values = _as_legacy_cache(past_key_values)

    return tuple(
        tuple(x[index:index + 1].contiguous() for x in layer)
        for layer in past_key_values
    )


def _select_past(past_key_values, indices: Sequence[int]):
    past_key_values = _as_legacy_cache(past_key_values)

    if len(indices) == 1:
        return _slice_past(past_key_values, int(indices[0]))

    device = past_key_values[0][0].device
    idx = torch.tensor([int(i) for i in indices], dtype=torch.long, device=device)

    return tuple(
        tuple(x.index_select(0, idx).contiguous() for x in layer)
        for layer in past_key_values
    )


def _concat_pasts(pasts: Sequence[Tuple[Tuple[torch.Tensor, ...], ...]]):
    if len(pasts) == 1:
        return pasts[0]

    layers = []

    for layer_idx in range(len(pasts[0])):
        parts = []
        for kv_idx in range(len(pasts[0][layer_idx])):
            parts.append(
                torch.cat(
                    [p[layer_idx][kv_idx] for p in pasts],
                    dim=0,
                ).contiguous()
            )
        layers.append(tuple(parts))

    return tuple(layers)


def _model_forward_cached(
    model,
    *,
    inputs_embeds,
    attention_mask,
    past_key_values,
):
    out = model(
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        past_key_values=past_key_values,
        use_cache=True,
        return_dict=True,
        output_attentions=False,
        output_hidden_states=False,
    )
    out.past_key_values = _as_legacy_cache(out.past_key_values)
    return out


def _prompt_forward(model, input_ids, attention_mask):
    out = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=True,
        return_dict=True,
        output_attentions=False,
        output_hidden_states=False,
    )
    out.past_key_values = _as_legacy_cache(out.past_key_values)
    return out


def predict_soft_coarse_embedding(
    next_logits,
    embedding_weight,
    coarse_token_ids,
    temperature,
    use_coarse_score,
):
    ids = torch.tensor(
        coarse_token_ids,
        dtype=torch.long,
        device=next_logits.device,
    )

    candidate_logits = next_logits.index_select(dim=-1, index=ids)

    log_probs = F.log_softmax(candidate_logits.float(), dim=-1)
    probs = F.softmax(
        candidate_logits.float() / max(float(temperature), 1e-6),
        dim=-1,
    ).to(embedding_weight.dtype)

    emb = probs @ embedding_weight.index_select(dim=0, index=ids).to(probs.dtype)

    if use_coarse_score:
        score = torch.sum(probs.float() * log_probs, dim=-1)
    else:
        score = torch.zeros(
            next_logits.size(0),
            dtype=torch.float32,
            device=next_logits.device,
        )

    return emb, score


def batch_generate_latent_interleaved_qwen(
    model,
    tokenizer,
    inputs,
    fine_trie: Trie,
    fine_code_len: int,
    coarse_position_token_ids: List[List[int]],
    num_beams: int,
    coarse_temperature: float = 0.8,
    use_coarse_score_in_rank: bool = False,
):
    """
    Fast implicit interleaved inference.

    Visible output contains fine tokens only.
    Hidden path for each fine position:
        previous state -> soft coarse latent embedding -> fine logits -> fine token
    """

    device = inputs["input_ids"].device
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    batch_size = input_ids.size(0)

    embedding_weight = get_model_input_embeddings(model).weight

    prompt_out = _prompt_forward(
        model=model,
        input_ids=input_ids,
        attention_mask=attention_mask,
    )

    prompt_past = prompt_out.past_key_values
    prompt_next_logits = prompt_out.logits[:, -1, :]

    beams_by_sample: List[List[Dict]] = []

    for b in range(batch_size):
        beams_by_sample.append(
            [
                {
                    "fine_token_ids": [],
                    "score": 0.0,
                    "past": _slice_past(prompt_past, b),
                    "attention_mask": attention_mask[b:b + 1].contiguous(),
                    "next_logits": prompt_next_logits[b:b + 1].contiguous(),
                }
            ]
        )

    trie_cache: Dict[tuple, List[int]] = {}
    per_beam_expand = max(1, int(num_beams))

    for step_idx in range(fine_code_len):
        flat_beams = []
        flat_sample_ids = []

        for sample_id, beams in enumerate(beams_by_sample):
            for beam in beams:
                flat_beams.append(beam)
                flat_sample_ids.append(sample_id)

        if len(flat_beams) == 0:
            break

        next_logits = torch.cat(
            [b["next_logits"] for b in flat_beams],
            dim=0,
        )

        batched_past = _concat_pasts(
            [b["past"] for b in flat_beams],
        )

        batched_attention = torch.cat(
            [b["attention_mask"] for b in flat_beams],
            dim=0,
        )

        coarse_pos = min(step_idx, len(coarse_position_token_ids) - 1)
        coarse_ids_this_pos = coarse_position_token_ids[coarse_pos]

        coarse_emb, coarse_scores = predict_soft_coarse_embedding(
            next_logits=next_logits,
            embedding_weight=embedding_weight,
            coarse_token_ids=coarse_ids_this_pos,
            temperature=coarse_temperature,
            use_coarse_score=use_coarse_score_in_rank,
        )

        ones = torch.ones(
            (len(flat_beams), 1),
            dtype=batched_attention.dtype,
            device=device,
        )

        attention_after_coarse = torch.cat(
            [batched_attention, ones],
            dim=1,
        )

        out_after_coarse = _model_forward_cached(
            model,
            inputs_embeds=coarse_emb.unsqueeze(1),
            attention_mask=attention_after_coarse,
            past_key_values=batched_past,
        )

        past_after_coarse = out_after_coarse.past_key_values
        fine_logits = out_after_coarse.logits[:, -1, :]
        fine_log_probs = F.log_softmax(fine_logits.float(), dim=-1)

        raw_candidates = []

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

            topk = min(per_beam_expand, allowed_tensor.numel())
            topk_scores, topk_pos = torch.topk(allowed_scores, k=topk)
            topk_ids = allowed_tensor[topk_pos]

            base_score = float(beam["score"])
            coarse_score = float(coarse_scores[flat_idx].item())
            sample_id = flat_sample_ids[flat_idx]

            for j in range(topk):
                fine_token_id = int(topk_ids[j].item())

                raw_candidates.append(
                    {
                        "sample_id": sample_id,
                        "from_flat_idx": flat_idx,
                        "fine_token_id": fine_token_id,
                        "fine_token_ids": fine_prefix + [fine_token_id],
                        "score": base_score + coarse_score + float(topk_scores[j].item()),
                        "attention_after_coarse": attention_after_coarse[
                            flat_idx:flat_idx + 1
                        ],
                    }
                )

        if len(raw_candidates) == 0:
            break

        grouped = [[] for _ in range(batch_size)]

        for i, cand in enumerate(raw_candidates):
            grouped[cand["sample_id"]].append(i)

        survivor_indices = []
        new_beams_by_sample = [[] for _ in range(batch_size)]

        for sample_id in range(batch_size):
            cand_indices = grouped[sample_id]
            cand_indices.sort(
                key=lambda i: raw_candidates[i]["score"],
                reverse=True,
            )

            seen = set()

            for i in cand_indices:
                key = tuple(raw_candidates[i]["fine_token_ids"])
                if key in seen:
                    continue

                seen.add(key)
                survivor_indices.append(i)

                if len(seen) >= num_beams:
                    break

        if len(survivor_indices) == 0:
            break

        fine_token_ids = torch.tensor(
            [raw_candidates[i]["fine_token_id"] for i in survivor_indices],
            dtype=torch.long,
            device=device,
        )

        fine_emb = embedding_weight.index_select(
            0,
            fine_token_ids,
        ).unsqueeze(1)

        survivor_flat_indices = [
            raw_candidates[i]["from_flat_idx"]
            for i in survivor_indices
        ]

        survivor_past_after_coarse = _select_past(
            past_after_coarse,
            survivor_flat_indices,
        )

        survivor_attention_after_coarse = torch.cat(
            [
                raw_candidates[i]["attention_after_coarse"]
                for i in survivor_indices
            ],
            dim=0,
        )

        ones2 = torch.ones(
            (len(survivor_indices), 1),
            dtype=survivor_attention_after_coarse.dtype,
            device=device,
        )

        attention_after_fine = torch.cat(
            [survivor_attention_after_coarse, ones2],
            dim=1,
        )

        out_after_fine = _model_forward_cached(
            model,
            inputs_embeds=fine_emb,
            attention_mask=attention_after_fine,
            past_key_values=survivor_past_after_coarse,
        )

        past_after_fine = out_after_fine.past_key_values
        next_logits_after_fine = out_after_fine.logits[:, -1, :]

        for new_idx, cand_idx in enumerate(survivor_indices):
            cand = raw_candidates[cand_idx]
            sample_id = cand["sample_id"]

            new_beams_by_sample[sample_id].append(
                {
                    "fine_token_ids": cand["fine_token_ids"],
                    "score": float(cand["score"]),
                    "past": _slice_past(past_after_fine, new_idx),
                    "attention_mask": attention_after_fine[
                        new_idx:new_idx + 1
                    ].contiguous(),
                    "next_logits": next_logits_after_fine[
                        new_idx:new_idx + 1
                    ].contiguous(),
                }
            )

        beams_by_sample = new_beams_by_sample

    decoded = []
    scores = []

    for sample_id in range(batch_size):
        beams = beams_by_sample[sample_id]

        if len(beams) == 0:
            beams = [
                {
                    "fine_token_ids": [],
                    "score": -1e9,
                }
            ]

        while len(beams) < num_beams:
            beams.append(
                {
                    "fine_token_ids": list(beams[-1]["fine_token_ids"]),
                    "score": float(beams[-1]["score"]),
                }
            )

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
    def prefix_fn(batch_id, sentence):
        try:
            return base_prefix_fn(
                batch_id,
                sentence,
                prompt_len=prompt_len,
            )
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
                use_cache=True,
            )
            break
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            num_beams -= 1
            print("Out of memory! Beam:", num_beams)
            if num_beams <= 0:
                raise RuntimeError("num_beams reduced to 0 due to OOM.")

    output_ids = output["sequences"]
    scores = output["sequences_scores"]
    gen_ids = output_ids[:, prompt_len:]
    decoded = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)

    return [normalize_code_text(x) for x in decoded], scores


def parse_prompt_ids(args):
    if args.test_prompt_ids == "all":
        task = args.test_task.lower()

        if task == "seqrec":
            return range(len(all_prompt["seqrec"]))

        if task == "itemsearch":
            return range(len(all_prompt["itemsearch"]))

        if task == "fusionseqrec":
            return range(len(all_prompt["fusionseqrec"]))

        raise ValueError(f"Unknown test_task: {args.test_task}")

    return [int(x) for x in args.test_prompt_ids.split(",")]


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

    tokenizer, model = build_model_and_tokenizer(
        args,
        device_map=device_map,
    )

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
            print("[latent interleaved fast-cache] enabled")
            print("[latent interleaved fast-cache] fine_code_len:", fine_code_len)
            print("[latent interleaved fast-cache] coarse positions:", len(coarse_position_token_ids))
            print("[latent interleaved fast-cache] fine trie size:", len(fine_trie))
    else:
        if local_rank == 0:
            print("[latent interleaved] disabled, using vanilla generate()")

    test_loader = DataLoader(
        test_data,
        batch_size=args.test_batch_size,
        collate_fn=collator,
        sampler=ddp_sampler,
        num_workers=getattr(args, "num_workers", 2),
        pin_memory=True,
    )

    if local_rank == 0:
        print("data num:", len(test_data))

    metrics = args.metrics.split(",")

    all_prompt_results = []
    all_pairs_rank0 = []
    all_txt_lines_rank0 = []

    if getattr(args, "use_bf16", False):
        amp_dtype = torch.bfloat16
    elif getattr(args, "use_fp16", False):
        amp_dtype = torch.float16
    else:
        amp_dtype = None

    with torch.inference_mode():
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
                    empty_cnt = sum(
                        1
                        for x in decoded
                        if x is None or x.strip() == ""
                    )
                    sample0 = repr(decoded[0]) if decoded else None
                    print(
                        f"[debug] empty decoded: {empty_cnt}/{len(decoded)} ; sample0={sample0}"
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
                dist.all_gather_object(
                    object_list=bs_gather_list,
                    obj=bs,
                )
                total += sum(bs_gather_list)

                res_gather_list = [None for _ in range(world_size)]
                dist.all_gather_object(
                    object_list=res_gather_list,
                    obj=topk_res,
                )

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
                        print(
                            {
                                m: metrics_results[m] / total
                                for m in metrics_results
                            }
                        )

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
                print("======================================================\n")

            dist.barrier()

    if local_rank == 0:
        mean_results = {}
        min_results = {}
        max_results = {}

        for m in metrics:
            all_res = [x[m] for x in all_prompt_results]
            mean_results[m] = sum(all_res) / len(all_res) if all_res else 0.0
            min_results[m] = min(all_res) if all_res else 0.0
            max_results[m] = max(all_res) if all_res else 0.0

        result = {
            "all_prompt_results": all_prompt_results,
            "mean_results": mean_results,
            "min_results": min_results,
            "max_results": max_results,
            "top1_pairs": all_pairs_rank0,
        }

        result_dir = os.path.dirname(args.results_file)
        if result_dir:
            os.makedirs(result_dir, exist_ok=True)

        with open(args.results_file, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        print("Mean results:", mean_results)
        print("Min results:", min_results)
        print("Max results:", max_results)
        print("Saved results to:", args.results_file)

        if args.save_simple_results:
            simple_file = args.results_file + ".txt"
            with open(simple_file, "w", encoding="utf-8") as f:
                f.write("\n".join(all_txt_lines_rank0))
            print("Saved simple results to:", simple_file)

    dist.barrier()
    dist.destroy_process_group()


def add_interleaved_test_args(parser):
    parser.add_argument(
        "--coarse_index_file",
        type=str,
        default=".tw8.json",
    )

    parser.add_argument(
        "--interleaved_temperature",
        type=float,
        default=0.8,
    )

    parser.add_argument(
        "--interleaved_inference",
        action="store_true",
        default=True,
    )

    parser.add_argument(
        "--no_interleaved_inference",
        dest="interleaved_inference",
        action="store_false",
    )

    parser.add_argument(
        "--use_coarse_score_in_rank",
        action="store_true",
        default=False,
    )

    parser.add_argument(
        "--eval_coarse_as_correct",
        action="store_true",
        default=False,
    )

    parser.add_argument(
        "--save_simple_results",
        action="store_true",
        default=False,
    )

    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--use_bf16",
        action="store_true",
        default=True,
    )

    parser.add_argument(
        "--use_fp16",
        action="store_true",
        default=False,
    )

    parser.add_argument(
        "--load_in_8bit",
        action="store_true",
        default=False,
    )

    parser.add_argument(
        "--attn_implementation",
        type=str,
        default="flash_attention_2",
    )

    parser.add_argument(
        "--num_workers",
        type=int,
        default=2,
    )

    return parser


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fast latent interleaved HoloRec-Qwen DDP test"
    )

    parser = parse_global_args(parser)
    parser = parse_dataset_args(parser)
    parser = parse_test_args(parser)
    parser = add_interleaved_test_args(parser)

    args = parser.parse_args()
    test_ddp(args)
