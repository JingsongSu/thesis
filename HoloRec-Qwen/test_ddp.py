import argparse
import json
import os
from typing import Dict, List

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

    # This fast inference path does not need KV cache.
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


def _flatten_tokenizer_ids(encoded):
    ids = encoded.get("input_ids", encoded)

    if isinstance(ids, torch.Tensor):
        ids = ids.tolist()

    if len(ids) > 0 and isinstance(ids[0], list):
        ids = ids[0]

    return [int(x) for x in ids]


def token_to_id(tokenizer, token: str) -> int:
    token_id = tokenizer.convert_tokens_to_ids(token)

    if token_id is None:
        encoded = tokenizer(token, add_special_tokens=False)
        ids = _flatten_tokenizer_ids(encoded)

        if len(ids) == 1:
            return int(ids[0])

        raise ValueError(f"Token {token!r} is not a single tokenizer id: {ids}")

    unk_id = getattr(tokenizer, "unk_token_id", None)

    if unk_id is not None and int(token_id) == int(unk_id):
        encoded = tokenizer(token, add_special_tokens=False)
        ids = _flatten_tokenizer_ids(encoded)

        if len(ids) == 1:
            return int(ids[0])

        raise ValueError(f"Token {token!r} is mapped to unk or multiple ids: {ids}")

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
            "Latent inference requires --coarse_index_file, for example .tw8.json"
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


def _to_list(x):
    if x is None:
        return []
    if isinstance(x, (list, tuple, set)):
        return list(x)
    return [x]


def unwrap_modules_to_save_embedding(module):
    """
    PEFT modules_to_save may wrap embed_tokens or lm_head.
    Return the real module when possible.
    """
    if module is None:
        raise AttributeError("Module is None.")

    if hasattr(module, "weight") and callable(getattr(module, "forward", None)):
        return module

    modules_to_save = getattr(module, "modules_to_save", None)

    if modules_to_save is not None:
        active_names = []

        if hasattr(module, "active_adapter"):
            active_names.extend(_to_list(getattr(module, "active_adapter")))

        if hasattr(module, "active_adapters"):
            active_adapters = getattr(module, "active_adapters")
            if callable(active_adapters):
                try:
                    active_adapters = active_adapters()
                except TypeError:
                    active_adapters = None
            active_names.extend(_to_list(active_adapters))

        if hasattr(module, "_active_adapter"):
            active_names.extend(_to_list(getattr(module, "_active_adapter")))

        active_names.append("default")

        seen = set()
        clean_names = []

        for name in active_names:
            if name is None:
                continue
            name = str(name)
            if name in seen:
                continue
            seen.add(name)
            clean_names.append(name)

        for name in clean_names:
            if name in modules_to_save:
                candidate = modules_to_save[name]
                if hasattr(candidate, "weight"):
                    return candidate

        for _, candidate in modules_to_save.items():
            if hasattr(candidate, "weight"):
                return candidate

    original_module = getattr(module, "original_module", None)

    if original_module is not None and hasattr(original_module, "weight"):
        return original_module

    child = getattr(module, "module", None)

    if child is not None and child is not module:
        return unwrap_modules_to_save_embedding(child)

    raise AttributeError(
        "Cannot find real module weight. "
        f"Got module type: {type(module).__name__}."
    )


def get_model_input_embeddings(model):
    """
    Compatible with normal HF model and PEFT / LoRA model.
    Return the real input embedding module.
    """
    if hasattr(model, "get_input_embeddings"):
        emb = model.get_input_embeddings()
        if emb is not None:
            return unwrap_modules_to_save_embedding(emb)

    if hasattr(model, "base_model") and hasattr(model.base_model, "get_input_embeddings"):
        emb = model.base_model.get_input_embeddings()
        if emb is not None:
            return unwrap_modules_to_save_embedding(emb)

    if hasattr(model, "model") and hasattr(model.model, "embed_tokens"):
        return unwrap_modules_to_save_embedding(model.model.embed_tokens)

    raise ValueError("Cannot locate model input embeddings.")


def get_model_output_embeddings(model):
    """
    Compatible with normal HF model and PEFT / LoRA model.
    Return lm_head / output embeddings module.
    """
    if hasattr(model, "get_output_embeddings"):
        lm_head = model.get_output_embeddings()
        if lm_head is not None:
            return lm_head

    if hasattr(model, "lm_head"):
        return model.lm_head

    if hasattr(model, "base_model") and hasattr(model.base_model, "lm_head"):
        return model.base_model.lm_head

    if hasattr(model, "model") and hasattr(model.model, "lm_head"):
        return model.model.lm_head

    raise ValueError("Cannot locate model output embeddings / lm_head.")


def model_next_logits_and_hidden_no_cache(model, inputs_embeds, attention_mask):
    """
    One no-cache forward.

    Returns:
        next_logits: [B, V]
        next_hidden: [B, H], hidden state fed into lm_head at the last position.

    This is used by the fast latent formula:
        hidden -> coarse soft emb
        hidden + coarse soft emb -> lm_head -> fine logits
    """
    lm_head = get_model_output_embeddings(model)
    captured = {}

    def _pre_hook(module, inputs):
        if len(inputs) > 0:
            captured["hidden"] = inputs[0]

    handle = lm_head.register_forward_pre_hook(_pre_hook)

    try:
        outputs = model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            use_cache=False,
            return_dict=True,
            output_attentions=False,
            output_hidden_states=False,
        )
    finally:
        handle.remove()

    if "hidden" not in captured:
        raise RuntimeError(
            "Failed to capture lm_head input hidden states. "
            "Please check get_model_output_embeddings()."
        )

    next_logits = outputs.logits[:, -1, :]
    next_hidden = captured["hidden"][:, -1, :]

    return next_logits, next_hidden


def latent_fused_fine_logits(
    model,
    context_hidden,
    coarse_embeds,
    coarse_fusion_scale=1.0,
):
    """
    Fast coarse-to-fine scoring:

        fused_hidden = context_hidden + coarse_fusion_scale * coarse_emb
        fine_logits = lm_head(fused_hidden)
    """
    lm_head = get_model_output_embeddings(model)
    fused_hidden = context_hidden + float(coarse_fusion_scale) * coarse_embeds
    return lm_head(fused_hidden)


def predict_soft_coarse_embedding(
    next_logits,
    embedding_weight,
    coarse_token_ids,
    temperature,
    use_coarse_score,
):
    """
    coarse logits -> soft coarse embedding.

    coarse is not decoded as text.
    coarse is restricted to the current position's coarse codebook.
    """
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
    ).to(dtype=embedding_weight.dtype)

    candidate_embeds = embedding_weight.index_select(dim=0, index=ids)
    soft_coarse_emb = probs @ candidate_embeds

    if use_coarse_score:
        coarse_score = torch.sum(probs.float() * log_probs, dim=-1)
    else:
        coarse_score = torch.zeros(
            next_logits.size(0),
            dtype=torch.float32,
            device=next_logits.device,
        )

    return soft_coarse_emb, coarse_score


@torch.no_grad()
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
    coarse_fusion_scale: float = 1.0,
):
    """
    Fast latent inference, matching the fast training formula.

    Each fine code position:
        context -> Qwen once -> next logits + hidden
        next logits -> soft coarse embedding
        hidden + soft coarse embedding -> lm_head -> fine logits
        trie-constrained beam selects fine token
        append selected fine embedding only

    Important:
        - coarse embedding is predicted, not gold.
        - coarse embedding is not decoded.
        - coarse embedding is not appended to next-step context.
        - no second Qwen forward after coarse embedding.
    """
    device = inputs["input_ids"].device
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    batch_size = input_ids.size(0)

    embedding_module = get_model_input_embeddings(model)
    embedding_weight = embedding_module.weight
    prompt_embeds = embedding_module(input_ids)
    hidden_size = embedding_weight.size(-1)

    beams_by_sample: List[List[Dict]] = []

    for _ in range(batch_size):
        beams_by_sample.append(
            [
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

                extra_attention = torch.ones(
                    extra_embeds.size(0),
                    dtype=sample_prompt_mask.dtype,
                    device=device,
                )

                full_attention = torch.cat(
                    [sample_prompt_mask, extra_attention],
                    dim=0,
                )
            else:
                full_embeds = sample_prompt_embeds
                full_attention = sample_prompt_mask

            full_embeds_list.append(full_embeds)
            full_attention_list.append(full_attention)

        full_inputs_embeds = torch.stack(full_embeds_list, dim=0)
        full_attention_mask = torch.stack(full_attention_list, dim=0)

        # 1. Current context predicts coarse logits and provides context hidden.
        coarse_logits, context_hidden = model_next_logits_and_hidden_no_cache(
            model=model,
            inputs_embeds=full_inputs_embeds,
            attention_mask=full_attention_mask,
        )

        # 2. Coarse logits -> soft coarse embedding.
        coarse_pos = min(step_idx, len(coarse_position_token_ids) - 1)
        coarse_ids_this_pos = coarse_position_token_ids[coarse_pos]

        if len(coarse_ids_this_pos) == 0:
            raise ValueError(f"Empty coarse codebook at position {coarse_pos}.")

        coarse_embeds, coarse_scores = predict_soft_coarse_embedding(
            next_logits=coarse_logits,
            embedding_weight=embedding_weight,
            coarse_token_ids=coarse_ids_this_pos,
            temperature=coarse_temperature,
            use_coarse_score=use_coarse_score_in_rank,
        )

        # 3. Fast coarse-to-fine path:
        #    hidden + coarse_emb -> lm_head -> fine logits.
        fine_logits = latent_fused_fine_logits(
            model=model,
            context_hidden=context_hidden,
            coarse_embeds=coarse_embeds,
            coarse_fusion_scale=coarse_fusion_scale,
        )

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
                dim=0,
                index=allowed_tensor,
            )

            topk = min(per_beam_expand, allowed_tensor.numel())

            topk_scores, topk_pos = torch.topk(
                allowed_scores,
                k=topk,
            )

            topk_ids = allowed_tensor[topk_pos]

            base_score = float(beam["score"])
            coarse_score = float(coarse_scores[flat_idx].item())
            sample_id = flat_sample_ids[flat_idx]

            for j in range(topk):
                fine_token_id = int(topk_ids[j].item())
                fine_token_score = float(topk_scores[j].item())

                fine_emb = embedding_weight[fine_token_id].unsqueeze(0)

                # Fast version appends only selected fine embedding.
                # Coarse embedding is latent and local to this step.
                new_extra_embeds = torch.cat(
                    [
                        beam["extra_embeds"],
                        fine_emb,
                    ],
                    dim=0,
                )

                raw_candidates.append(
                    {
                        "sample_id": sample_id,
                        "fine_token_ids": fine_prefix + [fine_token_id],
                        "score": base_score + coarse_score + fine_token_score,
                        "extra_embeds": new_extra_embeds,
                    }
                )

        if len(raw_candidates) == 0:
            break

        grouped = [[] for _ in range(batch_size)]

        for cand_idx, cand in enumerate(raw_candidates):
            grouped[cand["sample_id"]].append(cand_idx)

        new_beams_by_sample = [[] for _ in range(batch_size)]

        for sample_id in range(batch_size):
            cand_indices = grouped[sample_id]

            cand_indices.sort(
                key=lambda i: raw_candidates[i]["score"],
                reverse=True,
            )

            seen = set()

            for cand_idx in cand_indices:
                key = tuple(raw_candidates[cand_idx]["fine_token_ids"])

                if key in seen:
                    continue

                seen.add(key)

                new_beams_by_sample[sample_id].append(
                    {
                        "fine_token_ids": raw_candidates[cand_idx]["fine_token_ids"],
                        "score": float(raw_candidates[cand_idx]["score"]),
                        "extra_embeds": raw_candidates[cand_idx]["extra_embeds"],
                    }
                )

                if len(new_beams_by_sample[sample_id]) >= num_beams:
                    break

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
            print("[fast latent inference] enabled")
            print("[fast latent inference] fine_code_len:", fine_code_len)
            print("[fast latent inference] coarse positions:", len(coarse_position_token_ids))
            print("[fast latent inference] fine trie size:", len(fine_trie))
            print("[fast latent inference] coarse_fusion_scale:", args.coarse_fusion_scale)
    else:
        if local_rank == 0:
            print("[latent inference] disabled, using vanilla generate()")

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
                                coarse_fusion_scale=args.coarse_fusion_scale,
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
                            coarse_fusion_scale=args.coarse_fusion_scale,
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
                        1 for x in decoded if x is None or x.strip() == ""
                    )
                    sample0 = repr(decoded[0]) if decoded else None

                    print(
                        f"[debug] empty decoded: {empty_cnt}/{len(decoded)} ; "
                        f"sample0={sample0}"
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

                    if (step + 1) % 50 == 0:
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

    parser.add_argument(
        "--coarse_fusion_scale",
        type=float,
        default=1.0,
        help=(
            "Fusion scale for fast latent inference: "
            "fine_logits = lm_head(hidden + scale * coarse_soft_embedding). "
            "Use the same value as training."
        ),
    )

    return parser


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fast latent HoloRec-Qwen DDP test"
    )

    parser = parse_global_args(parser)
    parser = parse_dataset_args(parser)
    parser = parse_test_args(parser)
    parser = add_interleaved_test_args(parser)

    args = parser.parse_args()

    test_ddp(args)