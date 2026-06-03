import argparse
import json
import os
from typing import Dict, List, Optional, Tuple, Any

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
        if getattr(args, "load_in_8bit", True):
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
        raise ValueError(f"Token {token!r} is not in tokenizer.")

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

    if hasattr(test_data, "inter_data") and len(test_data.inter_data) > 0:
        sample = test_data.inter_data[0]
        if "fine_codes" in sample:
            return len(sample["fine_codes"])

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
    Build allowed coarse token ids for each code position.

    During implicit interleaved inference:
        prompt
        -> latent coarse_1
        -> visible fine_1
        -> latent coarse_2
        -> visible fine_2
        -> ...
    """
    if not hasattr(test_data, "coarse_indices") or test_data.coarse_indices is None:
        raise ValueError(
            "Interleaved inference requires coarse_indices. "
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

    return [sorted(list(pos_to_ids.get(i, set()))) for i in range(max_pos + 1)]


def get_model_input_embeddings(model):
    """
    Return input embedding module from normal HF model or PEFT model.

    With PEFT/LoRA, this may return ModulesToSaveWrapper.
    Do not directly use `.weight` on the returned object.
    Use get_input_embedding_weight(model) instead.
    """
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


def unwrap_peft_embedding(embedding_layer):
    """
    Fix for:
        AttributeError: 'ModulesToSaveWrapper' object has no attribute 'weight'

    PEFT may wrap embedding as:
        ModulesToSaveWrapper(
            original_module=Embedding(...),
            modules_to_save={"default": Embedding(...)}
        )

    The real trainable saved embedding is usually:
        modules_to_save["default"].weight
    """
    if hasattr(embedding_layer, "weight"):
        return embedding_layer

    if hasattr(embedding_layer, "modules_to_save"):
        modules_to_save = embedding_layer.modules_to_save

        candidate_adapter_names = []

        active_adapter = getattr(embedding_layer, "active_adapter", None)
        if isinstance(active_adapter, str):
            candidate_adapter_names.append(active_adapter)
        elif isinstance(active_adapter, (list, tuple)):
            candidate_adapter_names.extend(list(active_adapter))

        active_adapters = getattr(embedding_layer, "active_adapters", None)
        if isinstance(active_adapters, str):
            candidate_adapter_names.append(active_adapters)
        elif isinstance(active_adapters, (list, tuple)):
            candidate_adapter_names.extend(list(active_adapters))

        candidate_adapter_names.append("default")

        for adapter_name in candidate_adapter_names:
            if adapter_name in modules_to_save:
                real_module = modules_to_save[adapter_name]
                if hasattr(real_module, "weight"):
                    return real_module

        for real_module in modules_to_save.values():
            if hasattr(real_module, "weight"):
                return real_module

    if hasattr(embedding_layer, "original_module"):
        original_module = embedding_layer.original_module
        if hasattr(original_module, "weight"):
            return original_module

    if hasattr(embedding_layer, "module"):
        return unwrap_peft_embedding(embedding_layer.module)

    raise AttributeError(
        f"Cannot unwrap embedding layer of type {type(embedding_layer)}. "
        f"Available attributes: {dir(embedding_layer)}"
    )


def get_input_embedding_weight(model):
    embedding_layer = get_model_input_embeddings(model)
    embedding_layer = unwrap_peft_embedding(embedding_layer)
    return embedding_layer.weight


def past_to_legacy(past_key_values):
    if past_key_values is None:
        return None

    if hasattr(past_key_values, "to_legacy_cache"):
        return past_key_values.to_legacy_cache()

    return past_key_values


def index_select_past_key_values(past_key_values, indices: torch.Tensor):
    """
    Select batch entries from legacy past_key_values.

    Expected layer format:
        tuple(key, value, ...)
    """
    past_key_values = past_to_legacy(past_key_values)

    if past_key_values is None:
        return None

    selected = []

    for layer_past in past_key_values:
        if layer_past is None:
            selected.append(None)
            continue

        new_layer = []

        for x in layer_past:
            if torch.is_tensor(x):
                new_layer.append(x.index_select(0, indices))
            else:
                new_layer.append(x)

        selected.append(tuple(new_layer))

    return tuple(selected)


def cat_past_key_values(past_list: List[Any]):
    """
    Concatenate a list of legacy past_key_values along batch dimension.
    """
    if len(past_list) == 0:
        return None

    past_list = [past_to_legacy(p) for p in past_list]
    num_layers = len(past_list[0])
    packed = []

    for layer_idx in range(num_layers):
        layer_0 = past_list[0][layer_idx]

        if layer_0 is None:
            packed.append(None)
            continue

        part_num = len(layer_0)
        layer_parts = []

        for part_idx in range(part_num):
            xs = [p[layer_idx][part_idx] for p in past_list]

            if torch.is_tensor(xs[0]):
                layer_parts.append(torch.cat(xs, dim=0))
            else:
                layer_parts.append(xs[0])

        packed.append(tuple(layer_parts))

    return tuple(packed)


@torch.no_grad()
def lm_forward(
    model,
    input_ids: Optional[torch.Tensor] = None,
    inputs_embeds: Optional[torch.Tensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    past_key_values=None,
):
    outputs = model(
        input_ids=input_ids,
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        past_key_values=past_key_values,
        use_cache=True,
        return_dict=True,
    )

    logits = outputs.logits[:, -1, :]
    past = past_to_legacy(outputs.past_key_values)

    return logits, past


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
        [N, vocab_size], logits before latent coarse token.

    return:
        coarse_emb:
            [N, hidden_size], soft expected embedding over coarse token distribution.
        coarse_scores:
            [N], optional max log-prob over coarse candidates.
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


def build_allowed_cache_for_beams(
    fine_prefixes: List[List[int]],
    fine_trie: Trie,
    cache: Dict[tuple, List[int]],
) -> List[List[int]]:
    allowed_list = []

    for prefix in fine_prefixes:
        key = tuple(prefix)

        if key not in cache:
            cache[key] = fine_trie.get(prefix)

        allowed = cache[key]
        if allowed is None:
            allowed = []

        allowed_list.append(allowed)

    return allowed_list


@torch.no_grad()
@torch.no_grad()
def batch_generate_interleaved_qwen(
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
    Qwen3-safe implicit interleaved inference.

    This version does NOT use past_key_values / KV cache.

    Reason:
        Newer Qwen3 transformers uses Cache objects.
        Passing legacy tuple cache will cause:
            AttributeError: 'tuple' object has no attribute 'get_seq_length'

    Internal generation path:
        prompt
        -> latent coarse_1 soft embedding
        -> visible fine_1
        -> latent coarse_2 soft embedding
        -> visible fine_2
        -> ...

    Visible output:
        fine_1 fine_2 ... fine_L

    Coarse tokens:
        not decoded,
        not evaluated,
        only used as latent soft embeddings.
    """
    device = inputs["input_ids"].device
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    batch_size = input_ids.size(0)

    embedding_weight = get_input_embedding_weight(model)
    hidden_size = embedding_weight.size(-1)

    # prompt embeddings: [B, prompt_len, hidden]
    prompt_embeds = F.embedding(input_ids, embedding_weight)

    beams_by_sample = []

    for b in range(batch_size):
        beams_by_sample.append([
            {
                "fine_token_ids": [],
                "score": 0.0,
                # extra_embeds contains already generated latent coarse embeddings
                # and visible fine token embeddings.
                # shape: [extra_len, hidden]
                "extra_embeds": torch.empty(
                    0,
                    hidden_size,
                    dtype=embedding_weight.dtype,
                    device=device,
                ),
            }
        ])

    trie_cache = {}

    for step_idx in range(fine_code_len):
        flat_beams = []
        flat_sample_ids = []

        for sample_id in range(batch_size):
            for beam in beams_by_sample[sample_id]:
                flat_beams.append(beam)
                flat_sample_ids.append(sample_id)

        if len(flat_beams) == 0:
            break

        # ------------------------------------------------------------
        # 1. Build full inputs_embeds for each current beam:
        #       prompt + previous latent coarse/fine embeddings
        # ------------------------------------------------------------
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

        # In this decoding loop, all beams at the same step should have same length.
        full_inputs_embeds = torch.stack(full_embeds_list, dim=0)
        full_attention_mask = torch.stack(full_attention_list, dim=0)

        # ------------------------------------------------------------
        # 2. Predict latent coarse distribution at current position.
        # ------------------------------------------------------------
        coarse_outputs = model(
            inputs_embeds=full_inputs_embeds,
            attention_mask=full_attention_mask,
            use_cache=False,
            return_dict=True,
        )

        coarse_next_logits = coarse_outputs.logits[:, -1, :]

        coarse_pos = min(step_idx, len(coarse_position_token_ids) - 1)
        coarse_ids_this_pos = coarse_position_token_ids[coarse_pos]

        if len(coarse_ids_this_pos) == 0:
            raise ValueError(f"No coarse candidate tokens at position {coarse_pos}.")

        coarse_emb, coarse_scores = predict_soft_coarse_embedding(
            next_logits=coarse_next_logits,
            embedding_weight=embedding_weight,
            coarse_token_ids=coarse_ids_this_pos,
            temperature=coarse_temperature,
            use_coarse_score=use_coarse_score_in_rank,
        )

        # ------------------------------------------------------------
        # 3. Append latent coarse embedding and predict visible fine token.
        # ------------------------------------------------------------
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

        fine_outputs = model(
            inputs_embeds=inputs_after_coarse,
            attention_mask=attention_after_coarse,
            use_cache=False,
            return_dict=True,
        )

        fine_logits = fine_outputs.logits[:, -1, :]
        fine_log_probs = F.log_softmax(fine_logits.float(), dim=-1)

        # ------------------------------------------------------------
        # 4. Fine-only trie constrained beam expansion.
        # ------------------------------------------------------------
        candidate_sample_ids = []
        candidate_token_ids = []
        candidate_scores = []
        candidate_fine_prefixes = []
        candidate_extra_embeds = []

        per_beam_expand = max(1, num_beams)

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
                candidate_token_ids.append(fine_token_id)
                candidate_scores.append(base_score + coarse_score + fine_logprob)
                candidate_fine_prefixes.append(
                    beam["fine_token_ids"] + [fine_token_id]
                )
                candidate_extra_embeds.append(new_extra_embeds)

        if len(candidate_token_ids) == 0:
            break

        # ------------------------------------------------------------
        # 5. Keep top beams for each sample.
        # ------------------------------------------------------------
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

    # ------------------------------------------------------------
    # 6. Decode only visible fine token ids.
    # ------------------------------------------------------------
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

# def batch_generate_interleaved_qwen(
#     model,
#     tokenizer,
#     inputs,
#     fine_trie: Trie,
#     fine_code_len: int,
#     coarse_position_token_ids: List[List[int]],
#     num_beams: int,
#     coarse_temperature: float = 1.0,
#     use_coarse_score_in_rank: bool = False,
# ):
#     """
#     Qwen causal-LM implicit interleaved inference.

#     Internal path:
#         prompt
#         -> latent coarse_1 soft embedding
#         -> visible fine_1
#         -> latent coarse_2 soft embedding
#         -> visible fine_2
#         -> ...

#     Output:
#         only fine tokens are decoded and evaluated.
#     """
#     device = inputs["input_ids"].device
#     input_ids = inputs["input_ids"]
#     attention_mask = inputs["attention_mask"]

#     batch_size = input_ids.size(0)

#     # Important:
#     # PEFT/LoRA may wrap embedding as ModulesToSaveWrapper.
#     # So do not use get_model_input_embeddings(model).weight directly.
#     embedding_weight = get_input_embedding_weight(model)

#     init_logits, init_past = lm_forward(
#         model=model,
#         input_ids=input_ids,
#         attention_mask=attention_mask,
#         past_key_values=None,
#     )

#     beams_by_sample = []

#     for b in range(batch_size):
#         b_index = torch.tensor([b], dtype=torch.long, device=device)

#         beams_by_sample.append([
#             {
#                 "fine_token_ids": [],
#                 "score": 0.0,
#                 "next_logits": init_logits[b:b + 1].contiguous(),
#                 "past_key_values": index_select_past_key_values(init_past, b_index),
#                 "attention_mask": attention_mask[b:b + 1].contiguous(),
#             }
#         ])

#     trie_cache = {}

#     for step_idx in range(fine_code_len):
#         flat_beams = []
#         flat_sample_ids = []

#         for sample_id in range(batch_size):
#             for beam in beams_by_sample[sample_id]:
#                 flat_beams.append(beam)
#                 flat_sample_ids.append(sample_id)

#         if len(flat_beams) == 0:
#             break

#         flat_next_logits = torch.cat(
#             [beam["next_logits"] for beam in flat_beams],
#             dim=0,
#         )

#         flat_past = cat_past_key_values(
#             [beam["past_key_values"] for beam in flat_beams]
#         )

#         flat_attention_mask = torch.cat(
#             [beam["attention_mask"] for beam in flat_beams],
#             dim=0,
#         )

#         coarse_pos = min(step_idx, len(coarse_position_token_ids) - 1)
#         coarse_ids_this_pos = coarse_position_token_ids[coarse_pos]

#         if len(coarse_ids_this_pos) == 0:
#             raise ValueError(f"No coarse candidate tokens at position {coarse_pos}.")

#         coarse_emb, coarse_scores = predict_soft_coarse_embedding(
#             next_logits=flat_next_logits,
#             embedding_weight=embedding_weight,
#             coarse_token_ids=coarse_ids_this_pos,
#             temperature=coarse_temperature,
#             use_coarse_score=use_coarse_score_in_rank,
#         )

#         ones = torch.ones(
#             flat_attention_mask.size(0),
#             1,
#             dtype=flat_attention_mask.dtype,
#             device=device,
#         )

#         attention_after_coarse = torch.cat([flat_attention_mask, ones], dim=1)

#         fine_logits, past_after_coarse = lm_forward(
#             model=model,
#             input_ids=None,
#             inputs_embeds=coarse_emb.unsqueeze(1),
#             attention_mask=attention_after_coarse,
#             past_key_values=flat_past,
#         )

#         fine_log_probs = F.log_softmax(fine_logits.float(), dim=-1)

#         fine_prefixes = [beam["fine_token_ids"] for beam in flat_beams]
#         allowed_list = build_allowed_cache_for_beams(
#             fine_prefixes=fine_prefixes,
#             fine_trie=fine_trie,
#             cache=trie_cache,
#         )

#         candidate_parent_flat_idx = []
#         candidate_sample_ids = []
#         candidate_token_ids = []
#         candidate_scores = []
#         candidate_fine_prefixes = []

#         per_beam_expand = max(1, num_beams)

#         for flat_idx, beam in enumerate(flat_beams):
#             allowed = allowed_list[flat_idx]

#             if allowed is None or len(allowed) == 0:
#                 continue

#             allowed_tensor = torch.tensor(
#                 allowed,
#                 dtype=torch.long,
#                 device=device,
#             )

#             allowed_scores = fine_log_probs[flat_idx].index_select(0, allowed_tensor)
#             topk = min(per_beam_expand, allowed_tensor.size(0))

#             topk_scores, topk_pos = torch.topk(allowed_scores, k=topk)
#             topk_ids = allowed_tensor[topk_pos]

#             base_score = float(beam["score"])
#             coarse_score = float(coarse_scores[flat_idx].item())
#             sample_id = flat_sample_ids[flat_idx]

#             for j in range(topk):
#                 fine_token_id = int(topk_ids[j].item())
#                 fine_logprob = float(topk_scores[j].item())

#                 candidate_parent_flat_idx.append(flat_idx)
#                 candidate_sample_ids.append(sample_id)
#                 candidate_token_ids.append(fine_token_id)
#                 candidate_scores.append(base_score + coarse_score + fine_logprob)
#                 candidate_fine_prefixes.append(
#                     beam["fine_token_ids"] + [fine_token_id]
#                 )

#         if len(candidate_token_ids) == 0:
#             break

#         candidate_parent_flat_idx_t = torch.tensor(
#             candidate_parent_flat_idx,
#             dtype=torch.long,
#             device=device,
#         )

#         candidate_input_ids = torch.tensor(
#             candidate_token_ids,
#             dtype=torch.long,
#             device=device,
#         ).unsqueeze(1)

#         candidate_past = index_select_past_key_values(
#             past_after_coarse,
#             candidate_parent_flat_idx_t,
#         )

#         candidate_attention_parent = attention_after_coarse.index_select(
#             0,
#             candidate_parent_flat_idx_t,
#         )

#         ones = torch.ones(
#             candidate_attention_parent.size(0),
#             1,
#             dtype=candidate_attention_parent.dtype,
#             device=device,
#         )

#         candidate_attention_after_fine = torch.cat(
#             [candidate_attention_parent, ones],
#             dim=1,
#         )

#         candidate_next_logits, candidate_past_after_fine = lm_forward(
#             model=model,
#             input_ids=candidate_input_ids,
#             inputs_embeds=None,
#             attention_mask=candidate_attention_after_fine,
#             past_key_values=candidate_past,
#         )

#         candidates_grouped = [[] for _ in range(batch_size)]

#         for cand_idx, sample_id in enumerate(candidate_sample_ids):
#             candidates_grouped[sample_id].append(cand_idx)

#         new_beams_by_sample = [[] for _ in range(batch_size)]

#         for sample_id in range(batch_size):
#             cand_indices = candidates_grouped[sample_id]

#             if len(cand_indices) == 0:
#                 continue

#             cand_indices.sort(key=lambda i: candidate_scores[i], reverse=True)

#             dedup = []
#             seen = set()

#             for cand_idx in cand_indices:
#                 prefix_key = tuple(candidate_fine_prefixes[cand_idx])

#                 if prefix_key in seen:
#                     continue

#                 seen.add(prefix_key)

#                 cand_select = torch.tensor(
#                     [cand_idx],
#                     dtype=torch.long,
#                     device=device,
#                 )

#                 dedup.append({
#                     "fine_token_ids": candidate_fine_prefixes[cand_idx],
#                     "score": float(candidate_scores[cand_idx]),
#                     "next_logits": candidate_next_logits[
#                         cand_idx:cand_idx + 1
#                     ].contiguous(),
#                     "past_key_values": index_select_past_key_values(
#                         candidate_past_after_fine,
#                         cand_select,
#                     ),
#                     "attention_mask": candidate_attention_after_fine[
#                         cand_idx:cand_idx + 1
#                     ].contiguous(),
#                 })

#                 if len(dedup) >= num_beams:
#                     break

#             new_beams_by_sample[sample_id] = dedup

#         beams_by_sample = new_beams_by_sample

#     decoded = []
#     scores = []

#     for sample_id in range(batch_size):
#         beams = beams_by_sample[sample_id]

#         if len(beams) == 0:
#             beams = [{
#                 "fine_token_ids": [],
#                 "score": -1e9,
#             }]

#         while len(beams) < num_beams:
#             beams.append({
#                 "fine_token_ids": list(beams[-1]["fine_token_ids"]),
#                 "score": float(beams[-1]["score"]),
#             })

#         for beam in beams[:num_beams]:
#             text = tokenizer.decode(
#                 beam["fine_token_ids"],
#                 skip_special_tokens=True,
#             )
#             decoded.append(normalize_code_text(text))
#             scores.append(float(beam["score"]))

#     return decoded, scores


def run_vanilla_generate(
    model,
    tokenizer,
    inputs,
    base_prefix_fn,
    prompt_len,
    args,
):
    """
    Original non-interleaved generate path.
    Only used when passing --no_interleaved_inference.
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
    lm_model = model

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
            print("[interleaved] enabled")
            print("[interleaved] fine_code_len:", fine_code_len)
            print("[interleaved] coarse positions:", len(coarse_position_token_ids))
            print("[interleaved] fine trie size:", len(fine_trie))
    else:
        if local_rank == 0:
            print("[interleaved] disabled, using vanilla generate()")

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
                print("Start prompt: ", prompt_id)

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
                            decoded, scores = batch_generate_interleaved_qwen(
                                model=lm_model,
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
                        decoded, scores = batch_generate_interleaved_qwen(
                            model=lm_model,
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
                        model=lm_model,
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
        print("Mean results: ", mean_results)
        print("Min results: ", min_results)
        print("Max results: ", max_results)
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

        print("Save file: ", args.results_file)

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
        description="HoloRec-Qwen implicit interleaved inference: latent coarse + visible fine"
    )

    parser = parse_global_args(parser)
    parser = parse_dataset_args(parser)
    parser = parse_test_args(parser)

    parser.add_argument(
        "--coarse_index_file",
        type=str,
        default=".tw8.json",
        help="Coarse index file. Required for implicit interleaved inference.",
    )

    parser.add_argument(
        "--interleaved_inference",
        dest="interleaved_inference",
        action="store_true",
        default=True,
        help="Enable implicit interleaved inference. Default: enabled.",
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
        default=1.0,
        help="Temperature for latent coarse soft distribution.",
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
            "For fine-only interleaved decoding, default False is recommended."
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
        default=True,
        help="Load non-LoRA checkpoint with 8bit quantization.",
    )

    parser.add_argument(
        "--no_load_in_8bit",
        dest="load_in_8bit",
        action="store_false",
        help="Disable 8bit loading for non-LoRA checkpoint.",
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
