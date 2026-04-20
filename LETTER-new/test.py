import argparse
import json
import os
from typing import List, Dict, Optional

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from transformers import T5Tokenizer, T5Config

from utils import *
from collator import TestCollator
from evaluate import get_topk_results, get_metrics_results
from generation_trie import Trie, build_trie_from_token_sequences
from modeling_letter import LETTER


def init_distributed():
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        dist.init_process_group(backend="nccl")
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        rank = int(os.environ.get("RANK", "0"))
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        torch.cuda.set_device(local_rank)
        return True, local_rank, rank, world_size
    return False, 0, 0, 1


def is_rank0(is_dist: bool, rank: int) -> bool:
    return (not is_dist) or rank == 0


def parse_prompt_ids(test_prompt_ids) -> List[int]:
    if test_prompt_ids is None:
        return [0]
    if isinstance(test_prompt_ids, int):
        return [test_prompt_ids]
    if isinstance(test_prompt_ids, list):
        return [int(x) for x in test_prompt_ids]
    if isinstance(test_prompt_ids, str):
        s = test_prompt_ids.strip()
        if not s:
            return [0]
        return [int(x) for x in s.split(",")]
    return [0]


def all_reduce_metrics(metrics_results: dict, total: int, device: torch.device, is_dist: bool):
    if not is_dist:
        if total == 0:
            return {k: 0.0 for k in metrics_results}
        return {k: float(v) / float(total) for k, v in metrics_results.items()}

    total_tensor = torch.tensor([total], device=device, dtype=torch.long)
    dist.all_reduce(total_tensor, op=dist.ReduceOp.SUM)
    total_all = int(total_tensor.item())

    out = {}
    for k, v in metrics_results.items():
        t = torch.tensor([float(v)], device=device, dtype=torch.float64)
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        out[k] = (t.item() / total_all) if total_all > 0 else 0.0

    return out


def ensure_parent_dir(file_path: str):
    parent = os.path.dirname(file_path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def infer_fine_code_length(test_data) -> int:
    if hasattr(test_data, "indices") and len(test_data.indices) > 0:
        first_key = next(iter(test_data.indices.keys()))
        return len(test_data.indices[first_key])

    if hasattr(test_data, "inter_data") and len(test_data.inter_data) > 0:
        sample = test_data.inter_data[0]
        if "fine_codes" in sample:
            return len(sample["fine_codes"])

    raise ValueError("Cannot infer fine code length from test dataset.")


def build_fine_trie_from_dataset(test_data, tokenizer: T5Tokenizer) -> Trie:
    fine_token_sequences = list(test_data.indices.values())
    return build_trie_from_token_sequences(fine_token_sequences, tokenizer)


def normalize_code_text(x: str) -> str:
    return str(x).strip().replace(" ", "")


def fine_codes_to_text(fine_codes: List[str]) -> str:
    return normalize_code_text("".join(fine_codes))


@torch.no_grad()
def decoder_prefill(
    model,
    encoder_hidden_states: torch.Tensor,
    encoder_attention_mask: torch.Tensor,
    decoder_input_ids: torch.Tensor,
):
    decoder_outputs = model.decoder(
        input_ids=decoder_input_ids,
        encoder_hidden_states=encoder_hidden_states,
        encoder_attention_mask=encoder_attention_mask,
        use_cache=True,
        return_dict=True,
        output_attentions=False,
        output_hidden_states=False,
    )

    last_hidden = decoder_outputs.last_hidden_state[:, -1, :]
    if model.config.tie_word_embeddings:
        last_hidden = last_hidden * (model.model_dim ** -0.5)

    return last_hidden, decoder_outputs.past_key_values


@torch.no_grad()
def decoder_step(
    model,
    encoder_hidden_states: torch.Tensor,
    encoder_attention_mask: torch.Tensor,
    past_key_values,
    input_ids: Optional[torch.Tensor] = None,
    inputs_embeds: Optional[torch.Tensor] = None,
):
    decoder_outputs = model.decoder(
        input_ids=input_ids,
        inputs_embeds=inputs_embeds,
        past_key_values=past_key_values,
        encoder_hidden_states=encoder_hidden_states,
        encoder_attention_mask=encoder_attention_mask,
        use_cache=True,
        return_dict=True,
        output_attentions=False,
        output_hidden_states=False,
    )

    last_hidden = decoder_outputs.last_hidden_state[:, -1, :]
    if model.config.tie_word_embeddings:
        last_hidden = last_hidden * (model.model_dim ** -0.5)

    return last_hidden, decoder_outputs.past_key_values


def index_select_past_key_values(past_key_values, indices: torch.Tensor):
    selected = []
    for layer_past in past_key_values:
        new_layer = []
        for x in layer_past:
            new_layer.append(x.index_select(0, indices))
        selected.append(tuple(new_layer))
    return tuple(selected)


@torch.no_grad()
def predict_coarse_soft_batch(
    model,
    last_hidden: torch.Tensor,
    use_coarse_score: bool = False,
):
    """
    last_hidden: [N, D]
    return:
      coarse_emb: [N, D]
      coarse_scores: [N]
    """
    coarse_logits = model.coarse_head(last_hidden)  # [N, V]
    temp = getattr(model, "temperature", 1.0)

    coarse_probs = F.softmax(coarse_logits / temp, dim=-1)
    coarse_emb = torch.matmul(coarse_probs, model.shared.weight)

    if use_coarse_score:
        coarse_log_probs = F.log_softmax(coarse_logits, dim=-1)
        coarse_scores = coarse_log_probs.max(dim=-1).values
    else:
        coarse_scores = torch.zeros(
            last_hidden.size(0),
            dtype=last_hidden.dtype,
            device=last_hidden.device
        )

    return coarse_emb, coarse_scores


@torch.no_grad()
def compute_fine_log_probs_batch(
    model,
    hidden_after_coarse: torch.Tensor,
    coarse_emb: torch.Tensor,
):
    fine_hidden = model._inject_coarse(hidden_after_coarse, coarse_emb)
    fine_logits = model.fine_head(fine_hidden)
    fine_log_probs = F.log_softmax(fine_logits, dim=-1)
    return fine_log_probs


def build_allowed_cache_for_beams(
    fine_prefixes: List[List[int]],
    fine_trie: Trie,
    cache: Dict[tuple, List[int]],
):
    allowed_list = []
    for prefix in fine_prefixes:
        key = tuple(prefix)
        if key not in cache:
            cache[key] = fine_trie.get(prefix)
        allowed_list.append(cache[key])
    return allowed_list


@torch.no_grad()
def batch_generate_interleaved_ultra(
    model,
    tokenizer,
    inputs,
    fine_trie: Trie,
    fine_code_len: int,
    num_beams: int,
    use_coarse_score_in_rank: bool = False,
):
    """
    batch-level beam search:
    - coarse 仅做 latent
    - fine 做真实搜索
    - trie 只约束 fine prefix
    """
    device = inputs["input_ids"].device
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    batch_size = input_ids.size(0)

    # encoder once
    encoder_outputs = model.encoder(
        input_ids=input_ids,
        attention_mask=attention_mask,
        return_dict=True,
    )
    enc_h0 = encoder_outputs.last_hidden_state
    enc_m0 = attention_mask

    decoder_start_token_id = model.config.decoder_start_token_id
    if decoder_start_token_id is None:
        decoder_start_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

    start_ids = torch.full(
        (batch_size, 1),
        fill_value=decoder_start_token_id,
        dtype=torch.long,
        device=device
    )

    # prefill once
    init_last_hidden, init_past = decoder_prefill(
        model=model,
        encoder_hidden_states=enc_h0,
        encoder_attention_mask=enc_m0,
        decoder_input_ids=start_ids,
    )

    beams_by_sample = []
    for b in range(batch_size):
        beams_by_sample.append([
            {
                "fine_token_ids": [],
                "score": 0.0,
                "last_hidden": init_last_hidden[b:b + 1].contiguous(),
                "past_key_values": index_select_past_key_values(
                    init_past,
                    torch.tensor([b], dtype=torch.long, device=device)
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

        flat_sample_ids_t = torch.tensor(flat_sample_ids, dtype=torch.long, device=device)

        flat_last_hidden = torch.cat([b["last_hidden"] for b in flat_beams], dim=0)  # [N, D]

        # pack past
        packed_past = []
        num_layers = len(flat_beams[0]["past_key_values"])
        for layer_idx in range(num_layers):
            layer_parts = []
            part_num = len(flat_beams[0]["past_key_values"][layer_idx])
            for part_idx in range(part_num):
                cat_tensor = torch.cat(
                    [b["past_key_values"][layer_idx][part_idx] for b in flat_beams],
                    dim=0
                )
                layer_parts.append(cat_tensor)
            packed_past.append(tuple(layer_parts))
        packed_past = tuple(packed_past)

        flat_enc_h = enc_h0.index_select(0, flat_sample_ids_t)
        flat_enc_m = enc_m0.index_select(0, flat_sample_ids_t)

        # 1) coarse latent step
        coarse_emb, coarse_scores = predict_coarse_soft_batch(
            model=model,
            last_hidden=flat_last_hidden,
            use_coarse_score=use_coarse_score_in_rank,
        )

        hidden_after_coarse, past_after_coarse = decoder_step(
            model=model,
            encoder_hidden_states=flat_enc_h,
            encoder_attention_mask=flat_enc_m,
            past_key_values=packed_past,
            input_ids=None,
            inputs_embeds=coarse_emb.unsqueeze(1),
        )

        # 2) fine log probs
        fine_log_probs = compute_fine_log_probs_batch(
            model=model,
            hidden_after_coarse=hidden_after_coarse,
            coarse_emb=coarse_emb,
        )

        # 3) trie-constrained candidates
        fine_prefixes = [b["fine_token_ids"] for b in flat_beams]
        allowed_list = build_allowed_cache_for_beams(
            fine_prefixes=fine_prefixes,
            fine_trie=fine_trie,
            cache=trie_cache
        )

        candidate_parent_flat_idx = []
        candidate_sample_ids = []
        candidate_token_ids = []
        candidate_scores = []
        candidate_fine_prefixes = []

        per_beam_expand = max(1, num_beams)

        for flat_idx, beam in enumerate(flat_beams):
            allowed = allowed_list[flat_idx]
            if allowed is None or len(allowed) == 0:
                continue

            allowed_tensor = torch.tensor(allowed, dtype=torch.long, device=device)
            allowed_scores = fine_log_probs[flat_idx].index_select(0, allowed_tensor)

            topk = min(per_beam_expand, allowed_tensor.size(0))
            topk_scores, topk_pos = torch.topk(allowed_scores, k=topk)
            topk_ids = allowed_tensor[topk_pos]

            base_score = float(beam["score"])
            coarse_score = float(coarse_scores[flat_idx].item())
            sample_id = flat_sample_ids[flat_idx]

            for j in range(topk):
                fine_token_id = int(topk_ids[j].item())
                fine_logprob = float(topk_scores[j].item())

                candidate_parent_flat_idx.append(flat_idx)
                candidate_sample_ids.append(sample_id)
                candidate_token_ids.append(fine_token_id)
                candidate_scores.append(base_score + coarse_score + fine_logprob)
                candidate_fine_prefixes.append(beam["fine_token_ids"] + [fine_token_id])

        if len(candidate_token_ids) == 0:
            break

        candidate_parent_flat_idx_t = torch.tensor(candidate_parent_flat_idx, dtype=torch.long, device=device)
        candidate_sample_ids_t = torch.tensor(candidate_sample_ids, dtype=torch.long, device=device)
        candidate_input_ids = torch.tensor(candidate_token_ids, dtype=torch.long, device=device).unsqueeze(1)

        # 4) one batch decoder step for all fine candidates
        cand_enc_h = flat_enc_h.index_select(0, candidate_parent_flat_idx_t)
        cand_enc_m = flat_enc_m.index_select(0, candidate_parent_flat_idx_t)
        cand_past = index_select_past_key_values(past_after_coarse, candidate_parent_flat_idx_t)

        cand_hidden_after_fine, cand_past_after_fine = decoder_step(
            model=model,
            encoder_hidden_states=cand_enc_h,
            encoder_attention_mask=cand_enc_m,
            past_key_values=cand_past,
            input_ids=candidate_input_ids,
            inputs_embeds=None,
        )

        # 5) regroup candidates
        new_beams_by_sample = [[] for _ in range(batch_size)]
        candidates_grouped = [[] for _ in range(batch_size)]
        for idx in range(len(candidate_token_ids)):
            sample_id = candidate_sample_ids[idx]
            candidates_grouped[sample_id].append(idx)

        for sample_id in range(batch_size):
            cand_indices = candidates_grouped[sample_id]
            if len(cand_indices) == 0:
                continue

            cand_indices.sort(key=lambda i: candidate_scores[i], reverse=True)

            dedup = []
            seen = set()
            for i in cand_indices:
                key = tuple(candidate_fine_prefixes[i])
                if key in seen:
                    continue
                seen.add(key)

                dedup.append(
                    {
                        "fine_token_ids": candidate_fine_prefixes[i],
                        "score": candidate_scores[i],
                        "last_hidden": cand_hidden_after_fine[i:i + 1].contiguous(),
                        "past_key_values": index_select_past_key_values(
                            cand_past_after_fine,
                            torch.tensor([i], dtype=torch.long, device=device)
                        ),
                    }
                )
                if len(dedup) >= num_beams:
                    break

            new_beams_by_sample[sample_id] = dedup

        beams_by_sample = new_beams_by_sample

    # finalize
    decoded = []
    scores = []

    for sample_id in range(batch_size):
        beams = beams_by_sample[sample_id]

        if len(beams) == 0:
            beams = [{"fine_token_ids": [], "score": -1e9}]

        while len(beams) < num_beams:
            beams.append({
                "fine_token_ids": list(beams[-1]["fine_token_ids"]),
                "score": float(beams[-1]["score"]),
            })

        for beam in beams[:num_beams]:
            fine_text = tokenizer.decode(
                beam["fine_token_ids"],
                skip_special_tokens=True
            ).replace(" ", "")
            decoded.append(fine_text)
            scores.append(float(beam["score"]))

    return decoded, scores


def test(args):
    is_dist, local_rank, rank, world_size = init_distributed()
    device = torch.device("cuda", local_rank)

    base_seed = getattr(args, "seed", 42)
    set_seed(base_seed + rank)

    if is_rank0(is_dist, rank):
        print(vars(args))
        print(f"[DDP] is_dist={is_dist}, world_size={world_size}")

    config = T5Config.from_pretrained(
        args.ckpt_path,
        local_files_only=True
    )

    tokenizer = T5Tokenizer.from_pretrained(
        args.ckpt_path,
        model_max_length=512,
        local_files_only=True
    )

    train_data, valid_data = load_datasets(args)
    add_num = tokenizer.add_tokens(train_data.datasets[0].get_new_tokens())
    config.vocab_size = len(tokenizer)

    if is_rank0(is_dist, rank):
        print(f"add {add_num} new token.")
        print("train data num:", len(train_data))

    model = LETTER.from_pretrained(
        args.ckpt_path,
        config=config,
        local_files_only=True
    )

    if hasattr(model, "resize_token_embeddings_and_heads"):
        model.resize_token_embeddings_and_heads(len(tokenizer))
    else:
        model.resize_token_embeddings(len(tokenizer))

    if hasattr(model, "set_hyper"):
        model.set_hyper(
            temperature=getattr(args, "temperature", 1.0),
            coarse_loss_weight=1.0,
            fine_loss_weight=1.0,
            coarse_align_weight=1.0,
        )


    model.to(device)
    model.eval()

    prompt_ids = parse_prompt_ids(getattr(args, "test_prompt_ids", 0))

    test_data = load_test_dataset(args)
    collator = TestCollator(args, tokenizer)

    fine_trie = build_fine_trie_from_dataset(test_data, tokenizer)
    fine_code_len = infer_fine_code_length(test_data)
    all_items = [fine_codes_to_text(v) for v in test_data.indices.values()]

    sampler = DistributedSampler(
        test_data,
        num_replicas=world_size,
        rank=rank,
        shuffle=False,
        drop_last=False
    ) if is_dist else None

    test_loader = DataLoader(
        test_data,
        batch_size=args.test_batch_size,
        collate_fn=collator,
        shuffle=False,
        sampler=sampler,
        num_workers=4,
        pin_memory=True
    )

    if is_rank0(is_dist, rank):
        print("test data num:", len(test_data))
        print("fine code len:", fine_code_len)

    metrics = args.metrics.split(",")
    all_prompt_results = []
    local_pred_lines = []

    amp_dtype = None
    if getattr(args, "use_bf16", False):
        amp_dtype = torch.bfloat16
    elif getattr(args, "use_fp16", False):
        amp_dtype = torch.float16

    with torch.no_grad():
        for prompt_id in prompt_ids:
            test_loader.dataset.set_prompt(prompt_id)

            metrics_results = {}
            total = 0

            pbar = tqdm(test_loader, disable=not is_rank0(is_dist, rank))

            for step, batch in enumerate(pbar):
                inputs = batch[0]
                raw_targets = batch[1]
                bs = len(raw_targets)
                total += bs

                inputs = {k: v.to(device, non_blocking=True) for k, v in inputs.items()}
                target_texts = [fine_codes_to_text(t["fine_codes"]) for t in raw_targets]

                if step == 0 and is_rank0(is_dist, rank):
                    try:
                        print(inputs)
                        print(target_texts[:5])
                    except Exception:
                        pass

                if amp_dtype is not None:
                    with torch.cuda.amp.autocast(dtype=amp_dtype):
                        decoded, scores = batch_generate_interleaved_ultra(
                            model=model,
                            tokenizer=tokenizer,
                            inputs=inputs,
                            fine_trie=fine_trie,
                            fine_code_len=fine_code_len,
                            num_beams=args.num_beams,
                            use_coarse_score_in_rank=getattr(args, "use_coarse_score_in_rank", False),
                        )
                else:
                    decoded, scores = batch_generate_interleaved_ultra(
                        model=model,
                        tokenizer=tokenizer,
                        inputs=inputs,
                        fine_trie=fine_trie,
                        fine_code_len=fine_code_len,
                        num_beams=args.num_beams,
                        use_coarse_score_in_rank=getattr(args, "use_coarse_score_in_rank", False),
                    )

                for i, tgt in enumerate(target_texts):
                    start_idx = i * args.num_beams
                    end_idx = (i + 1) * args.num_beams
                    pred_list = decoded[start_idx:end_idx]

                    local_pred_lines.append(f"prompt_id: {prompt_id}")
                    local_pred_lines.append(f"target: {tgt}")
                    for beam_idx, pred in enumerate(pred_list, start=1):
                        local_pred_lines.append(f"pred_{beam_idx}: {normalize_code_text(pred)}")
                    local_pred_lines.append("")

                topk_res = get_topk_results(
                    decoded,
                    scores,
                    target_texts,
                    args.num_beams,
                    all_items=all_items if getattr(args, "filter_items", False) else None
                )

                batch_metrics_res = get_metrics_results(topk_res, metrics)
                for m, res in batch_metrics_res.items():
                    metrics_results[m] = metrics_results.get(m, 0.0) + float(res)

                if is_rank0(is_dist, rank):
                    temp = {
                        m: (metrics_results[m] / total if total > 0 else 0.0)
                        for m in metrics_results
                    }
                    pbar.set_postfix(temp)

            final_metrics = all_reduce_metrics(metrics_results, total, device, is_dist)
            all_prompt_results.append(final_metrics)

            if is_rank0(is_dist, rank):
                print("======================================================")
                print(f"Prompt {prompt_id} results: ", final_metrics)
                print("======================================================\n")

    if is_dist:
        gathered_lines = [None for _ in range(world_size)] if rank == 0 else None
        dist.gather_object(local_pred_lines, gathered_lines, dst=0)

        if rank == 0:
            all_pred_lines = []
            for lines in gathered_lines:
                if lines is not None:
                    all_pred_lines.extend(lines)
        else:
            all_pred_lines = None
    else:
        all_pred_lines = local_pred_lines

    if is_rank0(is_dist, rank):
        mean_results, min_results, max_results = {}, {}, {}
        for m in metrics:
            all_res = [r.get(m, 0.0) for r in all_prompt_results]
            mean_results[m] = sum(all_res) / len(all_res) if all_res else 0.0
            min_results[m] = min(all_res) if all_res else 0.0
            max_results[m] = max(all_res) if all_res else 0.0

        print("======================================================")
        print("Mean results: ", mean_results)
        print("Min results: ", min_results)
        print("Max results: ", max_results)
        print("======================================================")

        save_data = {
            "test_prompt_ids": prompt_ids,
            "mean_results": mean_results,
            "min_results": min_results,
            "max_results": max_results,
            "all_prompt_results": all_prompt_results,
        }

        ensure_parent_dir(args.results_file)
        with open(args.results_file, "w", encoding="utf-8") as f:
            json.dump(save_data, f, indent=4, ensure_ascii=False)

        ensure_parent_dir(args.save_pred_txt)
        with open(args.save_pred_txt, "w", encoding="utf-8") as f:
            for line in all_pred_lines:
                f.write(line + "\n")

        print(f"Prediction txt saved to: {args.save_pred_txt}")

    if is_dist:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLMRec_test_interleaved_latent_coarse_ultra")

    parser = parse_global_args(parser)
    parser = parse_dataset_args(parser)
    parser = parse_test_args(parser)

    parser.add_argument(
        "--coarse_index_file",
        type=str,
        default=".index.xinyan16.epoch10000.alpha2e-2-beta1e-4.json"
    )
    parser.add_argument(
        "--save_pred_txt",
        type=str,
        default="./results/predictions.txt",
        help="Path to save targets and predictions in txt format."
    )
    parser.add_argument(
        "--use_coarse_score_in_rank",
        action="store_true",
        help="Whether to include coarse max-logprob into beam score."
    )
    parser.add_argument(
        "--use_fp16",
        action="store_true",
        help="Use fp16 autocast in inference."
    )
    parser.add_argument(
        "--use_bf16",
        action="store_true",
        help="Use bf16 autocast in inference."
    )

    args = parser.parse_args()
    test(args)
