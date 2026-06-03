import argparse
import inspect
import os
import sys
from typing import Dict, List

import torch
import torch.nn.functional as F
import transformers
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)

from peft import (
    TaskType,
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    set_peft_model_state_dict,
)

from utils import *
from collator import Collator


IGNORE_INDEX = -100


def _flatten_tokenizer_ids(encoded):
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
        ids = _flatten_tokenizer_ids(encoded)
        if len(ids) == 1:
            return int(ids[0])
        raise ValueError(f"Token {token!r} is not a single token: {ids}")

    unk_id = getattr(tokenizer, "unk_token_id", None)
    if unk_id is not None and int(token_id) == int(unk_id):
        encoded = tokenizer(token, add_special_tokens=False)
        ids = _flatten_tokenizer_ids(encoded)
        if len(ids) == 1:
            return int(ids[0])
        raise ValueError(f"Token {token!r} maps to unk or multiple ids: {ids}")

    return int(token_id)


def build_align_codebooks_from_dataset(dataset, tokenizer) -> List[Dict[str, List[int]]]:
    """
    Build position-wise fine-token -> coarse-bucket codebooks.

    For each code position p:
        fine_ids[p]   = all fine token ids appearing at p
        coarse_ids[p] = corresponding coarse token id for each fine token candidate

    coarse_align_weight uses these codebooks to compute:

        -log P(fine token is inside gold coarse bucket | fine-position logits)

    This preserves a coarse-level alignment objective without another Qwen forward.
    """

    if not hasattr(dataset, "indices"):
        raise ValueError("Dataset has no `indices`; cannot build align codebooks.")

    if not hasattr(dataset, "coarse_indices") or dataset.coarse_indices is None:
        raise ValueError(
            "coarse_align_weight requires coarse_indices. "
            "Please pass --coarse_index_file, for example --coarse_index_file .tw8.json"
        )

    pos_to_pairs = {}

    for item_id, fine_seq in dataset.indices.items():
        item_key = str(item_id)

        if item_key not in dataset.coarse_indices:
            raise ValueError(f"Item {item_key} exists in fine indices but not coarse_indices.")

        coarse_seq = dataset.coarse_indices[item_key]

        if len(fine_seq) != len(coarse_seq):
            raise ValueError(
                f"fine/coarse code length mismatch for item {item_key}: "
                f"{len(fine_seq)} vs {len(coarse_seq)}"
            )

        for pos, (fine_tok, coarse_tok) in enumerate(zip(fine_seq, coarse_seq)):
            fine_id = token_to_id(tokenizer, fine_tok)
            coarse_id = token_to_id(tokenizer, coarse_tok)
            pos_to_pairs.setdefault(pos, set()).add((fine_id, coarse_id))

    if len(pos_to_pairs) == 0:
        raise ValueError("align codebooks are empty.")

    max_pos = max(pos_to_pairs.keys())
    codebooks = []

    for pos in range(max_pos + 1):
        pairs = sorted(list(pos_to_pairs.get(pos, set())))

        if len(pairs) == 0:
            codebooks.append({"fine_ids": [], "coarse_ids": []})
            continue

        fine_ids = [int(x[0]) for x in pairs]
        coarse_ids = [int(x[1]) for x in pairs]

        codebooks.append(
            {
                "fine_ids": fine_ids,
                "coarse_ids": coarse_ids,
            }
        )

    return codebooks


class HoloRecFastAlignTrainer(transformers.Trainer):
    """
    Single-forward HoloRec-Qwen trainer.

    Loss parts:
      1. weighted token CE for explicit interleaved coarse/fine tokens
      2. optional coarse bucket alignment loss on fine-token prediction positions

    This keeps coarse_align_weight meaningful without using InterleavedLatentQwen.
    """

    def __init__(self, *args, align_codebooks=None, coarse_align_weight=0.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.align_codebooks = align_codebooks or []
        self.coarse_align_weight = float(coarse_align_weight)
        self._align_tensor_cache = {}

    def _get_align_codebook_tensors(self, pos: int, device):
        cache_key = (int(pos), str(device))

        if cache_key in self._align_tensor_cache:
            return self._align_tensor_cache[cache_key]

        if pos < 0 or pos >= len(self.align_codebooks):
            return None

        codebook = self.align_codebooks[pos]
        fine_ids = codebook.get("fine_ids", [])
        coarse_ids = codebook.get("coarse_ids", [])

        if len(fine_ids) == 0:
            return None

        out = {
            "fine_ids": torch.tensor(fine_ids, dtype=torch.long, device=device),
            "coarse_ids": torch.tensor(coarse_ids, dtype=torch.long, device=device),
        }

        self._align_tensor_cache[cache_key] = out
        return out

    def _compute_weighted_token_loss(self, logits, labels, loss_weights):
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous().to(shift_logits.device)

        vocab_size = shift_logits.size(-1)

        flat_loss = F.cross_entropy(
            shift_logits.view(-1, vocab_size),
            shift_labels.view(-1),
            ignore_index=IGNORE_INDEX,
            reduction="none",
        )

        token_loss = flat_loss.view_as(shift_labels)
        valid_mask = shift_labels.ne(IGNORE_INDEX)

        if loss_weights is None:
            denom = valid_mask.sum().clamp_min(1)
            return (token_loss * valid_mask).sum() / denom

        shift_weights = loss_weights[..., 1:].contiguous().to(
            device=shift_logits.device,
            dtype=token_loss.dtype,
        )

        weighted_mask = shift_weights * valid_mask.to(token_loss.dtype)
        denom = weighted_mask.sum().clamp_min(1.0)

        return (token_loss * weighted_mask).sum() / denom

    def _compute_coarse_align_loss(self, logits, align_positions, align_coarse_labels):
        if self.coarse_align_weight <= 0:
            return logits.new_zeros(())

        if align_positions is None or align_coarse_labels is None:
            return logits.new_zeros(())

        shift_logits = logits[..., :-1, :].contiguous()
        shift_align_positions = align_positions[..., 1:].contiguous().to(shift_logits.device)
        shift_align_coarse = align_coarse_labels[..., 1:].contiguous().to(shift_logits.device)

        active_mask = (
            shift_align_positions.ne(IGNORE_INDEX)
            & shift_align_coarse.ne(IGNORE_INDEX)
        )

        if not active_mask.any():
            return logits.new_zeros(())

        flat_logits = shift_logits.view(-1, shift_logits.size(-1))
        flat_positions = shift_align_positions.view(-1)
        flat_coarse = shift_align_coarse.view(-1)
        flat_active = active_mask.view(-1)

        unique_positions = torch.unique(flat_positions[flat_active]).detach().cpu().tolist()

        losses = []

        for pos in unique_positions:
            pos = int(pos)
            codebook = self._get_align_codebook_tensors(pos, shift_logits.device)

            if codebook is None:
                continue

            candidate_fine_ids = codebook["fine_ids"]
            candidate_coarse_ids = codebook["coarse_ids"]

            row_mask = flat_active & flat_positions.eq(pos)
            row_indices = torch.nonzero(row_mask, as_tuple=False).view(-1)

            if row_indices.numel() == 0:
                continue

            row_logits = flat_logits.index_select(0, row_indices)
            candidate_logits = row_logits.index_select(1, candidate_fine_ids).float()

            # Denominator: all possible fine tokens at this code position.
            denom = torch.logsumexp(candidate_logits, dim=-1)

            target_coarse = flat_coarse.index_select(0, row_indices)
            group_mask = candidate_coarse_ids.unsqueeze(0).eq(target_coarse.unsqueeze(1))

            neg_inf = torch.finfo(candidate_logits.dtype).min
            group_logits = candidate_logits.masked_fill(~group_mask, neg_inf)

            # Numerator: fine-token candidates belonging to the gold coarse bucket.
            numer = torch.logsumexp(group_logits, dim=-1)

            valid = torch.isfinite(numer)
            if valid.any():
                losses.append(-(numer[valid] - denom[valid]))

        if len(losses) == 0:
            return logits.new_zeros(())

        return torch.cat(losses, dim=0).mean().to(dtype=logits.dtype)

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        loss_weights = inputs.pop("loss_weights", None)
        align_positions = inputs.pop("align_positions", None)
        align_coarse_labels = inputs.pop("align_coarse_labels", None)

        outputs = model(**inputs, return_dict=True)
        logits = outputs.logits

        token_loss = self._compute_weighted_token_loss(
            logits=logits,
            labels=labels,
            loss_weights=loss_weights,
        )

        align_loss = self._compute_coarse_align_loss(
            logits=logits,
            align_positions=align_positions,
            align_coarse_labels=align_coarse_labels,
        )

        loss = token_loss + self.coarse_align_weight * align_loss

        return (loss, outputs) if return_outputs else loss


def build_training_args(args, ddp: bool):
    report_to = []
    if args.report_to and args.report_to.lower() != "none":
        report_to = [x.strip() for x in args.report_to.split(",") if x.strip()]

    ta_kwargs = dict(
        seed=args.seed,
        per_device_train_batch_size=args.per_device_batch_size,
        per_device_eval_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        warmup_ratio=args.warmup_ratio,
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        lr_scheduler_type=args.lr_scheduler_type,
        report_to=report_to,
        fp16=args.fp16,
        bf16=args.bf16,
        logging_steps=args.logging_step,
        optim=args.optim,
        gradient_checkpointing=args.gradient_checkpointing,
        save_strategy=args.save_and_eval_strategy,
        eval_steps=args.save_and_eval_steps,
        save_steps=args.save_and_eval_steps,
        output_dir=args.output_dir,
        save_total_limit=args.save_total_limit,
        load_best_model_at_end=True,
        deepspeed=args.deepspeed,
        ddp_find_unused_parameters=False if ddp else None,
        eval_delay=1 if args.save_and_eval_strategy == "epoch" else 2000,
        remove_unused_columns=False,
    )

    sig = inspect.signature(transformers.TrainingArguments.__init__).parameters

    if "evaluation_strategy" in sig:
        ta_kwargs["evaluation_strategy"] = args.save_and_eval_strategy
    else:
        ta_kwargs["eval_strategy"] = args.save_and_eval_strategy

    if "gradient_checkpointing_kwargs" in sig and args.gradient_checkpointing:
        ta_kwargs["gradient_checkpointing_kwargs"] = {"use_reentrant": False}

    return transformers.TrainingArguments(**ta_kwargs)


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


def prepare_model_for_training(model, args):
    if args.load_in_4bit or args.load_in_8bit:
        try:
            model = prepare_model_for_kbit_training(
                model,
                use_gradient_checkpointing=args.gradient_checkpointing,
            )
        except TypeError:
            model = prepare_model_for_kbit_training(model)

    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False

    if args.gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        try:
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
        except TypeError:
            model.gradient_checkpointing_enable()

    return model


def train(args):
    set_seed(args.seed)
    ensure_dir(args.output_dir)

    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    local_rank = int(os.environ.get("LOCAL_RANK") or 0)

    if local_rank == 0:
        print(vars(args))

    device_map = {"": local_rank} if ddp else "auto"

    compute_dtype = torch.float32
    if args.bf16:
        compute_dtype = torch.bfloat16
    elif args.fp16:
        compute_dtype = torch.float16

    config = AutoConfig.from_pretrained(
        args.base_model,
        trust_remote_code=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model,
        model_max_length=args.model_max_length,
        padding_side="right",
        trust_remote_code=True,
        use_fast=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)

    train_data, valid_data = load_datasets(args)

    if not hasattr(train_data, "datasets") or len(train_data.datasets) == 0:
        raise ValueError("train_data should be a ConcatDataset with at least one dataset.")

    first_train_dataset = train_data.datasets[0]

    add_num = tokenizer.add_tokens(first_train_dataset.get_new_tokens())
    config.vocab_size = len(tokenizer)

    if local_rank == 0:
        print(f"add {add_num} new token.")
        print("train data num:", len(train_data))
        print("valid data num:", len(valid_data))
        print("vocab size:", len(tokenizer))

    tokenizer.save_pretrained(args.output_dir)
    config.save_pretrained(args.output_dir)

    collator = Collator(args, tokenizer)

    align_codebooks = build_align_codebooks_from_dataset(
        dataset=first_train_dataset,
        tokenizer=tokenizer,
    )

    if local_rank == 0:
        print("fast mode: one forward for prompt + interleaved coarse/fine labels")
        print("coarse positions:", len(align_codebooks))
        print("coarse_loss_weight:", args.coarse_loss_weight)
        print("fine_loss_weight:", args.fine_loss_weight)
        print("coarse_align_weight:", args.coarse_align_weight)

    quant_config = make_quant_config(args, compute_dtype)

    model_kwargs = dict(
        device_map=device_map,
        trust_remote_code=True,
    )

    if args.fp16 or args.bf16:
        model_kwargs["torch_dtype"] = compute_dtype

    if quant_config is not None:
        model_kwargs["quantization_config"] = quant_config

    if args.attn_implementation:
        model_kwargs["attn_implementation"] = args.attn_implementation

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        **model_kwargs,
    )

    model.resize_token_embeddings(len(tokenizer))

    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False

    model = prepare_model_for_training(model, args)

    lora_targets = (
        [x.strip() for x in args.lora_target_modules.split(",") if x.strip()]
        if args.lora_target_modules
        else [
            "qkv_proj",
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
    )

    modules_to_save = (
        [x.strip() for x in args.lora_modules_to_save.split(",") if x.strip()]
        if args.lora_modules_to_save
        else ["embed_tokens", "lm_head"]
    )

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=lora_targets,
        modules_to_save=modules_to_save,
        lora_dropout=args.lora_dropout,
        bias="none",
        inference_mode=False,
        task_type=TaskType.CAUSAL_LM,
    )

    model = get_peft_model(model, lora_config)

    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False

    if args.resume_from_checkpoint:
        checkpoint_name = os.path.join(args.resume_from_checkpoint, "adapter_model.bin")
        args.resume_from_checkpoint = False

        if os.path.exists(checkpoint_name):
            if local_rank == 0:
                print(f"Restarting from {checkpoint_name}")
            adapters_weights = torch.load(checkpoint_name, map_location="cpu")
            set_peft_model_state_dict(model, adapters_weights)
        else:
            if local_rank == 0:
                print(f"Checkpoint {checkpoint_name} not found")

    if local_rank == 0:
        model.print_trainable_parameters()

    if not ddp and torch.cuda.device_count() > 1:
        model.is_parallelizable = True
        model.model_parallel = True

    training_args = build_training_args(args, ddp)

    trainer = HoloRecFastAlignTrainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=valid_data,
        args=training_args,
        tokenizer=tokenizer,
        data_collator=collator,
        align_codebooks=align_codebooks,
        coarse_align_weight=args.coarse_align_weight,
    )

    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.save_state()
    trainer.save_model(output_dir=args.output_dir)

    if local_rank == 0:
        print("Save model to:", args.output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fast HoloRec-Qwen QLoRA with coarse alignment")

    parser = parse_global_args(parser)
    parser = parse_train_args(parser)
    parser = parse_dataset_args(parser)

    parser.add_argument("--coarse_index_file", type=str, default=".tw8.json")

    parser.add_argument("--coarse_loss_weight", type=float, default=1.0)
    parser.add_argument("--fine_loss_weight", type=float, default=1.0)
    parser.add_argument("--coarse_align_weight", type=float, default=2.0)

    parser.add_argument("--add_eos_token", action="store_true", default=True)
    parser.add_argument("--no_add_eos_token", action="store_false", dest="add_eos_token")

    parser.add_argument("--load_in_4bit", action="store_true", default=True)
    parser.add_argument("--no_load_in_4bit", action="store_false", dest="load_in_4bit")
    parser.add_argument("--load_in_8bit", action="store_true", default=False)

    parser.add_argument("--gradient_checkpointing", action="store_true", default=True)
    parser.add_argument(
        "--no_gradient_checkpointing",
        action="store_false",
        dest="gradient_checkpointing",
    )

    parser.add_argument(
        "--attn_implementation",
        type=str,
        default="flash_attention_2",
        help="Use flash_attention_2 if installed; set empty string to disable.",
    )

    parser.add_argument("--report_to", type=str, default="wandb")
    parser.add_argument("--save_total_limit", type=int, default=8)

    args = parser.parse_args()

    if args.load_in_4bit and args.load_in_8bit:
        raise ValueError("Choose only one of --load_in_4bit and --load_in_8bit.")

    if args.tasks.lower() != "seqrec":
        raise ValueError(
            "Fast HoloRec-Qwen currently supports only --tasks seqrec. "
            "Use: --tasks seqrec --train_prompt_sample_num 1 --train_data_sample_num 0"
        )

    train(args)
