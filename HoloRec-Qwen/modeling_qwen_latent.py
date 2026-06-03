import argparse
import os
import sys
import inspect

import torch
import transformers
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

from peft import (
    TaskType,
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    set_peft_model_state_dict,
)

from utils import *
from collator import Collator
from modeling_qwen_latent import InterleavedLatentQwen


def token_to_id(tokenizer, token):
    token_id = tokenizer.convert_tokens_to_ids(token)

    if token_id is None:
        encoded = tokenizer(token, add_special_tokens=False).get("input_ids", [])
        if len(encoded) == 1:
            return int(encoded[0])
        raise ValueError(f"Token {token!r} is not a single token: {encoded}")

    unk_id = getattr(tokenizer, "unk_token_id", None)
    if unk_id is not None and token_id == unk_id:
        encoded = tokenizer(token, add_special_tokens=False).get("input_ids", [])
        if len(encoded) == 1:
            return int(encoded[0])
        raise ValueError(f"Token {token!r} maps to unk or multiple ids: {encoded}")

    return int(token_id)


def build_coarse_position_token_ids_from_dataset(dataset, tokenizer):
    """
    dataset should be SeqRecDataset.
    """
    if not hasattr(dataset, "coarse_indices") or dataset.coarse_indices is None:
        raise ValueError(
            "Dataset has no coarse_indices. "
            "Please pass --coarse_index_file, for example --coarse_index_file .tw8.json"
        )

    pos_to_ids = {}

    for _, seq in dataset.coarse_indices.items():
        for pos, tok in enumerate(seq):
            tid = token_to_id(tokenizer, tok)
            pos_to_ids.setdefault(pos, set()).add(tid)

    if len(pos_to_ids) == 0:
        raise ValueError("coarse position token ids are empty.")

    max_pos = max(pos_to_ids.keys())

    return [
        sorted(list(pos_to_ids.get(i, set())))
        for i in range(max_pos + 1)
    ]


def prepare_kbit_model_without_gradient_checkpointing(model):
    """
    Original HoloRec-Qwen uses prepare_model_for_kbit_training(model).

    For this latent wrapper, we must avoid gradient checkpointing because
    the wrapper calls the same inner model multiple times in a single forward.
    DeepSpeed ZeRO-1/2 + gradient checkpointing can trigger:

        "parameter has already been reduced"

    This helper keeps k-bit preparation but disables gradient checkpointing.
    """
    try:
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=False,
        )
    except TypeError:
        model = prepare_model_for_kbit_training(model)

    if hasattr(model, "gradient_checkpointing_disable"):
        try:
            model.gradient_checkpointing_disable()
        except Exception:
            pass

    if hasattr(model, "config") and hasattr(model.config, "use_cache"):
        model.config.use_cache = False

    return model


def build_training_args(args, ddp: bool):
    """
    Compatible with different transformers versions:
    - some versions use evaluation_strategy
    - some versions use eval_strategy

    Important changes for latent coarse-to-fine training:
    - gradient_checkpointing=False
    - remove_unused_columns=False
    """

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
        report_to=["wandb"],
        fp16=args.fp16,
        bf16=args.bf16,
        logging_steps=args.logging_step,
        optim=args.optim,

        # Critical:
        # This wrapper calls the inner Qwen/PEFT model multiple times in one forward.
        # DeepSpeed ZeRO-1/2 + gradient checkpointing may reduce the same parameter twice.
        gradient_checkpointing=False,

        save_strategy=args.save_and_eval_strategy,
        eval_steps=args.save_and_eval_steps,
        save_steps=args.save_and_eval_steps,
        output_dir=args.output_dir,
        save_total_limit=8,
        load_best_model_at_end=True,
        deepspeed=args.deepspeed,
        ddp_find_unused_parameters=False if ddp else None,
        eval_delay=1 if args.save_and_eval_strategy == "epoch" else 2000,

        # Critical:
        # Trainer must not drop fine_labels / coarse_labels.
        remove_unused_columns=False,
    )

    sig = inspect.signature(transformers.TrainingArguments.__init__).parameters

    if "evaluation_strategy" in sig:
        ta_kwargs["evaluation_strategy"] = args.save_and_eval_strategy
    else:
        ta_kwargs["eval_strategy"] = args.save_and_eval_strategy

    return transformers.TrainingArguments(**ta_kwargs)


def train(args):
    set_seed(args.seed)
    ensure_dir(args.output_dir)

    device_map = "auto"
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    local_rank = int(os.environ.get("LOCAL_RANK") or 0)

    if local_rank == 0:
        print(vars(args))

    if ddp:
        device_map = {"": local_rank}

    # ===== 1) config / tokenizer =====
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

    # ===== 2) dataset =====
    train_data, valid_data = load_datasets(args)

    # This code assumes --tasks seqrec, so train_data.datasets[0] is SeqRecDataset.
    if not hasattr(train_data, "datasets") or len(train_data.datasets) == 0:
        raise ValueError("train_data should be a ConcatDataset with at least one dataset.")

    first_train_dataset = train_data.datasets[0]

    # Add both fine + coarse tokens.
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

    # ===== 3) quantization config, same as original script =====
    quant_config = BitsAndBytesConfig(
        load_in_8bit=True,
        llm_int8_threshold=6.0,
        llm_int8_has_fp16_weight=False,
    )

    # ===== 4) load base model =====
    dtype = torch.float16 if args.fp16 else (torch.bfloat16 if args.bf16 else None)

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        device_map=device_map,
        quantization_config=quant_config,
        torch_dtype=dtype,
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
    )

    model.resize_token_embeddings(len(tokenizer))

    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False

    # ===== 5) k-bit training preparation =====
    # Important: no gradient checkpointing here.
    model = prepare_kbit_model_without_gradient_checkpointing(model)

    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False

    # ===== 6) LoRA config =====
    lora_targets = (
        args.lora_target_modules.split(",")
        if args.lora_target_modules
        else ["qkv_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )

    modules_to_save = (
        args.lora_modules_to_save.split(",")
        if args.lora_modules_to_save
        else []
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

    peft_model = get_peft_model(model, lora_config)

    if hasattr(peft_model.config, "use_cache"):
        peft_model.config.use_cache = False

    # ===== 7) resume LoRA adapter =====
    if args.resume_from_checkpoint:
        checkpoint_name = os.path.join(args.resume_from_checkpoint, "adapter_model.bin")
        args.resume_from_checkpoint = False

        if os.path.exists(checkpoint_name):
            if local_rank == 0:
                print(f"Restarting from {checkpoint_name}")

            adapters_weights = torch.load(checkpoint_name, map_location="cpu")
            set_peft_model_state_dict(peft_model, adapters_weights)
        else:
            if local_rank == 0:
                print(f"Checkpoint {checkpoint_name} not found")

    if local_rank == 0:
        peft_model.print_trainable_parameters()

    if not ddp and torch.cuda.device_count() > 1:
        peft_model.is_parallelizable = True
        peft_model.model_parallel = True

    # ===== 8) build coarse codebook ids =====
    coarse_position_token_ids = build_coarse_position_token_ids_from_dataset(
        first_train_dataset,
        tokenizer,
    )

    if local_rank == 0:
        print("coarse positions:", len(coarse_position_token_ids))
        print("coarse_loss_weight:", args.coarse_loss_weight)
        print("fine_loss_weight:", args.fine_loss_weight)
        print("coarse_align_weight:", args.coarse_align_weight)
        print("temperature:", args.temperature)

    # ===== 9) wrap PEFT model with latent coarse-to-fine training module =====
    model = InterleavedLatentQwen(
        base_model=peft_model,
        pad_token_id=tokenizer.pad_token_id,
        temperature=args.temperature,
        coarse_loss_weight=args.coarse_loss_weight,
        fine_loss_weight=args.fine_loss_weight,
        coarse_align_weight=args.coarse_align_weight,
    )

    model.set_codebooks(coarse_position_token_ids)

    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False

    # Do not enable gradient checkpointing on this wrapper.
    model.gradient_checkpointing_disable()

    # ===== 10) Trainer =====
    training_args = build_training_args(args, ddp)

    trainer = transformers.Trainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=valid_data,
        args=training_args,
        tokenizer=tokenizer,
        data_collator=collator,
    )

    # Keep original behavior. This line is harmless because Trainer already holds model.
    # If torch.compile causes issues in your environment, comment it out.
    if torch.__version__ >= "2" and sys.platform != "win32":
        try:
            model = torch.compile(model)
        except Exception as e:
            if local_rank == 0:
                print("torch.compile skipped because:", repr(e))

    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    trainer.save_state()
    trainer.save_model(output_dir=args.output_dir)

    if local_rank == 0:
        print("Save model to:", args.output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLMRec")

    parser = parse_global_args(parser)
    parser = parse_train_args(parser)
    parser = parse_dataset_args(parser)

    # Coarse codebook. For your current setting, pass --coarse_index_file .tw8.json.
    parser.add_argument("--coarse_index_file", type=str, default=".tw8.json")

    # Latent coarse-to-fine loss weights.
    parser.add_argument("--coarse_loss_weight", type=float, default=1.0)
    parser.add_argument("--fine_loss_weight", type=float, default=1.0)
    parser.add_argument("--coarse_align_weight", type=float, default=2.0)

    args = parser.parse_args()

    if args.tasks.lower() != "seqrec":
        raise ValueError(
            "Latent coarse-to-fine training currently supports only --tasks seqrec. "
            "Please run with: --tasks seqrec --train_prompt_sample_num 1 --train_data_sample_num 0"
        )

    train(args)
