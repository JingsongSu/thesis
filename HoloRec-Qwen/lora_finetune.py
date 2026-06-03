
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


def build_training_args(args, ddp: bool):
    """
    兼容不同 transformers 版本：
    - 有的版本用 evaluation_strategy
    - 有的版本用 eval_strategy
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
        gradient_checkpointing=True,
        save_strategy=args.save_and_eval_strategy,
        eval_steps=args.save_and_eval_steps,
        save_steps=args.save_and_eval_steps,
        output_dir=args.output_dir,
        save_total_limit=8,
        load_best_model_at_end=True,
        deepspeed=args.deepspeed,
        ddp_find_unused_parameters=False if ddp else None,
        eval_delay=1 if args.save_and_eval_strategy == "epoch" else 2000,
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
    config = AutoConfig.from_pretrained(args.base_model, trust_remote_code=True)

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

    # ===== 2) 数据 =====
    train_data, valid_data = load_datasets(args)

    # 这里会同时加入 fine + coarse token
    add_num = tokenizer.add_tokens(train_data.datasets[0].get_new_tokens())
    config.vocab_size = len(tokenizer)

    if local_rank == 0:
        print(f"add {add_num} new token.")
        print("train data num:", len(train_data))
        print("valid data num:", len(valid_data))
        tokenizer.save_pretrained(args.output_dir)
        config.save_pretrained(args.output_dir)

    collator = Collator(args, tokenizer)

    # ===== 3) 量化加载（8bit）=====
    quant_config = BitsAndBytesConfig(
        load_in_8bit=True,
        llm_int8_threshold=6.0,
        llm_int8_has_fp16_weight=False,
    )

    # ===== 4) 加载模型 =====
    dtype = torch.float16 if args.fp16 else (torch.bfloat16 if args.bf16 else None)

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        device_map=device_map,
        quantization_config=quant_config,
        dtype=dtype,
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
    )

    model.resize_token_embeddings(len(tokenizer))

    # ===== 5) k-bit 训练准备 =====
    model = prepare_model_for_kbit_training(model)

    # ===== 6) LoRA 配置 =====
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
    model = get_peft_model(model, lora_config)

    # ===== 7) 断点恢复（LoRA adapter）=====
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

    trainer = transformers.Trainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=valid_data,
        args=training_args,
        tokenizer=tokenizer,
        data_collator=collator,
    )

    model.config.use_cache = False

    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)

    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.save_state()
    trainer.save_model(output_dir=args.output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLMRec")
    parser = parse_global_args(parser)
    parser = parse_train_args(parser)
    parser = parse_dataset_args(parser)

    # 新增：粗粒度码本
    parser.add_argument("--coarse_index_file", type=str, default=".index.4cu32xi.json")

    args = parser.parse_args()
    train(args)
