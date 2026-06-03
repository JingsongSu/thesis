import argparse
import os
import torch
import transformers
from transformers import EarlyStoppingCallback
from transformers import T5Tokenizer, T5Config

from modeling_letter import LETTER
from utils import *
from collator import Collator


def train(args):
    print(torch.cuda.is_available())

    set_seed(args.seed)
    ensure_dir(args.output_dir)

    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    local_rank = int(os.environ.get("LOCAL_RANK") or 0)

    if local_rank == 0:
        print(vars(args))

    if ddp:
        torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    config = T5Config.from_pretrained(args.base_model)
    tokenizer = T5Tokenizer.from_pretrained(
        args.base_model,
        model_max_length=512,
    )

    args.deepspeed = None

    train_data, valid_data = load_datasets(args)

    # 同时加入 coarse / fine token
    add_num = tokenizer.add_tokens(train_data.datasets[0].get_new_tokens())
    config.vocab_size = len(tokenizer)

    if local_rank == 0:
        print("add {} new token.".format(add_num))
        print("train data num:", len(train_data))
        print("valid data num:", len(valid_data))
        tokenizer.save_pretrained(args.output_dir)
        config.save_pretrained(args.output_dir)
        print(train_data[0])
        print(valid_data[0])

    collator = Collator(args, tokenizer)

    # 从 base_model 加载预训练权重，再替换/扩展新 token
    model = LETTER(config)

    if hasattr(model, "resize_token_embeddings_and_heads"):
        model.resize_token_embeddings_and_heads(len(tokenizer))
    else:
        model.resize_token_embeddings(len(tokenizer))

    model.set_hyper(
        temperature=args.temperature,
        coarse_loss_weight=args.coarse_loss_weight,
        fine_loss_weight=args.fine_loss_weight,
        coarse_align_weight=args.coarse_align_weight,
    )
    model.to(device)

    if local_rank == 0:
        print(model)

    trainer = transformers.Trainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=valid_data,
        args=transformers.TrainingArguments(
            seed=args.seed,
            per_device_train_batch_size=args.per_device_batch_size,
            per_device_eval_batch_size=args.per_device_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            warmup_ratio=args.warmup_ratio,
            num_train_epochs=args.epochs,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            lr_scheduler_type=args.lr_scheduler_type,
            logging_steps=args.logging_step,
            optim=args.optim,
            evaluation_strategy=args.save_and_eval_strategy,
            save_strategy=args.save_and_eval_strategy,
            eval_steps=args.save_and_eval_steps,
            save_steps=args.save_and_eval_steps,
            output_dir=args.output_dir,
            save_total_limit=2,
            load_best_model_at_end=True,
            ddp_find_unused_parameters=False if ddp else None,
            eval_delay=1 if args.save_and_eval_strategy == "epoch" else 2000,
            remove_unused_columns=False,
            report_to=[],
        ),
        tokenizer=tokenizer,
        data_collator=collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=20)]
    )

    model.config.use_cache = False

    trainer.train(
        resume_from_checkpoint=args.resume_from_checkpoint,
    )

    trainer.save_state()
    trainer.save_model(output_dir=args.output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='LLMRec')

    parser = parse_global_args(parser)
    parser = parse_train_args(parser)
    parser = parse_dataset_args(parser)

    # 只保留一份 coarse_index_file
    parser.add_argument(
        "--coarse_index_file",
        type=str,
        default=".index.xinyan32.epoch10000.alpha2e-2-beta1e-4.json"
    )

    parser.add_argument("--coarse_loss_weight", type=float, default=1.0)
    parser.add_argument("--fine_loss_weight", type=float, default=1.0)
    parser.add_argument("--coarse_align_weight", type=float, default=2.0)

    args = parser.parse_args()

    train(args)
