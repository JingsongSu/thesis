import argparse
import inspect
import os
import sys

import torch
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
from modeling_qwen_latent import InterleavedLatentQwen


def patch_transformers_peft_detection_for_latent_wrapper():
    """
    Transformers Trainer refuses to train quantized models unless the top-level
    model passes `_is_peft_model`.

    Our top-level model is InterleavedLatentQwen, while the real PeftModel is
    stored inside `model.base_model`. The default Trainer check can therefore
    incorrectly treat it as a purely quantized model.

    This patch makes Trainer accept wrappers that contain a PeftModel submodule.
    It must be called on every DDP rank before constructing transformers.Trainer.
    """
    try:
        import transformers.trainer as trainer_mod
        import transformers.trainer_utils as trainer_utils_mod
        from peft import PeftModel

        try:
            from peft import PeftMixedModel

            peft_types = (PeftModel, PeftMixedModel)
        except Exception:
            peft_types = (PeftModel,)

        def _is_peft_model_or_contains_peft(model):
            if isinstance(model, peft_types):
                return True

            inner = getattr(model, "base_model", None)
            if isinstance(inner, peft_types):
                return True

            if inner is not None:
                inner_inner = getattr(inner, "base_model", None)
                if isinstance(inner_inner, peft_types):
                    return True

            modules_fn = getattr(model, "modules", None)
            if callable(modules_fn):
                for module in modules_fn():
                    if module is model:
                        continue
                    if isinstance(module, peft_types):
                        return True

            return False

        trainer_mod._is_peft_model = _is_peft_model_or_contains_peft
        trainer_utils_mod._is_peft_model = _is_peft_model_or_contains_peft
    except Exception as e:
        print("[warning] failed to patch Trainer PEFT detection:", repr(e))


class HoloRecTrainer(transformers.Trainer):
    """
    Fix for HoloRec-Qwen eval_loss missing.

    Why this is needed:
    - The batch uses fine_labels/coarse_labels instead of the standard labels.
    - Some transformers versions do not treat these fields as labels during eval.
    - Then evaluation returns only eval_runtime / speed metrics.
    - load_best_model_at_end=True then looks for eval_loss and crashes.

    This trainer explicitly computes eval loss from model(**inputs) when the
    HoloRec label fields exist.
    """

    holo_label_names = ["fine_labels", "coarse_labels"]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.label_names = list(self.holo_label_names)

    def prediction_step(
        self,
        model,
        inputs,
        prediction_loss_only,
        ignore_keys=None,
    ):
        has_holorec_labels = all(name in inputs for name in self.holo_label_names)

        if not has_holorec_labels:
            return super().prediction_step(
                model=model,
                inputs=inputs,
                prediction_loss_only=prediction_loss_only,
                ignore_keys=ignore_keys,
            )

        inputs = self._prepare_inputs(inputs)

        with torch.no_grad():
            with self.compute_loss_context_manager():
                loss, outputs = self.compute_loss(
                    model,
                    inputs,
                    return_outputs=True,
                )

            if loss is None:
                raise ValueError(
                    "Model did not return loss during evaluation. "
                    "Please check InterleavedLatentQwen.forward()."
                )

            loss = loss.mean().detach()

        # We only need eval_loss for checkpoint selection.
        # Returning logits here can be very memory-heavy for [B, L, V].
        return loss, None, None

    def _determine_best_metric(self, metrics=None, trial=None, *args, **kwargs):
        """
        Compatibility guard.

        Normal case: eval_loss exists and the parent Trainer handles best model.
        Guard case: if a future code path still produces no eval_loss, do not
        crash checkpoint saving; just skip best-model update for that eval.
        """
        if metrics is None:
            metrics = kwargs.get("metrics", None)
        if metrics is None and len(args) > 0:
            metrics = args[0]

        if isinstance(metrics, dict):
            metric_for_best_model = getattr(self.args, "metric_for_best_model", None)
            if metric_for_best_model is None:
                metric_for_best_model = "eval_loss"

            metric_to_check = metric_for_best_model
            if not metric_to_check.startswith("eval_"):
                metric_to_check = f"eval_{metric_to_check}"

            if metric_to_check not in metrics:
                print(
                    "[warning] metric_for_best_model="
                    f"{metric_to_check!r} not found in eval metrics. "
                    f"Available metrics: {list(metrics.keys())}. "
                    "Skip best-model update for this checkpoint."
                )
                return False

        try:
            return super()._determine_best_metric(
                metrics=metrics,
                trial=trial,
            )
        except TypeError:
            return super()._determine_best_metric(metrics, trial)


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


def build_coarse_position_token_ids_from_dataset(dataset, tokenizer):
    """
    dataset should be SeqRecDataset.

    Returns:
        List[List[int]]

    coarse_position_token_ids[pos] contains all allowed coarse token ids at
    latent coarse position `pos`.
    """
    if not hasattr(dataset, "coarse_indices") or dataset.coarse_indices is None:
        raise ValueError(
            "Dataset has no coarse_indices. "
            "Please pass --coarse_index_file, for example "
            "--coarse_index_file .tw8.json"
        )

    pos_to_ids = {}

    for _, seq in dataset.coarse_indices.items():
        for pos, tok in enumerate(seq):
            tid = token_to_id(tokenizer, tok)
            pos_to_ids.setdefault(pos, set()).add(tid)

    if len(pos_to_ids) == 0:
        raise ValueError("coarse position token ids are empty.")

    max_pos = max(pos_to_ids.keys())
    return [sorted(list(pos_to_ids.get(i, set()))) for i in range(max_pos + 1)]


def prepare_kbit_model_for_latent_cache(model):
    """
    Keep k-bit preparation, but do not enable gradient checkpointing.

    The latent wrapper calls the same base model multiple times inside one
    forward. Gradient checkpointing plus repeated inner model calls can trigger
    parameter reduction conflicts, especially with DeepSpeed ZeRO.
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
        model.config.use_cache = True

    return model


def build_training_args(args, ddp: bool):
    """
    Compatible with different transformers versions:
    - some versions use evaluation_strategy
    - some versions use eval_strategy

    Important:
    - gradient_checkpointing=False because latent wrapper repeatedly calls model.
    - remove_unused_columns=False because Trainer must keep fine_labels/coarse_labels.
    - label_names tells Trainer that fine_labels/coarse_labels are label fields.
    - prediction_loss_only=True avoids gathering huge [B, L, V] logits in eval.
    """

    report_to = []
    if getattr(args, "report_to", "wandb") and args.report_to.lower() != "none":
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
        gradient_checkpointing=False,
        save_strategy=args.save_and_eval_strategy,
        eval_steps=args.save_and_eval_steps,
        save_steps=args.save_and_eval_steps,
        output_dir=args.output_dir,
        save_total_limit=args.save_total_limit,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        deepspeed=args.deepspeed,
        ddp_find_unused_parameters=False if ddp else None,
        eval_delay=1 if args.save_and_eval_strategy == "epoch" else 2000,
        remove_unused_columns=False,
        label_names=["fine_labels", "coarse_labels"],
        prediction_loss_only=True,
    )

    sig = inspect.signature(transformers.TrainingArguments.__init__).parameters

    if "evaluation_strategy" in sig:
        ta_kwargs["evaluation_strategy"] = args.save_and_eval_strategy
    else:
        ta_kwargs["eval_strategy"] = args.save_and_eval_strategy

    # Keep compatibility with older transformers versions.
    ta_kwargs = {k: v for k, v in ta_kwargs.items() if k in sig}

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


def split_arg_list(value):
    if value is None:
        return []
    return [x.strip() for x in value.split(",") if x.strip()]


def expose_peft_metadata_to_wrapper(wrapper_model, peft_model):
    """
    Make the latent wrapper look adapter-aware to utilities that inspect
    PEFT-related attributes on the top-level model.
    """
    if hasattr(peft_model, "peft_config"):
        wrapper_model.peft_config = peft_model.peft_config

    if hasattr(peft_model, "active_adapter"):
        wrapper_model.active_adapter = peft_model.active_adapter

    if hasattr(peft_model, "active_adapters"):
        wrapper_model.active_adapters = peft_model.active_adapters

    wrapper_model._hf_peft_config_loaded = True
    return wrapper_model


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
        padding_side="left",
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

    # Add both fine and coarse tokens.
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

    quant_config = make_quant_config(args, compute_dtype)

    model_kwargs = dict(
        device_map=device_map,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )

    if quant_config is not None:
        model_kwargs["quantization_config"] = quant_config

    if args.fp16 or args.bf16:
        model_kwargs["torch_dtype"] = compute_dtype

    if args.attn_implementation:
        model_kwargs["attn_implementation"] = args.attn_implementation

    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        **model_kwargs,
    )

    base_model.resize_token_embeddings(len(tokenizer))

    if hasattr(base_model.config, "use_cache"):
        base_model.config.use_cache = True

    if args.load_in_4bit or args.load_in_8bit:
        base_model = prepare_kbit_model_for_latent_cache(base_model)

    if hasattr(base_model.config, "use_cache"):
        base_model.config.use_cache = True

    lora_targets = split_arg_list(args.lora_target_modules)
    if len(lora_targets) == 0:
        lora_targets = [
            "qkv_proj",
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]

    modules_to_save = split_arg_list(args.lora_modules_to_save)
    if len(modules_to_save) == 0:
        # Needed because new fine/coarse code tokens are added to tokenizer.
        # Without this, resized embeddings / lm_head may stay frozen and may
        # not be saved into the LoRA checkpoint.
        modules_to_save = ["embed_tokens", "lm_head"]

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

    peft_model = get_peft_model(base_model, lora_config)

    if hasattr(peft_model.config, "use_cache"):
        peft_model.config.use_cache = True

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
        print("use_train_cache:", not args.no_train_cache)
        print("mode: implicit latent interleaved training; coarse tokens are not decoded.")

    model = InterleavedLatentQwen(
        base_model=peft_model,
        pad_token_id=tokenizer.pad_token_id,
        temperature=args.temperature,
        coarse_loss_weight=args.coarse_loss_weight,
        fine_loss_weight=args.fine_loss_weight,
        coarse_align_weight=args.coarse_align_weight,
        use_train_cache=not args.no_train_cache,
    )

    model.set_codebooks(coarse_position_token_ids)

    if hasattr(model.config, "use_cache"):
        model.config.use_cache = True

    model.gradient_checkpointing_disable()
    model = expose_peft_metadata_to_wrapper(model, peft_model)

    # Important: call this before constructing Trainer on every DDP rank.
    patch_transformers_peft_detection_for_latent_wrapper()

    training_args = build_training_args(args, ddp)

    trainer = HoloRecTrainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=valid_data,
        args=training_args,
        tokenizer=tokenizer,
        data_collator=collator,
    )

    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.save_state()

    # Save PEFT adapter directly, not the latent wrapper state dict.
    peft_model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    config.save_pretrained(args.output_dir)

    if local_rank == 0:
        print("Save model to:", args.output_dir)


def add_arg_if_absent(parser, *flags, **kwargs):
    existing = set()
    for action in parser._actions:
        existing.update(action.option_strings)

    if any(flag in existing for flag in flags):
        return parser

    parser.add_argument(*flags, **kwargs)
    return parser


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Implicit Latent HoloRec-Qwen")

    parser = parse_global_args(parser)
    parser = parse_train_args(parser)
    parser = parse_dataset_args(parser)

    add_arg_if_absent(
        parser,
        "--coarse_index_file",
        type=str,
        default=".tw8.json",
    )
    add_arg_if_absent(
        parser,
        "--coarse_loss_weight",
        type=float,
        default=1.0,
    )
    add_arg_if_absent(
        parser,
        "--fine_loss_weight",
        type=float,
        default=1.0,
    )
    add_arg_if_absent(
        parser,
        "--coarse_align_weight",
        type=float,
        default=2.0,
    )
    add_arg_if_absent(
        parser,
        "--temperature",
        type=float,
        default=1.0,
    )
    add_arg_if_absent(
        parser,
        "--load_in_8bit",
        action="store_true",
        default=True,
    )
    add_arg_if_absent(
        parser,
        "--no_load_in_8bit",
        action="store_false",
        dest="load_in_8bit",
    )
    add_arg_if_absent(
        parser,
        "--load_in_4bit",
        action="store_true",
        default=False,
    )
    add_arg_if_absent(
        parser,
        "--attn_implementation",
        type=str,
        default="flash_attention_2",
    )
    add_arg_if_absent(
        parser,
        "--report_to",
        type=str,
        default="wandb",
    )
    add_arg_if_absent(
        parser,
        "--save_total_limit",
        type=int,
        default=8,
    )
    add_arg_if_absent(
        parser,
        "--no_train_cache",
        action="store_true",
        default=False,
        help="Disable cached latent training and use slower full-prefix recomputation.",
    )

    args = parser.parse_args()

    if args.load_in_4bit and args.load_in_8bit:
        raise ValueError(
            "Choose only one of --load_in_4bit and --load_in_8bit. "
            "If you want 4-bit, pass: --load_in_4bit --no_load_in_8bit"
        )

    if args.tasks.lower() != "seqrec":
        raise ValueError(
            "Implicit latent HoloRec-Qwen training currently supports only --tasks seqrec. "
            "Please run with: --tasks seqrec --train_prompt_sample_num 1 "
            "--train_data_sample_num 0"
        )

    train(args)
