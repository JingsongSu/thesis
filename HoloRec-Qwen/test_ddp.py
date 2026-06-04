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
from peft import PeftModel, LoraConfig, TaskType, get_peft_model

from utils import *
from collator import TestCollator
from prompt import all_prompt
from evaluate import get_topk_results, get_metrics_results
from generation_trie import Trie, build_trie_from_token_sequences

try:
    from safetensors.torch import load_file as safe_load_file
except Exception:
    safe_load_file = None


def _rank0_print(*msg):
    if int(os.environ.get("LOCAL_RANK") or 0) == 0:
        print(*msg)


def _split_arg_list(value):
    if value is None:
        return []

    if isinstance(value, (list, tuple, set)):
        return [str(x).strip() for x in value if str(x).strip()]

    value = str(value).strip()

    if not value:
        return []

    return [x.strip() for x in value.split(",") if x.strip()]


def _has_tokenizer_files(path):
    if not path or not os.path.isdir(path):
        return False

    names = [
        "tokenizer_config.json",
        "tokenizer.json",
        "vocab.json",
        "merges.txt",
        "special_tokens_map.json",
    ]

    return any(os.path.exists(os.path.join(path, x)) for x in names)


def _find_weight_files(ckpt_path):
    """
    支持以下 checkpoint 结构：

    1. 标准 Trainer / HF checkpoint:
       model.safetensors
       pytorch_model.bin

    2. sharded checkpoint:
       model.safetensors.index.json
       pytorch_model.bin.index.json

    3. PEFT adapter 权重但缺 adapter_config.json:
       adapter_model.safetensors
       adapter_model.bin

    4. 直接传入单个 .safetensors / .bin 文件。
    """
    if os.path.isfile(ckpt_path):
        return [ckpt_path]

    if not os.path.isdir(ckpt_path):
        return []

    index_candidates = [
        "model.safetensors.index.json",
        "pytorch_model.bin.index.json",
        "adapter_model.safetensors.index.json",
        "adapter_model.bin.index.json",
    ]

    for index_name in index_candidates:
        index_path = os.path.join(ckpt_path, index_name)

        if os.path.exists(index_path):
            with open(index_path, "r", encoding="utf-8") as f:
                index = json.load(f)

            weight_map = index.get("weight_map", {})
            files = sorted(set(weight_map.values()))

            return [os.path.join(ckpt_path, x) for x in files]

    single_candidates = [
        "model.safetensors",
        "pytorch_model.bin",
        "adapter_model.safetensors",
        "adapter_model.bin",
    ]

    for name in single_candidates:
        path = os.path.join(ckpt_path, name)

        if os.path.exists(path):
            return [path]

    return []


def _load_state_dict_from_files(weight_files):
    state_dict = {}

    for weight_file in weight_files:
        if weight_file.endswith(".safetensors"):
            if safe_load_file is None:
                raise ImportError(
                    "Need safetensors to load .safetensors checkpoint. "
                    "Please run: pip install safetensors"
                )

            part = safe_load_file(weight_file, device="cpu")

        else:
            part = torch.load(weight_file, map_location="cpu")

            if isinstance(part, dict):
                for key in ["state_dict", "model", "module"]:
                    if key in part and isinstance(part[key], dict):
                        part = part[key]
                        break

        if not isinstance(part, dict):
            raise ValueError(f"Unsupported checkpoint format: {weight_file}")

        state_dict.update(part)

    return state_dict


def _state_dict_has_lora(state_dict):
    for key in state_dict.keys():
        if "lora_A" in key or "lora_B" in key:
            return True

    return False


def _infer_lora_targets_from_state_dict(state_dict):
    targets = set()

    for key in state_dict.keys():
        if ".lora_A." in key and key.endswith(".weight"):
            prefix = key.split(".lora_A.")[0]
            module_name = prefix.split(".")[-1]

            if module_name:
                targets.add(module_name)

        if ".lora_A.weight" in key:
            prefix = key.split(".lora_A.weight")[0]
            module_name = prefix.split(".")[-1]

            if module_name:
                targets.add(module_name)

    return sorted(targets)


def _infer_modules_to_save_from_state_dict(state_dict):
    modules_to_save = set()

    for key in state_dict.keys():
        if ".modules_to_save." in key:
            prefix = key.split(".modules_to_save.")[0]
            module_name = prefix.split(".")[-1]

            if module_name:
                modules_to_save.add(module_name)

        if ".modules_to_save.weight" in key:
            prefix = key.split(".modules_to_save.weight")[0]
            module_name = prefix.split(".")[-1]

            if module_name:
                modules_to_save.add(module_name)

    return sorted(modules_to_save)


def _infer_lora_r_from_state_dict(state_dict):
    for key, value in state_dict.items():
        if ".lora_A." in key and key.endswith(".weight"):
            if hasattr(value, "shape") and len(value.shape) >= 1:
                return int(value.shape[0])

        if ".lora_A.weight" in key:
            if hasattr(value, "shape") and len(value.shape) >= 1:
                return int(value.shape[0])

    return None


def _build_lora_config_for_test(args, raw_state_dict=None):
    """
    和 lora_finetune.py 保持一致：

    - 优先使用 utils.py / parse_train_args 中的 LoRA 参数。
    - 如果命令行参数为空，再从 checkpoint 里推断。
    - 如果仍然推断不到，再使用训练脚本里的 fallback。
    """
    inferred_targets = []
    inferred_modules_to_save = []
    inferred_r = None

    if raw_state_dict is not None:
        inferred_targets = _infer_lora_targets_from_state_dict(raw_state_dict)
        inferred_modules_to_save = _infer_modules_to_save_from_state_dict(raw_state_dict)
        inferred_r = _infer_lora_r_from_state_dict(raw_state_dict)

    target_modules = _split_arg_list(getattr(args, "lora_target_modules", ""))

    if len(target_modules) == 0:
        if len(inferred_targets) > 0:
            target_modules = inferred_targets
        else:
            target_modules = [
                "qkv_proj",
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ]

    modules_to_save = _split_arg_list(getattr(args, "lora_modules_to_save", ""))

    if len(modules_to_save) == 0:
        if len(inferred_modules_to_save) > 0:
            modules_to_save = inferred_modules_to_save
        else:
            modules_to_save = ["embed_tokens", "lm_head"]

    lora_r = getattr(args, "lora_r", None)

    if lora_r is None:
        lora_r = inferred_r

    if lora_r is None:
        lora_r = 16

    lora_alpha = getattr(args, "lora_alpha", 32)
    lora_dropout = getattr(args, "lora_dropout", 0.05)

    _rank0_print("[ckpt loader] LoRA target_modules:", target_modules)
    _rank0_print("[ckpt loader] LoRA modules_to_save:", modules_to_save)
    _rank0_print("[ckpt loader] LoRA r:", lora_r)
    _rank0_print("[ckpt loader] LoRA alpha:", lora_alpha)
    _rank0_print("[ckpt loader] LoRA dropout:", lora_dropout)

    return LoraConfig(
        r=int(lora_r),
        lora_alpha=int(lora_alpha),
        target_modules=target_modules,
        modules_to_save=modules_to_save,
        lora_dropout=float(lora_dropout),
        bias="none",
        inference_mode=True,
        task_type=TaskType.CAUSAL_LM,
    )


def _insert_default_adapter_name(key):
    key = key.replace(".lora_A.weight", ".lora_A.default.weight")
    key = key.replace(".lora_B.weight", ".lora_B.default.weight")
    key = key.replace(".lora_embedding_A.weight", ".lora_embedding_A.default.weight")
    key = key.replace(".lora_embedding_B.weight", ".lora_embedding_B.default.weight")
    key = key.replace(".modules_to_save.weight", ".modules_to_save.default.weight")
    return key


def _remap_key_identity(key):
    return key


def _remap_key_insert_default(key):
    return _insert_default_adapter_name(key)


def _remap_key_strip_module(key):
    if key.startswith("module."):
        key = key[len("module."):]

    return key


def _remap_key_strip_module_insert_default(key):
    return _insert_default_adapter_name(_remap_key_strip_module(key))


def _remap_key_strip_wrapper_prefix(key):
    """
    训练时 Trainer 保存的是 InterleavedLatentQwen wrapper。
    wrapper 内部的真实 PEFT 模型字段名也是 base_model。

    可能出现：
      base_model.base_model.model.model.layers...
    测试时直接构造 PeftModel：
      base_model.model.model.layers...

    因此去掉最外层 wrapper 的 base_model。
    """
    if key.startswith("base_model.base_model."):
        return key[len("base_model."):]

    return key


def _remap_key_strip_wrapper_prefix_insert_default(key):
    return _insert_default_adapter_name(_remap_key_strip_wrapper_prefix(key))


def _remap_key_strip_one_base_model(key):
    if key.startswith("base_model."):
        key = key[len("base_model."):]

    return key


def _remap_key_strip_one_base_model_insert_default(key):
    return _insert_default_adapter_name(_remap_key_strip_one_base_model(key))


def _remap_key_strip_two_base_model(key):
    for _ in range(2):
        if key.startswith("base_model."):
            key = key[len("base_model."):]

    return key


def _remap_key_strip_two_base_model_insert_default(key):
    return _insert_default_adapter_name(_remap_key_strip_two_base_model(key))


def _remap_key_strip_module_then_wrapper_prefix(key):
    key = _remap_key_strip_module(key)
    key = _remap_key_strip_wrapper_prefix(key)
    return key


def _remap_key_strip_module_then_wrapper_prefix_insert_default(key):
    return _insert_default_adapter_name(_remap_key_strip_module_then_wrapper_prefix(key))


def _remap_key_strip_module_then_one_base_model(key):
    key = _remap_key_strip_module(key)
    key = _remap_key_strip_one_base_model(key)
    return key


def _remap_key_strip_module_then_one_base_model_insert_default(key):
    return _insert_default_adapter_name(_remap_key_strip_module_then_one_base_model(key))


def _remap_key_add_peft_prefix(key):
    if key.startswith("module."):
        key = key[len("module."):]

    if key.startswith("base_model."):
        return key

    return "base_model.model." + key


def _remap_key_add_peft_prefix_insert_default(key):
    return _insert_default_adapter_name(_remap_key_add_peft_prefix(key))


def _load_best_matched_state_dict(model, raw_state_dict):
    """
    Trainer checkpoint / wrapper checkpoint / PEFT checkpoint 的 key 前缀可能不同。
    这里尝试多种安全映射，只加载 shape 完全一致的 tensor。
    """
    model_state = model.state_dict()

    remappers = [
        ("identity", _remap_key_identity),
        ("insert_default", _remap_key_insert_default),
        ("strip_module", _remap_key_strip_module),
        ("strip_module_insert_default", _remap_key_strip_module_insert_default),
        ("strip_wrapper_prefix", _remap_key_strip_wrapper_prefix),
        ("strip_wrapper_prefix_insert_default", _remap_key_strip_wrapper_prefix_insert_default),
        ("strip_one_base_model", _remap_key_strip_one_base_model),
        ("strip_one_base_model_insert_default", _remap_key_strip_one_base_model_insert_default),
        ("strip_two_base_model", _remap_key_strip_two_base_model),
        ("strip_two_base_model_insert_default", _remap_key_strip_two_base_model_insert_default),
        ("strip_module_then_wrapper_prefix", _remap_key_strip_module_then_wrapper_prefix),
        (
            "strip_module_then_wrapper_prefix_insert_default",
            _remap_key_strip_module_then_wrapper_prefix_insert_default,
        ),
        ("strip_module_then_one_base_model", _remap_key_strip_module_then_one_base_model),
        (
            "strip_module_then_one_base_model_insert_default",
            _remap_key_strip_module_then_one_base_model_insert_default,
        ),
        ("add_peft_prefix", _remap_key_add_peft_prefix),
        ("add_peft_prefix_insert_default", _remap_key_add_peft_prefix_insert_default),
    ]

    best_name = None
    best_matched = None
    best_count = -1

    for name, remap_fn in remappers:
        matched = {}

        for old_key, tensor in raw_state_dict.items():
            new_key = remap_fn(old_key)

            if new_key not in model_state:
                continue

            if tuple(tensor.shape) != tuple(model_state[new_key].shape):
                continue

            matched[new_key] = tensor

        if len(matched) > best_count:
            best_count = len(matched)
            best_name = name
            best_matched = matched

    if best_matched is None or len(best_matched) == 0:
        sample_ckpt_keys = list(raw_state_dict.keys())[:30]
        sample_model_keys = list(model_state.keys())[:30]

        raise RuntimeError(
            "No checkpoint weights matched current model.\n"
            f"Sample checkpoint keys: {sample_ckpt_keys}\n"
            f"Sample model keys: {sample_model_keys}\n"
            "Please check --base_model, tokenizer, and LoRA hyperparameters."
        )

    incompatible = model.load_state_dict(best_matched, strict=False)

    _rank0_print(
        f"[ckpt loader] loaded {len(best_matched)}/{len(raw_state_dict)} tensors "
        f"with key remap: {best_name}"
    )
    _rank0_print(
        f"[ckpt loader] missing keys: {len(incompatible.missing_keys)}, "
        f"unexpected keys: {len(incompatible.unexpected_keys)}"
    )

    if int(os.environ.get("LOCAL_RANK") or 0) == 0:
        if len(incompatible.missing_keys) > 0:
            print("[ckpt loader] first missing keys:", incompatible.missing_keys[:10])

        if len(incompatible.unexpected_keys) > 0:
            print("[ckpt loader] first unexpected keys:", incompatible.unexpected_keys[:10])


def _get_compute_dtype(args):
    if getattr(args, "use_bf16", False) or getattr(args, "bf16", False):
        return torch.bfloat16

    if getattr(args, "use_fp16", False) or getattr(args, "fp16", False):
        return torch.float16

    return torch.bfloat16


def build_model_and_tokenizer(args, device_map):
    ckpt_path = args.ckpt_path

    tokenizer_path = ckpt_path if _has_tokenizer_files(ckpt_path) else args.base_model

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
        trust_remote_code=True,
        use_fast=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    tokenizer.padding_side = "left"

    dtype = _get_compute_dtype(args)

    model_kwargs = dict(
        device_map=device_map,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )

    if getattr(args, "attn_implementation", None):
        model_kwargs["attn_implementation"] = args.attn_implementation

    if getattr(args, "load_in_8bit", False):
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_threshold=6.0,
            llm_int8_has_fp16_weight=False,
        )

    adapter_config_path = os.path.join(ckpt_path, "adapter_config.json")
    has_adapter_config = os.path.exists(adapter_config_path)

    # Case 1:
    # 标准 PEFT adapter checkpoint:
    #   adapter_config.json
    #   adapter_model.safetensors / adapter_model.bin
    if getattr(args, "lora", False) and has_adapter_config:
        _rank0_print("[ckpt loader] loading standard PEFT adapter checkpoint")

        base = AutoModelForCausalLM.from_pretrained(
            args.base_model,
            **model_kwargs,
        )
        base.resize_token_embeddings(len(tokenizer))

        model = PeftModel.from_pretrained(
            base,
            ckpt_path,
            device_map=device_map,
            torch_dtype=dtype,
        )

    else:
        weight_files = _find_weight_files(ckpt_path)

        # Case 2:
        # Trainer / wrapper checkpoint:
        #   model.safetensors
        #   trainer_state.json
        #   optimizer.pt / scheduler.pt
        #   但没有 adapter_config.json
        if len(weight_files) > 0:
            _rank0_print("[ckpt loader] loading raw checkpoint weights:")
            for wf in weight_files:
                _rank0_print("  -", wf)

            raw_state_dict = _load_state_dict_from_files(weight_files)
            has_lora_weights = _state_dict_has_lora(raw_state_dict)

            base = AutoModelForCausalLM.from_pretrained(
                args.base_model,
                **model_kwargs,
            )
            base.resize_token_embeddings(len(tokenizer))

            if getattr(args, "lora", False) or has_lora_weights:
                if has_lora_weights:
                    _rank0_print("[ckpt loader] LoRA weights detected in raw checkpoint")
                else:
                    _rank0_print("[ckpt loader] --lora enabled, but no LoRA keys detected")

                lora_config = _build_lora_config_for_test(
                    args=args,
                    raw_state_dict=raw_state_dict,
                )
                model = get_peft_model(base, lora_config)

            else:
                _rank0_print("[ckpt loader] no LoRA weights detected; loading as full model state_dict")
                model = base

            _load_best_matched_state_dict(model, raw_state_dict)

            del raw_state_dict
            torch.cuda.empty_cache()

        # Case 3:
        # 普通完整 HF model directory。
        else:
            _rank0_print("[ckpt loader] no raw weight file found; fallback to AutoModelForCausalLM.from_pretrained")

            model_source = ckpt_path

            if not os.path.exists(os.path.join(ckpt_path, "config.json")):
                model_source = args.base_model

            model = AutoModelForCausalLM.from_pretrained(
                model_source,
                **model_kwargs,
            )
            model.resize_token_embeddings(len(tokenizer))

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


def _to_list(x):
    if x is None:
        return []

    if isinstance(x, (list, tuple, set)):
        return list(x)

    return [x]


def unwrap_modules_to_save_embedding(module):
    """
    PEFT 的 modules_to_save 可能会包住 embed_tokens。
    这里取出真正带 .weight 的 embedding module。
    """
    if module is None:
        raise AttributeError("Input embedding module is None.")

    if hasattr(module, "weight"):
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
        "Cannot find real embedding weight from input embeddings. "
        f"Got module type: {type(module).__name__}."
    )


def get_model_input_embeddings(model):
    """
    兼容普通 HF model 和 PEFT/Lora model。
    返回真正的 embedding module。
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


def model_next_logits_no_cache(model, inputs_embeds, attention_mask):
    """
    no-cache forward。
    输入完整 hidden prefix，返回最后一个位置预测下一个 token 的 logits。
    """
    outputs = model(
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        use_cache=False,
        return_dict=True,
        output_attentions=False,
        output_hidden_states=False,
    )

    return outputs.logits[:, -1, :]


def predict_soft_coarse_embedding(
    next_logits,
    embedding_weight,
    coarse_token_ids,
    temperature,
    use_coarse_score,
):
    """
    推理阶段：
      coarse logits -> soft coarse embedding

    coarse 不解码成文本，不作为显式 token 输出。
    只在当前 coarse codebook 上做 softmax，然后对 embedding 加权求和。

    next_logits: [B, vocab_size]
    coarse_token_ids: 当前 coarse 位置允许的 coarse token id 列表

    return:
      soft_coarse_emb: [B, hidden_size]
      coarse_score: [B]
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
):
    """
    简单 no-cache 隐式交错推理。

    每个 fine code 位置执行：
      context
      -> coarse logits
      -> soft coarse embedding
      -> fine logits
      -> select fine token
      -> append soft coarse emb + fine emb to context

    输出只 decode fine tokens。
    coarse tokens 永远不作为文本输出。
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

        # 1. 当前 context 预测隐式 coarse。
        coarse_logits = model_next_logits_no_cache(
            model=model,
            inputs_embeds=full_inputs_embeds,
            attention_mask=full_attention_mask,
        )

        # 2. coarse logits 转为 soft coarse embedding。
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

        # 3. context + soft coarse embedding 预测显式 fine token。
        inputs_after_coarse = torch.cat(
            [full_inputs_embeds, coarse_embeds.unsqueeze(1)],
            dim=1,
        )

        ones = torch.ones(
            (inputs_after_coarse.size(0), 1),
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

                # 下一步 context 追加：
                # soft coarse embedding + selected fine embedding
                new_extra_embeds = torch.cat(
                    [
                        beam["extra_embeds"],
                        coarse_embeds[flat_idx].unsqueeze(0),
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
            print("[latent interleaved no-cache] enabled")
            print("[latent interleaved no-cache] fine_code_len:", fine_code_len)
            print("[latent interleaved no-cache] coarse positions:", len(coarse_position_token_ids))
            print("[latent interleaved no-cache] fine trie size:", len(fine_trie))

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

    if getattr(args, "use_bf16", False) or getattr(args, "bf16", False):
        amp_dtype = torch.bfloat16
    elif getattr(args, "use_fp16", False) or getattr(args, "fp16", False):
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
                        1 for x in decoded
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

    return parser


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Simple latent interleaved HoloRec-Qwen DDP test"
    )

    parser = parse_global_args(parser)

    # 复用 utils.py 中训练时的 LoRA 参数，保持和 lora_finetune.py 一致：
    # --lora_r
    # --lora_alpha
    # --lora_dropout
    # --lora_target_modules
    # --lora_modules_to_save
    parser = parse_train_args(parser)

    parser = parse_dataset_args(parser)
    parser = parse_test_args(parser)
    parser = add_interleaved_test_args(parser)

    args = parser.parse_args()

    test_ddp(args)
