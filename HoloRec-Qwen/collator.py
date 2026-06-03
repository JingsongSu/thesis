# import torch
# import copy
# import argparse
# from dataclasses import dataclass

# import transformers
# import math
# from torch.utils.data import Sampler
# import torch.distributed as dist
# from transformers import LlamaForCausalLM, LlamaTokenizer, LlamaConfig, T5Tokenizer, T5Config, T5ForConditionalGeneration


# class Collator(object):

#     def __init__(self, args, tokenizer):
#         self.args = args
#         self.only_train_response = args.only_train_response
#         self.tokenizer = tokenizer
#         if self.tokenizer.pad_token_id is None:
#             self.tokenizer.pad_token_id = self.tokenizer.unk_token_id
#         # print(self.tokenizer.model_max_length)

#     def __call__(self, batch):

#         input_texts = [d["input_ids"] for d in batch]
#         full_texts = [d["labels"] + self.tokenizer.eos_token for d in batch]

#         inputs = self.tokenizer(
#             text = full_texts,
#             text_target = input_texts,
#             return_tensors="pt",
#             padding="longest",
#             max_length=self.tokenizer.model_max_length,
#             truncation=True,
#             return_attention_mask=True,
#         )
#         labels = copy.deepcopy(inputs["input_ids"])
#         if self.only_train_response:
#             # ignore padding
#             labels[labels == self.tokenizer.pad_token_id] = -100
#             # ignore input text
#             labels[torch.where(inputs["labels"] != self.tokenizer.pad_token_id)] = -100

#         inputs["labels"] = labels


#         return inputs



# class TestCollator(object):

#     def __init__(self, args, tokenizer):
#         self.args = args
#         self.tokenizer = tokenizer
#         if self.tokenizer.pad_token_id is None:
#             self.tokenizer.pad_token_id = 0

#         if isinstance(self.tokenizer, LlamaTokenizer):
#             # Allow batched inference
#             self.tokenizer.padding_side = "left"

#     def __call__(self, batch):

#         input_texts = [d["input_ids"] for d in batch]
#         targets = [d["labels"] for d in batch]
#         inputs = self.tokenizer(
#             text=input_texts,
#             return_tensors="pt",
#             padding="longest",
#             max_length=self.tokenizer.model_max_length,
#             truncation=True,
#             return_attention_mask=True,
#         )

#         return (inputs, targets)



# import copy
# import torch


# class Collator(object):
#     def __init__(self, args, tokenizer):
#         self.args = args
#         self.only_train_response = args.only_train_response
#         self.tokenizer = tokenizer

#         # 训练时必须有 pad_token_id
#         if self.tokenizer.pad_token_id is None:
#             # Qwen 通常没有 pad_token，安全做法是用 eos 当 pad
#             if getattr(self.tokenizer, "eos_token_id", None) is not None:
#                 self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
#             else:
#                 # 兜底
#                 self.tokenizer.pad_token_id = 0

#     def __call__(self, batch):
#         input_texts = [d["input_ids"] for d in batch]
#         full_texts = [d["labels"] + self.tokenizer.eos_token for d in batch]

#         # 你这里用了 text / text_target 这种“seq2seq式”的编码方式
#         # 如果你的 tokenizer 支持（大多数 HF tokenizer 支持），就保持不变
#         inputs = self.tokenizer(
#             text=full_texts,
#             text_target=input_texts,
#             return_tensors="pt",
#             padding="longest",
#             max_length=self.tokenizer.model_max_length,
#             truncation=True,
#             return_attention_mask=True,
#         )

#         labels = copy.deepcopy(inputs["input_ids"])
#         if self.only_train_response:
#             # ignore padding
#             labels[labels == self.tokenizer.pad_token_id] = -100
#             # ignore input text（inputs["labels"] 是 text_target 编码结果）
#             labels[torch.where(inputs["labels"] != self.tokenizer.pad_token_id)] = -100

#         inputs["labels"] = labels
#         return inputs


# class TestCollator(object):
#     def __init__(self, args, tokenizer):
#         self.args = args
#         self.tokenizer = tokenizer

#         if self.tokenizer.pad_token_id is None:
#             if getattr(self.tokenizer, "eos_token_id", None) is not None:
#                 self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
#             else:
#                 self.tokenizer.pad_token_id = 0

#         # 推理 batching 通常需要 left padding（对因果 LM 更友好）
#         self.tokenizer.padding_side = "left"

#     def __call__(self, batch):
#         input_texts = [d["input_ids"] for d in batch]
#         targets = [d["labels"] for d in batch]

#         inputs = self.tokenizer(
#             text=input_texts,
#             return_tensors="pt",
#             padding="longest",
#             max_length=self.tokenizer.model_max_length,
#             truncation=True,
#             return_attention_mask=True,
#         )
#         return (inputs, targets)





import copy
import torch


class Collator(object):
    def __init__(self, args, tokenizer):
        self.args = args
        self.only_train_response = args.only_train_response
        self.tokenizer = tokenizer

        if self.tokenizer.pad_token_id is None:
            if getattr(self.tokenizer, "eos_token_id", None) is not None:
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            else:
                self.tokenizer.pad_token_id = 0

    def __call__(self, batch):
        input_texts = [d["input_ids"] for d in batch]
        full_texts = [d["labels"] + self.tokenizer.eos_token for d in batch]

        inputs = self.tokenizer(
            text=full_texts,
            text_target=input_texts,
            return_tensors="pt",
            padding="longest",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_attention_mask=True,
        )

        labels = copy.deepcopy(inputs["input_ids"])
        if self.only_train_response:
            labels[labels == self.tokenizer.pad_token_id] = -100
            labels[torch.where(inputs["labels"] != self.tokenizer.pad_token_id)] = -100

        inputs["labels"] = labels
        return inputs


class TestCollator(object):
    def __init__(self, args, tokenizer):
        self.args = args
        self.tokenizer = tokenizer

        if self.tokenizer.pad_token_id is None:
            if getattr(self.tokenizer, "eos_token_id", None) is not None:
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            else:
                self.tokenizer.pad_token_id = 0

        self.tokenizer.padding_side = "left"

    def __call__(self, batch):
        input_texts = [d["input_ids"] for d in batch]
        fine_targets = [d["labels"] for d in batch]
        coarse_targets = [d.get("coarse_labels", None) for d in batch]

        inputs = self.tokenizer(
            text=input_texts,
            return_tensors="pt",
            padding="longest",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_attention_mask=True,
        )
        return inputs, fine_targets, coarse_targets

