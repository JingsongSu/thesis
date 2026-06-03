
import copy
import random
import argparse
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from tqdm import tqdm
from collections import defaultdict
import torch.distributed as dist
import logging
import re
import pdb
import json
from prompt import sft_prompt, all_prompt
import numpy as np


class BaseDataset(Dataset):

    def __init__(self, args):
        super().__init__()

        self.args = args
        self.dataset = args.dataset
        self.data_path = os.path.join(args.data_path, self.dataset)

        self.max_his_len = args.max_his_len
        self.his_sep = args.his_sep
        self.index_file = args.index_file
        self.coarse_index_file = getattr(args, "coarse_index_file", None)
        self.add_prefix = args.add_prefix

        self.new_tokens = None
        self.allowed_tokens = None
        self.all_items = None

    def _load_data(self):
        with open(os.path.join(self.data_path, self.dataset + self.index_file), "r") as f:
            self.indices = json.load(f)

        self.coarse_indices = None
        if self.coarse_index_file is not None:
            coarse_path = os.path.join(self.data_path, self.dataset + self.coarse_index_file)
            if os.path.exists(coarse_path):
                with open(coarse_path, "r") as f:
                    self.coarse_indices = json.load(f)

    def get_new_tokens(self):
        """
        同时加入 fine / coarse 两套 token。
        """
        if self.new_tokens is not None:
            return self.new_tokens

        self.new_tokens = set()

        for index in self.indices.values():
            for token in index:
                self.new_tokens.add(token)

        if self.coarse_indices is not None:
            for index in self.coarse_indices.values():
                for token in index:
                    self.new_tokens.add(token)

        self.new_tokens = sorted(list(self.new_tokens))
        return self.new_tokens

    def get_all_items(self):
        """
        测试时允许 fine / coarse 两种合法输出，所以 all_items 也用并集。
        """
        if self.all_items is not None:
            return self.all_items

        self.all_items = set()
        for index in self.indices.values():
            self.all_items.add("".join(index))

        if self.coarse_indices is not None:
            for index in self.coarse_indices.values():
                self.all_items.add("".join(index))

        return self.all_items

    def get_prefix_allowed_tokens_fn(self, tokenizer):
        """
        限制性解码：允许 fine / coarse 两条路径任一合法 token。
        对每个生成位置，allowed token = fine 该位 token集合 ∪ coarse 该位 token集合
        """

        def _encode_ids(text: str):
            enc = tokenizer(text, add_special_tokens=False)
            ids = enc.get("input_ids", [])
            if len(ids) > 0 and isinstance(ids[0], list):
                ids = ids[0]
            return ids

        if self.allowed_tokens is None:
            self.allowed_tokens = {}

            def add_codebook(codebook):
                if codebook is None:
                    return
                for index in codebook.values():
                    for i, token in enumerate(index):
                        ids = _encode_ids(token)
                        if len(ids) == 0:
                            continue
                        token_id = ids[-1]
                        self.allowed_tokens.setdefault(i, set()).add(token_id)

            add_codebook(self.indices)
            add_codebook(self.coarse_indices)

            if len(self.allowed_tokens) == 0:
                raise ValueError("allowed_tokens is empty.")

            max_pos = max(self.allowed_tokens.keys())
            self.allowed_tokens[max_pos + 1] = set([tokenizer.eos_token_id])

        seen_prompt_len = {}

        def prefix_allowed_tokens_fn(batch_id, sentence, prompt_len=None):
            sent_len = len(sentence)

            if prompt_len is None:
                if batch_id not in seen_prompt_len:
                    seen_prompt_len[batch_id] = sent_len
                prompt_len = seen_prompt_len[batch_id]

            gen_pos = sent_len - prompt_len

            if gen_pos in self.allowed_tokens:
                allowed = list(self.allowed_tokens[gen_pos])
                return allowed if len(allowed) > 0 else [tokenizer.eos_token_id]

            return [tokenizer.eos_token_id]

        return prefix_allowed_tokens_fn

    def _process_data(self):
        raise NotImplementedError


class SeqRecDataset(BaseDataset):

    def __init__(self, args, mode="train",
                 prompt_sample_num=1, prompt_id=0, sample_num=-1):
        super().__init__(args)

        self.mode = mode
        self.prompt_sample_num = prompt_sample_num
        self.prompt_id = prompt_id
        self.sample_num = sample_num

        self.prompts = all_prompt["seqrec"]

        self._load_data()
        self._remap_items()

        if self.mode == "train":
            self.inter_data = self._process_train_data()
        elif self.mode == "valid":
            self.sample_valid = args.sample_valid
            self.valid_prompt_id = args.valid_prompt_id
            self.inter_data = self._process_valid_data()
            self._construct_valid_text()
        elif self.mode == "test":
            self.inter_data = self._process_test_data()
        else:
            raise NotImplementedError

    def _load_data(self):
        with open(os.path.join(self.data_path, self.dataset + ".inter.json"), "r") as f:
            self.inters = json.load(f)
        with open(os.path.join(self.data_path, self.dataset + self.index_file), "r") as f:
            self.indices = json.load(f)

        self.coarse_indices = None
        if self.coarse_index_file is not None:
            coarse_path = os.path.join(self.data_path, self.dataset + self.coarse_index_file)
            if os.path.exists(coarse_path):
                with open(coarse_path, "r") as f:
                    self.coarse_indices = json.load(f)

    def _remap_items(self):
        self.remapped_inters = dict()
        self.remapped_coarse_inters = dict()

        for uid, items in self.inters.items():
            fine_items = ["".join(self.indices[str(i)]) for i in items]
            self.remapped_inters[uid] = fine_items

            if self.coarse_indices is not None:
                coarse_items = ["".join(self.coarse_indices[str(i)]) for i in items]
                self.remapped_coarse_inters[uid] = coarse_items
            else:
                self.remapped_coarse_inters[uid] = None

    def _process_train_data(self):
        """
        训练时双路径混训：
        - fine_history -> next_fine
        - coarse_history -> next_coarse
        """
        inter_data = []

        for uid in self.remapped_inters:
            fine_items = self.remapped_inters[uid][:-2]
            coarse_items = self.remapped_coarse_inters[uid][:-2] if self.remapped_coarse_inters[uid] is not None else None

            for i in range(1, len(fine_items)):
                # fine 样本
                one_data = dict()
                one_data["item"] = fine_items[i]
                history = fine_items[:i]
                if self.max_his_len > 0:
                    history = history[-self.max_his_len:]
                if self.add_prefix:
                    history = [str(k + 1) + ". " + item_idx for k, item_idx in enumerate(history)]
                one_data["inters"] = self.his_sep.join(history)
                inter_data.append(one_data)

                # coarse 样本
                if coarse_items is not None:
                    one_data = dict()
                    one_data["item"] = coarse_items[i]
                    history = coarse_items[:i]
                    if self.max_his_len > 0:
                        history = history[-self.max_his_len:]
                    if self.add_prefix:
                        history = [str(k + 1) + ". " + item_idx for k, item_idx in enumerate(history)]
                    one_data["inters"] = self.his_sep.join(history)
                    inter_data.append(one_data)

        return inter_data

#     def _process_train_data(self):
#         """
#         训练时不再双倍扩充数据：
#         - 每个位置只保留 1 条样本
#         - 样本内部同时保存 fine / coarse 两套候选
#         - 真正取样时在 __getitem__ 里随机选择一路

#         这样训练集长度与 fine-only 基本一致，不再变成双倍。
#         """
#         inter_data = []

#         for uid in self.remapped_inters:
#             fine_items = self.remapped_inters[uid][:-2]
#             coarse_items = (
#                 self.remapped_coarse_inters[uid][:-2]
#                 if self.remapped_coarse_inters[uid] is not None
#                 else None
#             )

#             for i in range(1, len(fine_items)):
#                 one_data = dict()

#                 # ===== fine 路径 =====
#                 fine_history = fine_items[:i]
#                 if self.max_his_len > 0:
#                     fine_history = fine_history[-self.max_his_len:]
#                 if self.add_prefix:
#                     fine_history = [
#                         str(k + 1) + ". " + item_idx
#                         for k, item_idx in enumerate(fine_history)
#                     ]

#                 one_data["fine_item"] = fine_items[i]
#                 one_data["fine_inters"] = self.his_sep.join(fine_history)

#                 # ===== coarse 路径（如果存在）=====
#                 if coarse_items is not None:
#                     coarse_history = coarse_items[:i]
#                     if self.max_his_len > 0:
#                         coarse_history = coarse_history[-self.max_his_len:]
#                     if self.add_prefix:
#                         coarse_history = [
#                             str(k + 1) + ". " + item_idx
#                             for k, item_idx in enumerate(coarse_history)
#                         ]

#                     one_data["coarse_item"] = coarse_items[i]
#                     one_data["coarse_inters"] = self.his_sep.join(coarse_history)
#                 else:
#                     one_data["coarse_item"] = None
#                     one_data["coarse_inters"] = None

#                 inter_data.append(one_data)

#         return inter_data


    def _process_valid_data(self):
        """
        valid 保持 fine-only，主要给 Trainer 做验证 loss。
        """
        inter_data = []
        for uid in self.remapped_inters:
            items = self.remapped_inters[uid]
            one_data = dict()
            one_data["item"] = items[-2]
            history = items[:-2]
            if self.max_his_len > 0:
                history = history[-self.max_his_len:]
            if self.add_prefix:
                history = [str(k + 1) + ". " + item_idx for k, item_idx in enumerate(history)]
            one_data["inters"] = self.his_sep.join(history)
            inter_data.append(one_data)

        return inter_data

    def _process_test_data(self):
        """
        test:
        - 输入仍然用 fine history
        - 但 target 同时带 fine / coarse
        """
        inter_data = []
        for uid in self.remapped_inters:
            fine_items = self.remapped_inters[uid]
            coarse_items = self.remapped_coarse_inters[uid] if self.remapped_coarse_inters[uid] is not None else None

            one_data = dict()
            one_data["item"] = fine_items[-1]
            one_data["coarse_item"] = coarse_items[-1] if coarse_items is not None else None

            history = fine_items[:-1]
            if self.max_his_len > 0:
                history = history[-self.max_his_len:]
            if self.add_prefix:
                history = [str(k + 1) + ". " + item_idx for k, item_idx in enumerate(history)]
            one_data["inters"] = self.his_sep.join(history)

            inter_data.append(one_data)

        if self.sample_num > 0:
            all_inter_idx = range(len(inter_data))
            sample_idx = np.random.choice(all_inter_idx, self.sample_num, replace=False)
            inter_data = np.array(inter_data)[sample_idx].tolist()

        return inter_data

    def set_prompt(self, prompt_id):
        self.prompt_id = prompt_id

    def __len__(self):
        if self.mode == "train":
            return len(self.inter_data) * self.prompt_sample_num
        elif self.mode == "valid":
            return len(self.valid_text_data)
        elif self.mode == "test":
            return len(self.inter_data)
        else:
            raise NotImplementedError

    def _construct_valid_text(self):
        self.valid_text_data = []
        if self.sample_valid:
            all_prompt_ids = range(len(self.prompts))
            for i in range(len(self.inter_data)):
                d = self.inter_data[i]
                prompt_ids = np.random.choice(all_prompt_ids, self.prompt_sample_num, replace=False)
                for prompt_id in prompt_ids:
                    prompt = self.prompts[prompt_id]
                    input_text, output_text = self._get_text_data(d, prompt)
                    self.valid_text_data.append({"input_ids": input_text, "labels": output_text})
        else:
            self.prompt_sample_num = 1
            prompt = self.prompts[self.valid_prompt_id]
            for i in range(len(self.inter_data)):
                d = self.inter_data[i]
                input_text, output_text = self._get_text_data(d, prompt)
                self.valid_text_data.append({"input_ids": input_text, "labels": output_text})

    def _get_text_data(self, data, prompt):
        instruction = prompt["instruction"].format(**data)
        response = prompt["response"].format(**data)

        input_text = sft_prompt.format(instruction=instruction, response="")
        output_text = sft_prompt.format(instruction=instruction, response=response)

        if self.mode == "test":
            return input_text, response

        return input_text, output_text

    def __getitem__(self, index):
        if self.mode == "valid":
            return self.valid_text_data[index]

        idx = index // self.prompt_sample_num
        d = self.inter_data[idx]

        if self.mode == "train":
            prompt_id = random.randint(0, len(self.prompts) - 1)
        elif self.mode == "test":
            prompt_id = self.prompt_id

        prompt = self.prompts[prompt_id]
        input_text, output_text = self._get_text_data(d, prompt)

        out = dict(input_ids=input_text, labels=output_text)
        if self.mode == "test":
            out["coarse_labels"] = d.get("coarse_item", None)
        return out
#     def __getitem__(self, index):
#         if self.mode == "valid":
#             return self.valid_text_data[index]

#         idx = index // self.prompt_sample_num
#         d = self.inter_data[idx]

#         if self.mode == "train":
#             prompt_id = random.randint(0, len(self.prompts) - 1)

#             # 训练时随机选择 fine / coarse 一条路径
#             # 如果没有 coarse，就退化为 fine
#             if d.get("coarse_item", None) is not None and random.random() < 0.5:
#                 chosen_data = {
#                     "item": d["coarse_item"],
#                     "inters": d["coarse_inters"],
#                 }
#             else:
#                 chosen_data = {
#                     "item": d["fine_item"],
#                     "inters": d["fine_inters"],
#                 }

#         elif self.mode == "test":
#             prompt_id = self.prompt_id
#             # 测试仍然使用 fine history，保持你原来的评测逻辑
#             chosen_data = {
#                 "item": d["item"],
#                 "inters": d["inters"],
#             }

#         else:
#             raise NotImplementedError

#         prompt = self.prompts[prompt_id]
#         input_text, output_text = self._get_text_data(chosen_data, prompt)

#         out = dict(input_ids=input_text, labels=output_text)
#         if self.mode == "test":
#             out["coarse_labels"] = d.get("coarse_item", None)
#         return out



class FusionSeqRecDataset(BaseDataset):

    def __init__(self, args, mode="train",
                 prompt_sample_num=1, prompt_id=0, sample_num=-1):
        super().__init__(args)

        self.mode = mode
        self.prompt_sample_num = prompt_sample_num
        self.prompt_id = prompt_id
        self.sample_num = sample_num

        self.prompts = all_prompt["fusionseqrec"]

        self._load_data()

        if self.mode == "train":
            self.inter_data = self._process_train_data()
        elif self.mode == "valid":
            self.sample_valid = args.sample_valid
            self.valid_prompt_id = args.valid_prompt_id
            self.inter_data = self._process_valid_data()
            self._construct_valid_text()
        elif self.mode == "test":
            self.inter_data = self._process_test_data()
        else:
            raise NotImplementedError

    def _load_data(self):
        with open(os.path.join(self.data_path, self.dataset + ".inter.json"), "r") as f:
            self.inters = json.load(f)
        with open(os.path.join(self.data_path, self.dataset + self.index_file), "r") as f:
            self.indices = json.load(f)
        with open(os.path.join(self.data_path, self.dataset + ".item.json"), "r") as f:
            self.item_feat = json.load(f)

    def _process_train_data(self):
        inter_data = []
        for uid in self.inters:
            items = self.inters[uid][:-2]
            for i in range(1, len(items)):
                one_data = dict()
                one_data["item"] = "".join(self.indices[str(items[i])])
                one_data["title"] = self.item_feat[str(items[i])]["title"].strip().strip(".!?,;:`")
                one_data["description"] = self.item_feat[str(items[i])]["description"]
                history = items[:i]
                if self.max_his_len > 0:
                    history = history[-self.max_his_len:]
                inters = ["".join(self.indices[str(j)]) for j in history]
                inter_titles = ['"' + self.item_feat[str(j)]["title"].strip().strip(".!?,;:`") + '"' for j in history]

                if self.add_prefix:
                    inters = [str(k + 1) + ". " + item_idx for k, item_idx in enumerate(inters)]
                    inter_titles = [str(k + 1) + ". " + item_title for k, item_title in enumerate(inter_titles)]

                one_data["inters"] = self.his_sep.join(inters)
                one_data["inter_titles"] = self.his_sep.join(inter_titles)
                inter_data.append(one_data)

        if self.sample_num > 0:
            all_inter_idx = range(len(inter_data))
            sample_idx = np.random.choice(all_inter_idx, self.sample_num, replace=False)
            inter_data = np.array(inter_data)[sample_idx].tolist()

        return inter_data

    def _process_valid_data(self):
        inter_data = []
        for uid in self.inters:
            items = self.inters[uid]
            one_data = dict()
            one_data["item"] = "".join(self.indices[str(items[-2])])
            one_data["title"] = self.item_feat[str(items[-2])]["title"].strip().strip(".!?,;:`")
            one_data["description"] = self.item_feat[str(items[-2])]["description"]

            history = items[:-2]
            if self.max_his_len > 0:
                history = history[-self.max_his_len:]
            inters = ["".join(self.indices[str(j)]) for j in history]
            inter_titles = ['"' + self.item_feat[str(j)]["title"].strip().strip(".!?,;:`") + '"' for j in history]

            if self.add_prefix:
                inters = [str(k + 1) + ". " + item_idx for k, item_idx in enumerate(inters)]
                inter_titles = [str(k + 1) + ". " + item_title for k, item_title in enumerate(inter_titles)]

            one_data["inters"] = self.his_sep.join(inters)
            one_data["inter_titles"] = self.his_sep.join(inter_titles)
            inter_data.append(one_data)

        if self.sample_num > 0:
            all_inter_idx = range(len(inter_data))
            sample_idx = np.random.choice(all_inter_idx, self.sample_num, replace=False)
            inter_data = np.array(inter_data)[sample_idx].tolist()

        return inter_data

    def _process_test_data(self):
        inter_data = []
        for uid in self.inters:
            items = self.inters[uid]
            one_data = dict()
            one_data["item"] = "".join(self.indices[str(items[-1])])
            one_data["title"] = self.item_feat[str(items[-1])]["title"].strip().strip(".!?,;:`")
            one_data["description"] = self.item_feat[str(items[-1])]["description"]

            history = items[:-1]
            if self.max_his_len > 0:
                history = history[-self.max_his_len:]
            inters = ["".join(self.indices[str(j)]) for j in history]
            inter_titles = ['"' + self.item_feat[str(j)]["title"].strip().strip(".!?,;:`") + '"' for j in history]

            if self.add_prefix:
                inters = [str(k + 1) + ". " + item_idx for k, item_idx in enumerate(inters)]
                inter_titles = [str(k + 1) + ". " + item_title for k, item_title in enumerate(inter_titles)]

            one_data["inters"] = self.his_sep.join(inters)
            one_data["inter_titles"] = self.his_sep.join(inter_titles)
            inter_data.append(one_data)

        if self.sample_num > 0:
            all_inter_idx = range(len(inter_data))
            sample_idx = np.random.choice(all_inter_idx, self.sample_num, replace=False)
            inter_data = np.array(inter_data)[sample_idx].tolist()

        return inter_data

    def set_prompt(self, prompt_id):
        self.prompt_id = prompt_id

    def __len__(self):
        if self.mode == "train":
            return len(self.inter_data) * self.prompt_sample_num
        elif self.mode == "valid":
            return len(self.valid_text_data)
        elif self.mode == "test":
            return len(self.inter_data)
        else:
            raise NotImplementedError

    def _construct_valid_text(self):
        self.valid_text_data = []
        if self.sample_valid:
            all_prompt_ids = range(len(self.prompts))
            for i in range(len(self.inter_data)):
                d = self.inter_data[i]
                prompt_ids = np.random.choice(all_prompt_ids, self.prompt_sample_num, replace=False)
                for prompt_id in prompt_ids:
                    prompt = self.prompts[prompt_id]
                    input_text, output_text = self._get_text_data(d, prompt)
                    self.valid_text_data.append({"input_ids": input_text, "labels": output_text})
        else:
            self.prompt_sample_num = 1
            prompt = self.prompts[self.valid_prompt_id]
            for i in range(len(self.inter_data)):
                d = self.inter_data[i]
                input_text, output_text = self._get_text_data(d, prompt)
                self.valid_text_data.append({"input_ids": input_text, "labels": output_text})

    def _get_text_data(self, data, prompt):
        instruction = prompt["instruction"].format(**data)
        response = prompt["response"].format(**data)

        input_text = sft_prompt.format(instruction=instruction, response="")
        output_text = sft_prompt.format(instruction=instruction, response=response)

        if self.mode == "test":
            return input_text, response

        return input_text, output_text

    def __getitem__(self, index):
        if self.mode == "valid":
            return self.valid_text_data[index]

        idx = index // self.prompt_sample_num
        d = self.inter_data[idx]

        if self.mode == "train":
            prompt_id = random.randint(0, len(self.prompts) - 1)
        elif self.mode == "test":
            prompt_id = self.prompt_id

        prompt = self.prompts[prompt_id]
        input_text, output_text = self._get_text_data(d, prompt)

        return dict(input_ids=input_text, labels=output_text)


# class ItemFeatDataset(BaseDataset):

#     def __init__(self, args, task="item2index", prompt_sample_num=1, sample_num=-1):
#         super().__init__(args)

#         self.task = task.lower()
#         self.prompt_sample_num = prompt_sample_num
#         self.sample_num = sample_num

#         self.prompts = all_prompt[self.task]

#         self._load_data()
#         self.feat_data = self._process_data()

#     def _load_data(self):
#         with open(os.path.join(self.data_path, self.dataset + self.index_file), "r") as f:
#             self.indices = json.load(f)
#         with open(os.path.join(self.data_path, self.dataset + ".item.json"), "r") as f:
#             self.item_feat = json.load(f)

#     def _process_data(self):
#         feat_data = []
#         for iid in self.item_feat:
#             feat = self.item_feat[iid]
#             index = "".join(self.indices[iid])
#             feat["item"] = index
#             feat["title"] = feat["title"].strip().strip(".!?,;:`")
#             feat_data.append(feat)

#         if self.sample_num > 0:
#             all_idx = range(len(feat_data))
#             sample_idx = np.random.choice(all_idx, self.sample_num, replace=False)
#             feat_data = np.array(feat_data)[sample_idx].tolist()

#         return feat_data

#     def __len__(self):
#         return len(self.feat_data) * self.prompt_sample_num

#     def _get_text_data(self, data, prompt):
#         instruction = prompt["instruction"].format(**data)
#         response = prompt["response"].format(**data)

#         input_text = sft_prompt.format(instruction=instruction, response="")
#         output_text = sft_prompt.format(instruction=instruction, response=response)

#         return input_text, output_text

#     def __getitem__(self, index):
#         idx = index // self.prompt_sample_num
#         d = self.feat_data[idx]

#         prompt_id = random.randint(0, len(self.prompts) - 1)
#         prompt = self.prompts[prompt_id]

#         input_text, output_text = self._get_text_data(d, prompt)
#         return dict(input_ids=input_text, labels=output_text)


class ItemFeatDataset(BaseDataset):

    def __init__(self, args, task="item2index", prompt_sample_num=1, sample_num=-1):
        super().__init__(args)

        self.task = task.lower()
        self.prompt_sample_num = prompt_sample_num
        self.sample_num = sample_num

        self.prompts = all_prompt[self.task]

        self._load_data()
        self.feat_data = self._process_data()

    def _load_data(self):
        with open(os.path.join(self.data_path, self.dataset + self.index_file), "r") as f:
            self.indices = json.load(f)

        self.coarse_indices = None
        if self.coarse_index_file is not None:
            coarse_path = os.path.join(self.data_path, self.dataset + self.coarse_index_file)
            if os.path.exists(coarse_path):
                with open(coarse_path, "r") as f:
                    self.coarse_indices = json.load(f)

        with open(os.path.join(self.data_path, self.dataset + ".item.json"), "r") as f:
            self.item_feat = json.load(f)

    def _process_data(self):
        """
        不再把 fine / coarse 各扩一份样本，
        而是每个 item 只保留 1 条数据，同时存 fine / coarse 两套 target。
        训练时在 __getitem__ 里随机选一路。
        """
        feat_data = []
        for iid in self.item_feat:
            raw_feat = self.item_feat[iid]

            one_data = dict(raw_feat)
            one_data["title"] = one_data["title"].strip().strip(".!?,;:`")

            # fine item
            one_data["fine_item"] = "".join(self.indices[iid])

            # coarse item
            if self.coarse_indices is not None and iid in self.coarse_indices:
                one_data["coarse_item"] = "".join(self.coarse_indices[iid])
            else:
                one_data["coarse_item"] = None

            feat_data.append(one_data)

        if self.sample_num > 0:
            all_idx = range(len(feat_data))
            sample_idx = np.random.choice(all_idx, self.sample_num, replace=False)
            feat_data = np.array(feat_data)[sample_idx].tolist()

        return feat_data

    def __len__(self):
        return len(self.feat_data) * self.prompt_sample_num

    def _get_text_data(self, data, prompt):
        instruction = prompt["instruction"].format(**data)
        response = prompt["response"].format(**data)

        input_text = sft_prompt.format(instruction=instruction, response="")
        output_text = sft_prompt.format(instruction=instruction, response=response)

        return input_text, output_text

    def __getitem__(self, index):
        idx = index // self.prompt_sample_num
        d = self.feat_data[idx]

        prompt_id = random.randint(0, len(self.prompts) - 1)
        prompt = self.prompts[prompt_id]

        # 训练时随机选择 fine / coarse 一条路径
        # 如果没有 coarse，就自动退化为 fine
        if d.get("coarse_item", None) is not None and random.random() < 0.5:
            chosen_item = d["coarse_item"]
        else:
            chosen_item = d["fine_item"]

        chosen_data = dict(d)
        chosen_data["item"] = chosen_item

        input_text, output_text = self._get_text_data(chosen_data, prompt)
        return dict(input_ids=input_text, labels=output_text)



class ItemSearchDataset(BaseDataset):

    def __init__(self, args, mode="train",
                 prompt_sample_num=1, prompt_id=0, sample_num=-1):
        super().__init__(args)

        self.mode = mode
        self.prompt_sample_num = prompt_sample_num
        self.prompt_id = prompt_id
        self.sample_num = sample_num

        self.prompts = all_prompt["itemsearch"]

        self._load_data()
        self.search_data = self._process_data()

    def _load_data(self):
        with open(os.path.join(self.data_path, self.dataset + self.index_file), "r") as f:
            self.indices = json.load(f)
        with open(os.path.join(self.data_path, self.dataset + ".user.json"), "r") as f:
            self.user_info = json.load(f)

    def _process_data(self):
        search_data = []
        user_explicit_preference = self.user_info["user_explicit_preference"]
        user_vague_intention = self.user_info["user_vague_intention"]
        if self.mode == "train":
            user_vague_intention = user_vague_intention["train"]
        elif self.mode == "test":
            user_vague_intention = user_vague_intention["test"]
        else:
            raise NotImplementedError

        for uid in user_explicit_preference.keys():
            one_data = {}
            user_ep = user_explicit_preference[uid]
            user_vi = user_vague_intention[uid]["querys"]
            one_data["explicit_preferences"] = user_ep
            one_data["user_related_intention"] = user_vi[0]
            one_data["item_related_intention"] = user_vi[1]

            iid = user_vague_intention[uid]["item"]
            inters = user_vague_intention[uid]["inters"]

            index = "".join(self.indices[str(iid)])
            one_data["item"] = index

            if self.max_his_len > 0:
                inters = inters[-self.max_his_len:]
            inters = ["".join(self.indices[str(i)]) for i in inters]
            if self.add_prefix:
                inters = [str(k + 1) + ". " + item_idx for k, item_idx in enumerate(inters)]

            one_data["inters"] = self.his_sep.join(inters)
            search_data.append(one_data)

        if self.sample_num > 0:
            all_idx = range(len(search_data))
            sample_idx = np.random.choice(all_idx, self.sample_num, replace=False)
            search_data = np.array(search_data)[sample_idx].tolist()

        return search_data

    def set_prompt(self, prompt_id):
        self.prompt_id = prompt_id

    def __len__(self):
        if self.mode == "train":
            return len(self.search_data) * self.prompt_sample_num
        elif self.mode == "test":
            return len(self.search_data)
        else:
            return len(self.search_data)

    def _get_text_data(self, data, prompt):
        instruction = prompt["instruction"].format(**data)
        response = prompt["response"].format(**data)

        input_text = sft_prompt.format(instruction=instruction, response="")
        output_text = sft_prompt.format(instruction=instruction, response=response)

        if self.mode == "test":
            return input_text, response

        return input_text, output_text

    def __getitem__(self, index):
        idx = index // self.prompt_sample_num
        d = self.search_data[idx]
        if self.mode == "train":
            prompt_id = random.randint(0, len(self.prompts) - 1)
        elif self.mode == "test":
            prompt_id = self.prompt_id

        prompt = self.prompts[prompt_id]

        d["explicit_preference"] = copy.deepcopy(random.choice(d["explicit_preferences"]))
        all_querys = [d["user_related_intention"], d["item_related_intention"]]
        d["query"] = random.choice(all_querys)

        input_text, output_text = self._get_text_data(d, prompt)

        return dict(input_ids=input_text, labels=output_text)


class PreferenceObtainDataset(BaseDataset):

    def __init__(self, args, prompt_sample_num=1, sample_num=-1):
        super().__init__(args)

        self.prompt_sample_num = prompt_sample_num
        self.sample_num = sample_num

        self.prompts = all_prompt["preferenceobtain"]

        self._load_data()
        self._remap_items()

        self.preference_data = self._process_data()

    def _load_data(self):
        with open(os.path.join(self.data_path, self.dataset + ".user.json"), "r") as f:
            self.user_info = json.load(f)
        with open(os.path.join(self.data_path, self.dataset + ".inter.json"), "r") as f:
            self.inters = json.load(f)
        with open(os.path.join(self.data_path, self.dataset + self.index_file), "r") as f:
            self.indices = json.load(f)

    def _remap_items(self):
        self.remapped_inters = dict()
        for uid, items in self.inters.items():
            new_items = ["".join(self.indices[str(i)]) for i in items]
            self.remapped_inters[uid] = new_items

    def _process_data(self):
        preference_data = []
        user_explicit_preference = self.user_info["user_explicit_preference"]

        for uid in user_explicit_preference.keys():
            one_data = {}
            inters = self.remapped_inters[uid][:-3]
            user_ep = user_explicit_preference[uid]

            if self.max_his_len > 0:
                inters = inters[-self.max_his_len:]
            if self.add_prefix:
                inters = [str(k + 1) + ". " + item_idx for k, item_idx in enumerate(inters)]

            one_data["explicit_preferences"] = user_ep
            one_data["inters"] = self.his_sep.join(inters)

            preference_data.append(one_data)

        if self.sample_num > 0:
            all_idx = range(len(preference_data))
            sample_idx = np.random.choice(all_idx, self.sample_num, replace=False)
            preference_data = np.array(preference_data)[sample_idx].tolist()

        return preference_data

    def set_prompt(self, prompt_id):
        self.prompt_id = prompt_id

    def __len__(self):
        return len(self.preference_data) * self.prompt_sample_num

    def _get_text_data(self, data, prompt):
        instruction = prompt["instruction"].format(**data)
        response = prompt["response"].format(**data)

        input_text = sft_prompt.format(instruction=instruction, response="")
        output_text = sft_prompt.format(instruction=instruction, response=response)

        return input_text, output_text

    def __getitem__(self, index):
        idx = index // self.prompt_sample_num

        d = self.preference_data[idx]
        prompt_id = random.randint(0, len(self.prompts) - 1)

        prompt = self.prompts[prompt_id]
        d["explicit_preference"] = copy.deepcopy(random.choice(d["explicit_preferences"]))

        input_text, output_text = self._get_text_data(d, prompt)

        return dict(input_ids=input_text, labels=output_text)


class SeqRecTestDataset(BaseDataset):

    def __init__(self, args, prompt_id=0, sample_num=-1):
        super().__init__(args)

        self.prompt_id = prompt_id
        self.sample_num = sample_num

        self.prompt = all_prompt["seqrec"][self.prompt_id]

        self._load_data()
        self._remap_items()

        self.inter_data = self._process_test_data()

    def _load_data(self):
        with open(os.path.join(self.data_path, self.dataset + ".inter.json"), "r") as f:
            self.inters = json.load(f)
        with open(os.path.join(self.data_path, self.dataset + self.index_file), "r") as f:
            self.indices = json.load(f)

        self.coarse_indices = None
        if self.coarse_index_file is not None:
            coarse_path = os.path.join(self.data_path, self.dataset + self.coarse_index_file)
            if os.path.exists(coarse_path):
                with open(coarse_path, "r") as f:
                    self.coarse_indices = json.load(f)

    def _remap_items(self):
        self.remapped_inters = dict()
        self.remapped_coarse_inters = dict()

        for uid, items in self.inters.items():
            fine_items = ["".join(self.indices[str(i)]) for i in items]
            self.remapped_inters[uid] = fine_items

            if self.coarse_indices is not None:
                coarse_items = ["".join(self.coarse_indices[str(i)]) for i in items]
                self.remapped_coarse_inters[uid] = coarse_items
            else:
                self.remapped_coarse_inters[uid] = None

    def _process_test_data(self):
        inter_data = []
        for uid in self.remapped_inters:
            fine_items = self.remapped_inters[uid]
            coarse_items = self.remapped_coarse_inters[uid] if self.remapped_coarse_inters[uid] is not None else None

            one_data = dict()
            one_data["item"] = fine_items[-1]
            one_data["coarse_item"] = coarse_items[-1] if coarse_items is not None else None

            history = fine_items[:-1]
            if self.max_his_len > 0:
                history = history[-self.max_his_len:]
            if self.add_prefix:
                history = [str(k + 1) + ". " + item_idx for k, item_idx in enumerate(history)]
            one_data["inters"] = self.his_sep.join(history)

            inter_data.append(one_data)

        if self.sample_num > 0:
            all_inter_idx = range(len(inter_data))
            sample_idx = np.random.choice(all_inter_idx, self.sample_num, replace=False)
            inter_data = np.array(inter_data)[sample_idx].tolist()

        return inter_data

    def set_prompt(self, prompt_id):
        self.prompt_id = prompt_id
        self.prompt = all_prompt["seqrec"][self.prompt_id]

    def __len__(self):
        return len(self.inter_data)

    def _get_text_data(self, data, prompt):
        instruction = prompt["instruction"].format(**data)
        response = prompt["response"].format(**data)

        input_text = sft_prompt.format(instruction=instruction, response="")
        return input_text, response

    def __getitem__(self, index):
        d = self.inter_data[index]
        input_text, target = self._get_text_data(d, self.prompt)

        return dict(
            input_ids=input_text,
            labels=target,
            coarse_labels=d.get("coarse_item", None),
        )
