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
import numpy as np
from transformers import T5Tokenizer


class BaseDataset(Dataset):

    def __init__(self, args):
        super().__init__()

        self.args = args
        self.dataset = args.dataset
        self.data_path = os.path.join(args.data_path, self.dataset)

        self.max_his_len = args.max_his_len
        self.his_sep = args.his_sep
        self.index_file = args.index_file
        self.coarse_index_file = args.coarse_index_file
        self.add_prefix = args.add_prefix

        self.new_tokens = None
        self.allowed_tokens = None
        self.all_items = None

    def _load_data(self):
        raise NotImplementedError

    def get_new_tokens(self):
        """
        同时收集细粒度 token 和粗粒度 token
        """
        if self.new_tokens is not None:
            return self.new_tokens

        self.new_tokens = set()

        if hasattr(self, "indices"):
            for index in self.indices.values():
                for token in index:
                    self.new_tokens.add(token)

        if hasattr(self, "coarse_indices"):
            for index in self.coarse_indices.values():
                for token in index:
                    self.new_tokens.add(token)

        self.new_tokens = sorted(list(self.new_tokens))
        return self.new_tokens

    def get_all_items(self):
        if self.all_items is not None:
            return self.all_items

        self.all_items = set()
        for index in self.indices.values():
            self.all_items.add("".join(index))

        return self.all_items

    def get_all_items_v2(self):
        if self.all_items is not None:
            return self.all_items

        self.all_items = []
        for index in self.indices.values():
            self.all_items.append("".join(index))

        return self.all_items

    def get_prefix_allowed_tokens_fn(self, tokenizer):
        """
        如果你测试阶段要做 constrained decoding，
        这里改成支持交错序列：
        [coarse_1, fine_1, coarse_2, fine_2, ..., eos]
        """
        if self.allowed_tokens is None:
            self.allowed_tokens = {}

            for item_id in self.indices.keys():
                fine_index = self.indices[item_id]
                coarse_index = self.coarse_indices[item_id]

                assert len(fine_index) == len(coarse_index), \
                    f"item {item_id} coarse/fine code length mismatch."

                interleaved = []
                for c, f in zip(coarse_index, fine_index):
                    interleaved.append(c)
                    interleaved.append(f)

                for i, token in enumerate(interleaved):
                    token_id = tokenizer.convert_tokens_to_ids(token)
                    if i not in self.allowed_tokens:
                        self.allowed_tokens[i] = set()
                    self.allowed_tokens[i].add(token_id)

            self.allowed_tokens[len(self.allowed_tokens)] = set([tokenizer.eos_token_id])

        sep = [0]

        def prefix_allowed_tokens_fn(batch_id, sentence):
            sentence = sentence.tolist()
            reversed_sent = sentence[::-1]
            for i in range(len(reversed_sent)):
                if reversed_sent[i:i + len(sep)] == sep[::-1]:
                    pos = i
                    if pos in self.allowed_tokens:
                        return list(self.allowed_tokens[pos])
                    else:
                        return [tokenizer.eos_token_id]
            return [tokenizer.eos_token_id]

        return prefix_allowed_tokens_fn

    def _process_data(self):
        raise NotImplementedError


class SeqRecDataset(BaseDataset):

    def __init__(self, args, mode="train",
                 prompt_sample_num=1, prompt_id=0, sample_num=-1):
        super().__init__(args)

        self.mode = mode
        self.prompt_id = prompt_id
        self.sample_num = sample_num

        self._load_data()
        self._remap_items()

        if self.mode == 'train':
            self.inter_data = self._process_train_data()
        elif self.mode == 'valid':
            self.inter_data = self._process_valid_data()
        elif self.mode == 'test':
            self.inter_data = self._process_test_data()
        elif self.mode == 'test_ranking':
            self.inter_data = self._process_test_data_ids()
        else:
            raise NotImplementedError

    def _load_data(self):
        with open(os.path.join(self.data_path, self.dataset + ".inter.json"), 'r') as f:
            self.inters = json.load(f)

        with open(os.path.join(self.data_path, self.dataset + self.index_file), 'r') as f:
            self.indices = json.load(f)

        with open(os.path.join(self.data_path, self.dataset + self.coarse_index_file), 'r') as f:
            self.coarse_indices = json.load(f)

    def _remap_items(self):
        """
        fine_remapped_inters: 每个用户历史中每个 item -> 拼接后的细粒度字符串
        coarse_remapped_inters: 每个用户历史中每个 item -> 拼接后的粗粒度字符串
        fine_code_seqs / coarse_code_seqs: 每个 item 的 token list，供 label 使用
        """
        self.fine_remapped_inters = dict()
        self.coarse_remapped_inters = dict()
        self.fine_code_seqs = dict()
        self.coarse_code_seqs = dict()

        for uid, items in self.inters.items():
            fine_items = []
            coarse_items = []
            for i in items:
                i = str(i)

                fine_tokens = self.indices[i]
                coarse_tokens = self.coarse_indices[i]

                assert len(fine_tokens) == len(coarse_tokens), \
                    f"item {i} coarse/fine code length mismatch."

                self.fine_code_seqs[i] = fine_tokens
                self.coarse_code_seqs[i] = coarse_tokens

                fine_items.append("".join(fine_tokens))
                coarse_items.append("".join(coarse_tokens))

            self.fine_remapped_inters[uid] = fine_items
            self.coarse_remapped_inters[uid] = coarse_items

    def _build_history(self, items):
        if self.max_his_len > 0:
            items = items[-self.max_his_len:]
        if self.add_prefix:
            items = [str(k + 1) + ". " + item_idx for k, item_idx in enumerate(items)]
        return "".join(items)

    def _build_one_data(self, uid, target_item_id, history_items):
        fine_tokens = self.indices[str(target_item_id)]
        coarse_tokens = self.coarse_indices[str(target_item_id)]

        one_data = dict()
        one_data["fine_codes"] = fine_tokens
        one_data["coarse_codes"] = coarse_tokens
        one_data["inters"] = self._build_history(history_items)
        return one_data

    def _process_train_data(self):
        inter_data = []
        for uid in self.inters:
            raw_items = self.inters[uid][:-2]  # 保持和你原来的 train 切法一致
            fine_items = self.fine_remapped_inters[uid][:-2]

            for i in range(1, len(raw_items)):
                target_item_id = raw_items[i]
                history_items = fine_items[:i]
                one_data = self._build_one_data(uid, target_item_id, history_items)
                inter_data.append(one_data)

        return inter_data

    def _process_valid_data(self):
        inter_data = []
        for uid in self.inters:
            raw_items = self.inters[uid]
            fine_items = self.fine_remapped_inters[uid]

            target_item_id = raw_items[-2]
            history_items = fine_items[:-2]

            one_data = self._build_one_data(uid, target_item_id, history_items)
            inter_data.append(one_data)

        return inter_data

    def _process_test_data(self):
        inter_data = []
        for uid in self.inters:
            raw_items = self.inters[uid]
            fine_items = self.fine_remapped_inters[uid]

            target_item_id = raw_items[-1]
            history_items = fine_items[:-1]

            one_data = self._build_one_data(uid, target_item_id, history_items)
            inter_data.append(one_data)

        if self.sample_num > 0:
            all_inter_idx = range(len(inter_data))
            sample_idx = np.random.choice(all_inter_idx, self.sample_num, replace=False)
            inter_data = np.array(inter_data)[sample_idx].tolist()

        return inter_data

    def _process_test_data_ids(self):
        """
        ranking 模式下保留 item id，方便后处理
        """
        inter_data = []
        for uid in self.inters:
            raw_items = self.inters[uid]

            target_item_id = raw_items[-1]
            history_item_ids = raw_items[:-1]
            if self.max_his_len > 0:
                history_item_ids = history_item_ids[-self.max_his_len:]

            one_data = dict()
            one_data["fine_codes"] = self.indices[str(target_item_id)]
            one_data["coarse_codes"] = self.coarse_indices[str(target_item_id)]
            one_data["inters"] = history_item_ids
            inter_data.append(one_data)

        if self.sample_num > 0:
            all_inter_idx = range(len(inter_data))
            sample_idx = np.random.choice(all_inter_idx, self.sample_num, replace=False)
            inter_data = np.array(inter_data)[sample_idx].tolist()

        return inter_data

    def set_prompt(self, prompt_id):
        self.prompt_id = prompt_id

    def __len__(self):
        return len(self.inter_data)

    def __getitem__(self, index):
        d = self.inter_data[index]
        return dict(
            input_ids=d["inters"],
            fine_codes=d["fine_codes"],
            coarse_codes=d["coarse_codes"]
        )
