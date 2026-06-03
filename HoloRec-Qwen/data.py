import copy
import random
import os
import json

import numpy as np
import torch
from torch.utils.data import Dataset

from prompt import sft_prompt, all_prompt


def code_tokens_to_text(tokens):
    """
    ["<a_1>", "<b_2>", "<c_3>", "<d_4>"] -> "<a_1><b_2><c_3><d_4>"
    """
    return "".join(tokens)


def build_interleaved_code_text(fine_tokens, coarse_tokens):
    """
    Build visible training target:
        coarse_1 fine_1 coarse_2 fine_2 ...

    Example:
        fine_tokens   = ["<a_1>", "<b_2>", "<c_3>", "<d_4>"]
        coarse_tokens = ["<A_1>", "<B_2>", "<C_3>", "<D_4>"]

    return:
        "<A_1><a_1><B_2><b_2><C_3><c_3><D_4><d_4>"

    Training:
        model learns to predict coarse before each fine code.

    Inference:
        coarse is converted to latent soft embedding and is not decoded.
    """
    if coarse_tokens is None:
        return code_tokens_to_text(fine_tokens)

    if len(fine_tokens) != len(coarse_tokens):
        raise ValueError(
            f"fine/coarse code length mismatch: "
            f"{len(fine_tokens)} vs {len(coarse_tokens)}"
        )

    out = []
    for c_tok, f_tok in zip(coarse_tokens, fine_tokens):
        out.append(c_tok)
        out.append(f_tok)

    return "".join(out)


def safe_strip_title(text):
    if text is None:
        return ""
    return str(text).strip().strip(".!?,;:`")


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
        Must include both fine and coarse tokens.

        Even though final decoding outputs only fine tokens, training target contains:
            coarse_1 fine_1 coarse_2 fine_2 ...
        """
        if self.new_tokens is not None:
            return self.new_tokens

        self.new_tokens = set()

        if hasattr(self, "indices") and self.indices is not None:
            for index in self.indices.values():
                for token in index:
                    self.new_tokens.add(token)

        if hasattr(self, "coarse_indices") and self.coarse_indices is not None:
            for index in self.coarse_indices.values():
                for token in index:
                    self.new_tokens.add(token)

        self.new_tokens = sorted(list(self.new_tokens))
        return self.new_tokens

    def get_all_items(self):
        """
        For implicit interleaved inference, final visible output is fine-only.
        Therefore item filtering should use fine-only item codes.
        """
        if self.all_items is not None:
            return self.all_items

        self.all_items = set()

        for index in self.indices.values():
            self.all_items.add(code_tokens_to_text(index))

        return self.all_items

    def get_prefix_allowed_tokens_fn(self, tokenizer):
        """
        Compatibility function for vanilla generate() path.

        Interleaved inference does not use this function.
        For vanilla fallback, constrain to fine-only code positions.
        """
        def _token_to_id(token):
            token_id = tokenizer.convert_tokens_to_ids(token)

            if token_id is None:
                enc = tokenizer(token, add_special_tokens=False)
                ids = enc.get("input_ids", [])
                if len(ids) > 0 and isinstance(ids[0], list):
                    ids = ids[0]
                if len(ids) != 1:
                    raise ValueError(f"Token {token!r} is not a single tokenizer id: {ids}")
                return int(ids[0])

            unk_id = getattr(tokenizer, "unk_token_id", None)
            if unk_id is not None and token_id == unk_id:
                enc = tokenizer(token, add_special_tokens=False)
                ids = enc.get("input_ids", [])
                if len(ids) > 0 and isinstance(ids[0], list):
                    ids = ids[0]
                if len(ids) != 1:
                    raise ValueError(f"Token {token!r} is mapped to unk or multiple ids: {ids}")
                return int(ids[0])

            return int(token_id)

        if self.allowed_tokens is None:
            self.allowed_tokens = {}

            for index in self.indices.values():
                for pos, token in enumerate(index):
                    self.allowed_tokens.setdefault(pos, set()).add(_token_to_id(token))

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
    """
    Sequential recommendation dataset for implicit interleaved reasoning.

    Train / valid:
        input  = fine-code history
        target = coarse_1 fine_1 coarse_2 fine_2 ...

    Test:
        input  = fine-code history
        target = fine-code next item
    """

    def __init__(self, args, mode="train", prompt_sample_num=1, prompt_id=0, sample_num=-1):
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
        self.remapped_inters = {}
        self.remapped_coarse_inters = {}
        self.remapped_interleaved_inters = {}

        for uid, items in self.inters.items():
            fine_items = []
            coarse_items = []
            interleaved_items = []

            for raw_iid in items:
                iid = str(raw_iid)

                fine_tokens = self.indices[iid]
                fine_text = code_tokens_to_text(fine_tokens)
                fine_items.append(fine_text)

                if self.coarse_indices is not None:
                    coarse_tokens = self.coarse_indices[iid]
                    coarse_text = code_tokens_to_text(coarse_tokens)
                    interleaved_text = build_interleaved_code_text(
                        fine_tokens=fine_tokens,
                        coarse_tokens=coarse_tokens,
                    )
                else:
                    coarse_text = None
                    interleaved_text = fine_text

                coarse_items.append(coarse_text)
                interleaved_items.append(interleaved_text)

            self.remapped_inters[uid] = fine_items
            self.remapped_coarse_inters[uid] = coarse_items
            self.remapped_interleaved_inters[uid] = interleaved_items

    def _format_history(self, history):
        if self.max_his_len > 0:
            history = history[-self.max_his_len:]

        if self.add_prefix:
            history = [
                str(k + 1) + ". " + item_idx
                for k, item_idx in enumerate(history)
            ]

        return self.his_sep.join(history)

    def _process_train_data(self):
        """
        Key modification:
            old:
                fine_history   -> next_fine
                coarse_history -> next_coarse

            new:
                fine_history -> coarse_1 fine_1 coarse_2 fine_2 ...
        """
        inter_data = []

        for uid in self.remapped_inters:
            fine_items = self.remapped_inters[uid][:-2]
            interleaved_items = self.remapped_interleaved_inters[uid][:-2]

            for i in range(1, len(fine_items)):
                one_data = {}

                history = fine_items[:i]
                one_data["inters"] = self._format_history(history)

                one_data["item"] = interleaved_items[i]

                # debug fields
                one_data["fine_item"] = fine_items[i]
                one_data["interleaved_item"] = interleaved_items[i]

                inter_data.append(one_data)

        if self.sample_num > 0:
            all_idx = range(len(inter_data))
            sample_idx = np.random.choice(all_idx, self.sample_num, replace=False)
            inter_data = np.array(inter_data)[sample_idx].tolist()

        return inter_data

    def _process_valid_data(self):
        """
        Eval loss should match the training objective.
        Therefore valid target is also interleaved code.
        """
        inter_data = []

        for uid in self.remapped_inters:
            fine_items = self.remapped_inters[uid]
            interleaved_items = self.remapped_interleaved_inters[uid]

            one_data = {}
            history = fine_items[:-2]

            one_data["inters"] = self._format_history(history)
            one_data["item"] = interleaved_items[-2]

            # debug fields
            one_data["fine_item"] = fine_items[-2]
            one_data["interleaved_item"] = interleaved_items[-2]

            inter_data.append(one_data)

        if self.sample_num > 0:
            all_idx = range(len(inter_data))
            sample_idx = np.random.choice(all_idx, self.sample_num, replace=False)
            inter_data = np.array(inter_data)[sample_idx].tolist()

        return inter_data

    def _process_test_data(self):
        """
        Test target remains fine-only because final visible output is fine-only.
        coarse_item is kept only for debug / optional analysis.
        """
        inter_data = []

        for uid in self.remapped_inters:
            fine_items = self.remapped_inters[uid]
            coarse_items = self.remapped_coarse_inters[uid]

            one_data = {}
            one_data["item"] = fine_items[-1]
            one_data["coarse_item"] = coarse_items[-1] if coarse_items is not None else None

            history = fine_items[:-1]
            one_data["inters"] = self._format_history(history)

            inter_data.append(one_data)

        if self.sample_num > 0:
            all_idx = range(len(inter_data))
            sample_idx = np.random.choice(all_idx, self.sample_num, replace=False)
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

            for d in self.inter_data:
                prompt_ids = np.random.choice(
                    all_prompt_ids,
                    self.prompt_sample_num,
                    replace=False,
                )
                for prompt_id in prompt_ids:
                    prompt = self.prompts[prompt_id]
                    input_text, output_text = self._get_text_data(d, prompt)
                    self.valid_text_data.append(
                        {
                            "input_ids": input_text,
                            "labels": output_text,
                        }
                    )
        else:
            self.prompt_sample_num = 1
            prompt = self.prompts[self.valid_prompt_id]

            for d in self.inter_data:
                input_text, output_text = self._get_text_data(d, prompt)
                self.valid_text_data.append(
                    {
                        "input_ids": input_text,
                        "labels": output_text,
                    }
                )

    def _get_text_data(self, data, prompt):
        instruction = prompt["instruction"].format(**data)
        response = prompt["response"].format(**data)

        input_text = sft_prompt.format(
            instruction=instruction,
            response="",
        )
        output_text = sft_prompt.format(
            instruction=instruction,
            response=response,
        )

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
        else:
            raise NotImplementedError

        prompt = self.prompts[prompt_id]
        input_text, output_text = self._get_text_data(d, prompt)

        out = {
            "input_ids": input_text,
            "labels": output_text,
        }

        if self.mode == "test":
            out["coarse_labels"] = d.get("coarse_item", None)

        return out


class SeqRecTestDataset(SeqRecDataset):
    """
    Compatibility alias.
    utils.py imports SeqRecTestDataset in the original repository.
    """
    pass


class FusionSeqRecDataset(BaseDataset):
    """
    Original fusion sequential recommendation task.
    This class is kept mostly unchanged.
    """

    def __init__(self, args, mode="train", prompt_sample_num=1, prompt_id=0, sample_num=-1):
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

        self.coarse_indices = None
        if self.coarse_index_file is not None:
            coarse_path = os.path.join(self.data_path, self.dataset + self.coarse_index_file)
            if os.path.exists(coarse_path):
                with open(coarse_path, "r") as f:
                    self.coarse_indices = json.load(f)

        with open(os.path.join(self.data_path, self.dataset + ".item.json"), "r") as f:
            self.item_feat = json.load(f)

    def _item_code(self, iid):
        iid = str(iid)
        fine_tokens = self.indices[iid]

        if self.coarse_indices is not None and iid in self.coarse_indices:
            return build_interleaved_code_text(
                fine_tokens=fine_tokens,
                coarse_tokens=self.coarse_indices[iid],
            )

        return code_tokens_to_text(fine_tokens)

    def _fine_item_code(self, iid):
        return code_tokens_to_text(self.indices[str(iid)])

    def _format_history(self, items, use_title=True):
        inters = [self._fine_item_code(j) for j in items]
        inter_titles = [
            '"' + safe_strip_title(self.item_feat[str(j)].get("title", "")) + '"'
            for j in items
        ]

        if self.add_prefix:
            inters = [
                str(k + 1) + ". " + item_idx
                for k, item_idx in enumerate(inters)
            ]
            inter_titles = [
                str(k + 1) + ". " + item_title
                for k, item_title in enumerate(inter_titles)
            ]

        return self.his_sep.join(inters), self.his_sep.join(inter_titles)

    def _process_train_data(self):
        inter_data = []

        for uid in self.inters:
            items = self.inters[uid][:-2]

            for i in range(1, len(items)):
                iid = str(items[i])
                one_data = {}

                one_data["item"] = self._item_code(iid)
                one_data["title"] = safe_strip_title(self.item_feat[iid].get("title", ""))
                one_data["description"] = self.item_feat[iid].get("description", "")

                history = items[:i]
                if self.max_his_len > 0:
                    history = history[-self.max_his_len:]

                one_data["inters"], one_data["inter_titles"] = self._format_history(history)
                inter_data.append(one_data)

        if self.sample_num > 0:
            all_idx = range(len(inter_data))
            sample_idx = np.random.choice(all_idx, self.sample_num, replace=False)
            inter_data = np.array(inter_data)[sample_idx].tolist()

        return inter_data

    def _process_valid_data(self):
        inter_data = []

        for uid in self.inters:
            items = self.inters[uid]
            iid = str(items[-2])

            one_data = {}
            one_data["item"] = self._item_code(iid)
            one_data["title"] = safe_strip_title(self.item_feat[iid].get("title", ""))
            one_data["description"] = self.item_feat[iid].get("description", "")

            history = items[:-2]
            if self.max_his_len > 0:
                history = history[-self.max_his_len:]

            one_data["inters"], one_data["inter_titles"] = self._format_history(history)
            inter_data.append(one_data)

        if self.sample_num > 0:
            all_idx = range(len(inter_data))
            sample_idx = np.random.choice(all_idx, self.sample_num, replace=False)
            inter_data = np.array(inter_data)[sample_idx].tolist()

        return inter_data

    def _process_test_data(self):
        inter_data = []

        for uid in self.inters:
            items = self.inters[uid]
            iid = str(items[-1])

            one_data = {}
            one_data["item"] = self._fine_item_code(iid)
            one_data["title"] = safe_strip_title(self.item_feat[iid].get("title", ""))
            one_data["description"] = self.item_feat[iid].get("description", "")

            history = items[:-1]
            if self.max_his_len > 0:
                history = history[-self.max_his_len:]

            one_data["inters"], one_data["inter_titles"] = self._format_history(history)
            inter_data.append(one_data)

        if self.sample_num > 0:
            all_idx = range(len(inter_data))
            sample_idx = np.random.choice(all_idx, self.sample_num, replace=False)
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

            for d in self.inter_data:
                prompt_ids = np.random.choice(
                    all_prompt_ids,
                    self.prompt_sample_num,
                    replace=False,
                )
                for prompt_id in prompt_ids:
                    prompt = self.prompts[prompt_id]
                    input_text, output_text = self._get_text_data(d, prompt)
                    self.valid_text_data.append(
                        {
                            "input_ids": input_text,
                            "labels": output_text,
                        }
                    )
        else:
            self.prompt_sample_num = 1
            prompt = self.prompts[self.valid_prompt_id]

            for d in self.inter_data:
                input_text, output_text = self._get_text_data(d, prompt)
                self.valid_text_data.append(
                    {
                        "input_ids": input_text,
                        "labels": output_text,
                    }
                )

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
        else:
            raise NotImplementedError

        prompt = self.prompts[prompt_id]
        input_text, output_text = self._get_text_data(d, prompt)

        return {
            "input_ids": input_text,
            "labels": output_text,
        }


class ItemFeatDataset(BaseDataset):
    """
    item2index / index2item.

    For consistency with interleaved training:
        item2index target uses interleaved code if coarse code exists.
        index2item input code also uses interleaved code if coarse code exists.

    If this affects your ablation, you can train with:
        --tasks seqrec
    """

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

    def _item_code(self, iid):
        iid = str(iid)
        fine_tokens = self.indices[iid]

        if self.coarse_indices is not None and iid in self.coarse_indices:
            return build_interleaved_code_text(
                fine_tokens=fine_tokens,
                coarse_tokens=self.coarse_indices[iid],
            )

        return code_tokens_to_text(fine_tokens)

    def _process_data(self):
        feat_data = []

        for iid in self.item_feat:
            raw_feat = self.item_feat[iid]
            one_data = dict(raw_feat)

            one_data["title"] = safe_strip_title(one_data.get("title", ""))
            one_data["description"] = one_data.get("description", "")
            one_data["item"] = self._item_code(iid)

            # debug fields
            one_data["fine_item"] = code_tokens_to_text(self.indices[str(iid)])
            one_data["interleaved_item"] = self._item_code(iid)

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

        input_text, output_text = self._get_text_data(d, prompt)

        return {
            "input_ids": input_text,
            "labels": output_text,
        }


class ItemSearchDataset(BaseDataset):
    """
    Original item search task.

    Test output is still fine-only.
    Train target uses fine-only unless you explicitly adapt the prompt/task.
    """

    def __init__(self, args, mode="train", prompt_sample_num=1, prompt_id=0, sample_num=-1):
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

        self.coarse_indices = None
        if self.coarse_index_file is not None:
            coarse_path = os.path.join(self.data_path, self.dataset + self.coarse_index_file)
            if os.path.exists(coarse_path):
                with open(coarse_path, "r") as f:
                    self.coarse_indices = json.load(f)

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

            iid = str(user_vague_intention[uid]["item"])
            inters = user_vague_intention[uid]["inters"]

            one_data["item"] = code_tokens_to_text(self.indices[iid])

            if self.max_his_len > 0:
                inters = inters[-self.max_his_len:]

            inters = [
                code_tokens_to_text(self.indices[str(i)])
                for i in inters
            ]

            if self.add_prefix:
                inters = [
                    str(k + 1) + ". " + item_idx
                    for k, item_idx in enumerate(inters)
                ]

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
        d = copy.deepcopy(self.search_data[idx])

        if self.mode == "train":
            prompt_id = random.randint(0, len(self.prompts) - 1)
        elif self.mode == "test":
            prompt_id = self.prompt_id
        else:
            raise NotImplementedError

        prompt = self.prompts[prompt_id]

        d["explicit_preference"] = copy.deepcopy(
            random.choice(d["explicit_preferences"])
        )

        all_querys = [
            d["user_related_intention"],
            d["item_related_intention"],
        ]
        d["query"] = random.choice(all_querys)

        input_text, output_text = self._get_text_data(d, prompt)

        return {
            "input_ids": input_text,
            "labels": output_text,
        }


class PreferenceObtainDataset(BaseDataset):
    """
    Original preference obtain task.
    Kept for compatibility with default task loader.
    """

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

        self.coarse_indices = None
        if self.coarse_index_file is not None:
            coarse_path = os.path.join(self.data_path, self.dataset + self.coarse_index_file)
            if os.path.exists(coarse_path):
                with open(coarse_path, "r") as f:
                    self.coarse_indices = json.load(f)

    def _remap_items(self):
        self.remapped_inters = {}

        for uid, items in self.inters.items():
            new_items = [
                code_tokens_to_text(self.indices[str(i)])
                for i in items
            ]
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
                inters = [
                    str(k + 1) + ". " + item_idx
                    for k, item_idx in enumerate(inters)
                ]

            one_data["explicit_preferences"] = user_ep
            one_data["inters"] = self.his_sep.join(inters)

            preference_data.append(one_data)

        if self.sample_num > 0:
            all_idx = range(len(preference_data))
            sample_idx = np.random.choice(all_idx, self.sample_num, replace=False)
            preference_data = np.array(preference_data)[sample_idx].tolist()

        return preference_data

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
        d = copy.deepcopy(self.preference_data[idx])

        prompt_id = random.randint(0, len(self.prompts) - 1)
        prompt = self.prompts[prompt_id]

        d["explicit_preference"] = copy.deepcopy(
            random.choice(d["explicit_preferences"])
        )

        input_text, output_text = self._get_text_data(d, prompt)

        return {
            "input_ids": input_text,
            "labels": output_text,
        }
