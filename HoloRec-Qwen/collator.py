
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

