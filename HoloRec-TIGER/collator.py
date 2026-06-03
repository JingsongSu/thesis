import torch


class Collator(object):
    def __init__(self, args, tokenizer):
        self.args = args
        self.only_train_response = getattr(args, "only_train_response", False)
        self.tokenizer = tokenizer

        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = 0

    def _build_step_labels(self, coarse_codes, fine_codes):
        """
        coarse_labels = [c1, c2, ..., cT]
        fine_labels   = [f1, f2, ..., fT]
        """
        assert len(coarse_codes) == len(fine_codes), "coarse/fine code length mismatch"

        coarse_ids = []
        fine_ids = []

        for c_tok, f_tok in zip(coarse_codes, fine_codes):
            c_id = self.tokenizer.convert_tokens_to_ids(c_tok)
            f_id = self.tokenizer.convert_tokens_to_ids(f_tok)

            if c_id == self.tokenizer.unk_token_id:
                raise ValueError(f"Unknown coarse token: {c_tok}")
            if f_id == self.tokenizer.unk_token_id:
                raise ValueError(f"Unknown fine token: {f_tok}")

            coarse_ids.append(c_id)
            fine_ids.append(f_id)

        return coarse_ids, fine_ids

    def __call__(self, batch):
        input_texts = [d["input_ids"] for d in batch]
        coarse_codes_batch = [d["coarse_codes"] for d in batch]
        fine_codes_batch = [d["fine_codes"] for d in batch]

        inputs = self.tokenizer(
            input_texts,
            return_tensors="pt",
            padding="longest",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_attention_mask=True
        )

        all_coarse_labels = []
        all_fine_labels = []

        for coarse_codes, fine_codes in zip(coarse_codes_batch, fine_codes_batch):
            coarse_labels, fine_labels = self._build_step_labels(coarse_codes, fine_codes)
            all_coarse_labels.append(coarse_labels)
            all_fine_labels.append(fine_labels)

        max_tgt_len = max(len(x) for x in all_fine_labels)

        padded_coarse_labels = []
        padded_fine_labels = []

        for coarse_labels, fine_labels in zip(all_coarse_labels, all_fine_labels):
            pad_len = max_tgt_len - len(fine_labels)

            padded_coarse_labels.append(coarse_labels + [-100] * pad_len)
            padded_fine_labels.append(fine_labels + [-100] * pad_len)

        coarse_labels = torch.tensor(padded_coarse_labels, dtype=torch.long)
        fine_labels = torch.tensor(padded_fine_labels, dtype=torch.long)

        # Trainer 默认会把 labels 传给 model
        inputs["labels"] = fine_labels
        inputs["coarse_labels"] = coarse_labels
        inputs["fine_labels"] = fine_labels

        return inputs


class TestCollator(object):
    def __init__(self, args, tokenizer):
        self.args = args
        self.tokenizer = tokenizer
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = 0

    def __call__(self, batch):
        input_texts = [d["input_ids"] for d in batch]
        targets = []
        for d in batch:
            targets.append({
                "coarse_codes": d["coarse_codes"],
                "fine_codes": d["fine_codes"]
            })

        inputs = self.tokenizer(
            text=input_texts,
            return_tensors="pt",
            padding="longest",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_attention_mask=True,
        )

        return (inputs, targets)
