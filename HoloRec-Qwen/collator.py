import torch


def _token_to_id(tokenizer, token):
    token_id = tokenizer.convert_tokens_to_ids(token)

    if token_id is None:
        encoded = tokenizer(token, add_special_tokens=False).get("input_ids", [])
        if len(encoded) == 1:
            return int(encoded[0])
        raise ValueError(f"Token {token!r} is not a single token: {encoded}")

    unk_id = getattr(tokenizer, "unk_token_id", None)
    if unk_id is not None and token_id == unk_id:
        encoded = tokenizer(token, add_special_tokens=False).get("input_ids", [])
        if len(encoded) == 1:
            return int(encoded[0])
        raise ValueError(f"Token {token!r} maps to unk or multiple ids: {encoded}")

    return int(token_id)


class Collator(object):
    """
    Training collator for latent interleaved Qwen.

    Required fields from dataset:
        input_ids: prompt/history text
        fine_tokens:   List[str]
        coarse_tokens: List[str]

    Output:
        input_ids
        attention_mask
        fine_labels
        coarse_labels
    """

    def __init__(self, args, tokenizer):
        self.args = args
        self.tokenizer = tokenizer

        if self.tokenizer.pad_token_id is None:
            if getattr(self.tokenizer, "eos_token_id", None) is not None:
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            else:
                self.tokenizer.pad_token_id = 0

    def _tokens_to_ids(self, tokens):
        return [_token_to_id(self.tokenizer, tok) for tok in tokens]

    def _pad_2d(self, seqs, pad_value=-100):
        max_len = max(len(x) for x in seqs)
        out = torch.full(
            (len(seqs), max_len),
            fill_value=pad_value,
            dtype=torch.long,
        )

        for i, seq in enumerate(seqs):
            if len(seq) > 0:
                out[i, :len(seq)] = torch.tensor(seq, dtype=torch.long)

        return out

    def __call__(self, batch):
        input_texts = [d["input_ids"] for d in batch]

        inputs = self.tokenizer(
            input_texts,
            return_tensors="pt",
            padding="longest",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_attention_mask=True,
        )

        fine_ids = []
        coarse_ids = []

        for d in batch:
            if "fine_tokens" not in d or "coarse_tokens" not in d:
                raise ValueError(
                    "Latent interleaved training requires dataset item to contain "
                    "`fine_tokens` and `coarse_tokens`. "
                    "Please train with --tasks seqrec and modified SeqRecDataset."
                )

            f = self._tokens_to_ids(d["fine_tokens"])
            c = self._tokens_to_ids(d["coarse_tokens"])

            if len(f) != len(c):
                raise ValueError(f"fine/coarse length mismatch: {len(f)} vs {len(c)}")

            fine_ids.append(f)
            coarse_ids.append(c)

        inputs["fine_labels"] = self._pad_2d(fine_ids, pad_value=-100)
        inputs["coarse_labels"] = self._pad_2d(coarse_ids, pad_value=-100)

        return inputs


class TestCollator(object):
    """
    Test collator remains prompt-only.
    Target labels are returned separately for evaluate.py.
    """

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
        targets = [d["labels"] for d in batch]
        coarse_targets = [d.get("coarse_labels", None) for d in batch]

        inputs = self.tokenizer(
            input_texts,
            return_tensors="pt",
            padding="longest",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_attention_mask=True,
        )

        return inputs, targets, coarse_targets
