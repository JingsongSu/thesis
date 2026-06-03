import torch


IGNORE_INDEX = -100


def _flatten_tokenizer_ids(encoded):
    ids = encoded.get("input_ids", encoded)
    if isinstance(ids, torch.Tensor):
        ids = ids.tolist()
    if len(ids) > 0 and isinstance(ids[0], list):
        ids = ids[0]
    return [int(x) for x in ids]


def _token_to_id(tokenizer, token):
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


class Collator(object):
    """
    Collator for implicit interleaved latent HoloRec-Qwen training.

    Important:
    - coarse tokens are NOT placed into visible input_ids.
    - fine tokens are NOT concatenated into input_ids either.
    - input_ids contains only the prompt.
    - fine_labels / coarse_labels are passed separately to InterleavedLatentQwen.

    The model wrapper performs the hidden training loop:

        prompt -> coarse_logits_1
        coarse_logits_1 -> soft coarse embedding_1
        prompt + soft_coarse_1 -> fine_logits_1
        prompt + soft_coarse_1 + gold_fine_1 -> coarse_logits_2
        ...

    Therefore training and test_ddp.py latent inference stay consistent:
    coarse is latent, fine is the only decoded recommendation code.
    """

    def __init__(self, args, tokenizer):
        self.args = args
        self.tokenizer = tokenizer
        self.max_length = int(getattr(args, "model_max_length", tokenizer.model_max_length))
        self._token_id_cache = {}

        if self.tokenizer.pad_token_id is None:
            if getattr(self.tokenizer, "eos_token_id", None) is not None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            else:
                self.tokenizer.pad_token_id = 0

        # Left padding is safer for cached latent decoding/training:
        # the last column is always the last real prompt token.
        self.tokenizer.padding_side = "left"

    def _token_id(self, token):
        if token not in self._token_id_cache:
            self._token_id_cache[token] = _token_to_id(self.tokenizer, token)
        return self._token_id_cache[token]

    def __call__(self, batch):
        input_texts = []
        fine_label_rows = []
        coarse_label_rows = []

        max_code_len = 0

        for example in batch:
            if "fine_tokens" not in example or "coarse_tokens" not in example:
                raise ValueError(
                    "Implicit interleaved HoloRec-Qwen training requires "
                    "`fine_tokens` and `coarse_tokens` in each example. "
                    "Please train with --tasks seqrec and pass --coarse_index_file."
                )

            fine_tokens = example["fine_tokens"]
            coarse_tokens = example["coarse_tokens"]

            if len(fine_tokens) != len(coarse_tokens):
                raise ValueError(
                    f"fine/coarse length mismatch: "
                    f"{len(fine_tokens)} vs {len(coarse_tokens)}"
                )

            input_texts.append(example["input_ids"])
            fine_ids = [self._token_id(tok) for tok in fine_tokens]
            coarse_ids = [self._token_id(tok) for tok in coarse_tokens]

            fine_label_rows.append(fine_ids)
            coarse_label_rows.append(coarse_ids)
            max_code_len = max(max_code_len, len(fine_ids))

        inputs = self.tokenizer(
            input_texts,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=self.max_length,
            return_attention_mask=True,
            add_special_tokens=True,
        )

        batch_size = len(batch)

        fine_labels = torch.full(
            (batch_size, max_code_len),
            fill_value=IGNORE_INDEX,
            dtype=torch.long,
        )
        coarse_labels = torch.full(
            (batch_size, max_code_len),
            fill_value=IGNORE_INDEX,
            dtype=torch.long,
        )

        for i in range(batch_size):
            fine_len = len(fine_label_rows[i])
            coarse_len = len(coarse_label_rows[i])

            fine_labels[i, :fine_len] = torch.tensor(
                fine_label_rows[i],
                dtype=torch.long,
            )
            coarse_labels[i, :coarse_len] = torch.tensor(
                coarse_label_rows[i],
                dtype=torch.long,
            )

        inputs["fine_labels"] = fine_labels
        inputs["coarse_labels"] = coarse_labels

        return inputs


class TestCollator(object):
    """
    Test collator remains prompt-only.

    test_ddp.py performs latent interleaved inference internally and decodes
    only fine tokens.
    """

    def __init__(self, args, tokenizer):
        self.args = args
        self.tokenizer = tokenizer

        if self.tokenizer.pad_token_id is None:
            if getattr(self.tokenizer, "eos_token_id", None) is not None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
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
            add_special_tokens=True,
        )

        return inputs, targets, coarse_targets
