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
    Fast HoloRec-Qwen training collator.

    It converts each example into one standard CausalLM sequence:

        prompt + coarse_1 + fine_1 + coarse_2 + fine_2 + ...

    Labels are masked on the prompt and active only on the target part.
    This makes training a single forward pass, instead of calling Qwen once
    for each coarse/fine step.
    """

    def __init__(self, args, tokenizer):
        self.args = args
        self.tokenizer = tokenizer
        self.max_length = int(getattr(args, "model_max_length", tokenizer.model_max_length))
        self.add_eos_token = bool(getattr(args, "add_eos_token", True))
        self.coarse_loss_weight = float(getattr(args, "coarse_loss_weight", 1.0))
        self.fine_loss_weight = float(getattr(args, "fine_loss_weight", 1.0))
        self._token_id_cache = {}

        if self.tokenizer.pad_token_id is None:
            if getattr(self.tokenizer, "eos_token_id", None) is not None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            else:
                self.tokenizer.pad_token_id = 0

        self.tokenizer.padding_side = "right"

    def _token_id(self, token):
        if token not in self._token_id_cache:
            self._token_id_cache[token] = _token_to_id(self.tokenizer, token)
        return self._token_id_cache[token]

    def _encode_prompt(self, text):
        encoded = self.tokenizer(
            text,
            add_special_tokens=True,
            truncation=False,
            return_attention_mask=False,
        )
        return _flatten_tokenizer_ids(encoded)

    def _build_target_ids_and_weights(self, fine_tokens, coarse_tokens):
        if coarse_tokens is None:
            target_ids = [self._token_id(tok) for tok in fine_tokens]
            weights = [self.fine_loss_weight for _ in target_ids]
        else:
            if len(fine_tokens) != len(coarse_tokens):
                raise ValueError(
                    f"fine/coarse length mismatch: {len(fine_tokens)} vs {len(coarse_tokens)}"
                )

            target_ids = []
            weights = []
            for coarse_tok, fine_tok in zip(coarse_tokens, fine_tokens):
                target_ids.append(self._token_id(coarse_tok))
                weights.append(self.coarse_loss_weight)
                target_ids.append(self._token_id(fine_tok))
                weights.append(self.fine_loss_weight)

        if self.add_eos_token and self.tokenizer.eos_token_id is not None:
            target_ids.append(int(self.tokenizer.eos_token_id))
            weights.append(self.fine_loss_weight)

        return target_ids, weights

    def __call__(self, batch):
        input_ids_list = []
        labels_list = []
        weights_list = []

        for example in batch:
            if "fine_tokens" not in example:
                raise ValueError(
                    "Fast HoloRec-Qwen training requires `fine_tokens` in each dataset item. "
                    "Please train with --tasks seqrec."
                )

            fine_tokens = example["fine_tokens"]
            coarse_tokens = example.get("coarse_tokens", None)

            prompt_ids = self._encode_prompt(example["input_ids"])
            target_ids, target_weights = self._build_target_ids_and_weights(
                fine_tokens=fine_tokens,
                coarse_tokens=coarse_tokens,
            )

            if len(target_ids) >= self.max_length:
                raise ValueError(
                    f"Target code length {len(target_ids)} >= model_max_length {self.max_length}. "
                    "Increase --model_max_length or shorten the item code."
                )

            max_prompt_len = self.max_length - len(target_ids)
            if len(prompt_ids) > max_prompt_len:
                prompt_ids = prompt_ids[-max_prompt_len:]

            ids = prompt_ids + target_ids
            labels = [IGNORE_INDEX] * len(prompt_ids) + target_ids
            loss_weights = [0.0] * len(prompt_ids) + target_weights

            input_ids_list.append(ids)
            labels_list.append(labels)
            weights_list.append(loss_weights)

        batch_max_len = max(len(x) for x in input_ids_list)
        pad_id = int(self.tokenizer.pad_token_id)

        input_ids = torch.full(
            (len(batch), batch_max_len),
            fill_value=pad_id,
            dtype=torch.long,
        )
        attention_mask = torch.zeros(
            (len(batch), batch_max_len),
            dtype=torch.long,
        )
        labels = torch.full(
            (len(batch), batch_max_len),
            fill_value=IGNORE_INDEX,
            dtype=torch.long,
        )
        loss_weights = torch.zeros(
            (len(batch), batch_max_len),
            dtype=torch.float,
        )

        for i, (ids, labs, weights) in enumerate(
            zip(input_ids_list, labels_list, weights_list)
        ):
            length = len(ids)
            input_ids[i, :length] = torch.tensor(ids, dtype=torch.long)
            attention_mask[i, :length] = 1
            labels[i, :length] = torch.tensor(labs, dtype=torch.long)
            loss_weights[i, :length] = torch.tensor(weights, dtype=torch.float)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "loss_weights": loss_weights,
        }


class TestCollator(object):
    """
    Test collator remains prompt-only.
    Target labels are returned separately for evaluate.py / test_ddp.py.
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
        )

        return inputs, targets, coarse_targets
