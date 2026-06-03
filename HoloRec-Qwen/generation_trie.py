from typing import Dict, List, Iterable, Optional


class Trie(object):
    """
    Token-level trie for constrained decoding.

    For latent interleaved Qwen inference:
        trie prefix = visible fine token prefix
        trie output = allowed next fine token ids

    Coarse tokens are NOT inserted into this trie.
    Coarse is latent and only affects hidden states / embeddings.
    """

    def __init__(self, sequences: Optional[List[List[int]]] = None):
        self.trie_dict = {}
        self.len = 0

        if sequences is None:
            sequences = []

        for sequence in sequences:
            self.add(sequence)

        self.append_trie = None
        self.bos_token_id = None

    def append(self, trie, bos_token_id):
        self.append_trie = trie
        self.bos_token_id = bos_token_id

    def add(self, sequence: List[int]):
        if sequence is None:
            return
        Trie._add_to_trie(sequence, self.trie_dict)
        self.len += 1

    def get(self, prefix_sequence: List[int]):
        return Trie._get_from_trie(
            prefix_sequence,
            self.trie_dict,
            self.append_trie,
            self.bos_token_id,
        )

    @staticmethod
    def load_from_dict(trie_dict: Dict):
        trie = Trie()
        trie.trie_dict = trie_dict
        trie.len = sum(1 for _ in trie)
        return trie

    @staticmethod
    def _add_to_trie(sequence: List[int], trie_dict: Dict):
        if len(sequence) == 0:
            return

        token = sequence[0]

        if token not in trie_dict:
            trie_dict[token] = {}

        Trie._add_to_trie(sequence[1:], trie_dict[token])

    @staticmethod
    def _get_from_trie(
        prefix_sequence: List[int],
        trie_dict: Dict,
        append_trie=None,
        bos_token_id: Optional[int] = None,
    ):
        if len(prefix_sequence) == 0:
            output = list(trie_dict.keys())

            if append_trie is not None and bos_token_id in output:
                output.remove(bos_token_id)
                output += list(append_trie.trie_dict.keys())

            return output

        head = prefix_sequence[0]

        if head in trie_dict:
            return Trie._get_from_trie(
                prefix_sequence[1:],
                trie_dict[head],
                append_trie,
                bos_token_id,
            )

        if append_trie is not None:
            return append_trie.get(prefix_sequence)

        return []

    def __iter__(self):
        def _traverse(prefix_sequence, trie_dict):
            if len(trie_dict) == 0:
                yield prefix_sequence
                return

            for next_token in trie_dict:
                yield from _traverse(
                    prefix_sequence + [next_token],
                    trie_dict[next_token],
                )

        return _traverse([], self.trie_dict)

    def __len__(self):
        return self.len

    def __getitem__(self, value):
        return self.get(value)


def _token_to_id(tokenizer, token: str) -> int:
    token_id = tokenizer.convert_tokens_to_ids(token)

    if token_id is None:
        encoded = tokenizer(token, add_special_tokens=False).get("input_ids", [])

        if len(encoded) == 1:
            return int(encoded[0])

        raise ValueError(
            f"Token {token!r} is not a single tokenizer id: {encoded}. "
            f"Please ensure code tokens are added to tokenizer."
        )

    unk_id = getattr(tokenizer, "unk_token_id", None)

    if unk_id is not None and token_id == unk_id:
        encoded = tokenizer(token, add_special_tokens=False).get("input_ids", [])

        if len(encoded) == 1:
            return int(encoded[0])

        raise ValueError(
            f"Token {token!r} is mapped to unk or multiple ids: {encoded}. "
            f"Please ensure code tokens are added to tokenizer."
        )

    return int(token_id)


def build_trie_from_token_sequences(token_sequences: Iterable[List[str]], tokenizer) -> Trie:
    """
    token_sequences:
        [
            ["<a_1>", "<b_2>", "<c_3>", "<d_4>"],
            ...
        ]
    """
    sequences = []

    for seq in token_sequences:
        ids = [_token_to_id(tokenizer, tok) for tok in seq]
        sequences.append(ids)

    return Trie(sequences)


def build_trie_from_texts(texts: Iterable[str], tokenizer) -> Trie:
    sequences = []

    for text in texts:
        token_ids = tokenizer.encode(text, add_special_tokens=False)
        sequences.append(token_ids)

    return Trie(sequences)


def prefix_allowed_tokens_fn(candidate_trie: Trie):
    """
    HuggingFace generate() compatible prefix function.
    Not used by latent interleaved decoder, kept for fallback compatibility.
    """

    def prefix_allowed_tokens(batch_id, sentence):
        sentence = sentence.tolist()
        trie_out = candidate_trie.get(sentence)
        return trie_out if trie_out is not None and len(trie_out) > 0 else []

    return prefix_allowed_tokens
