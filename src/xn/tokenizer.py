import json
import re
from collections import Counter
from itertools import pairwise
from pathlib import Path

# split text into chunks so merges never cross word / script boundaries
_SPLIT = re.compile(r" ?[A-Za-z_][A-Za-z0-9_]*| ?[0-9]+|[\u3040-\u309f]+|[\u30a0-\u30ff]+|[\u4e00-\u9fff]+|\s+|.")


class ByteTokenizer:
    """UTF-8 byte-level tokenizer. Handles any language (incl. Japanese) with vocab_size=256."""

    vocab_size = 256

    def encode(self, text: str) -> list[int]:
        return list(text.encode("utf-8"))

    def decode(self, ids: list[int]) -> str:
        return bytes(ids).decode("utf-8", errors="replace")

    def to_dict(self) -> dict:
        return {"type": "byte"}


def _merge(ids: tuple[int, ...], pair: tuple[int, int], new_id: int) -> tuple[int, ...]:
    out, i = [], 0
    while i < len(ids):
        if i + 1 < len(ids) and (ids[i], ids[i + 1]) == pair:
            out.append(new_id)
            i += 2
        else:
            out.append(ids[i])
            i += 1
    return tuple(out)


class BPETokenizer:
    """Byte-level BPE tokenizer trained on your own corpus."""

    def __init__(self, merges: list[tuple[int, int]]):
        self.merges = [tuple(m) for m in merges]
        self.ranks = {pair: 256 + i for i, pair in enumerate(self.merges)}
        self.vocab = {i: bytes([i]) for i in range(256)}
        for (a, b), idx in self.ranks.items():
            self.vocab[idx] = self.vocab[a] + self.vocab[b]
        self._cache: dict[str, list[int]] = {}

    @property
    def vocab_size(self) -> int:
        return 256 + len(self.merges)

    @classmethod
    def train(cls, text: str, vocab_size: int, verbose: bool = False) -> "BPETokenizer":
        assert vocab_size >= 256, "vocab_size must be >= 256"
        words = Counter(tuple(w.encode("utf-8")) for w in _SPLIT.findall(text))
        merges = []
        for new_id in range(256, vocab_size):
            pairs: Counter = Counter()
            for w, n in words.items():
                for p in pairwise(w):
                    pairs[p] += n
            if not pairs:
                break
            pair, count = pairs.most_common(1)[0]
            if count < 2:
                break
            merges.append(pair)
            words = Counter({_merge(w, pair, new_id): n for w, n in words.items()})
            if verbose and len(merges) % 100 == 0:
                print(f"bpe: {len(merges) + 256}/{vocab_size}")
        return cls(merges)

    def _encode_chunk(self, chunk: str) -> list[int]:
        if chunk in self._cache:
            return self._cache[chunk]
        ids = tuple(chunk.encode("utf-8"))
        while len(ids) > 1:
            pair = min(pairwise(ids), key=lambda p: self.ranks.get(p, float("inf")))
            if pair not in self.ranks:
                break
            ids = _merge(ids, pair, self.ranks[pair])
        self._cache[chunk] = list(ids)
        return self._cache[chunk]

    def encode(self, text: str) -> list[int]:
        return [i for chunk in _SPLIT.findall(text) for i in self._encode_chunk(chunk)]

    def decode(self, ids: list[int]) -> str:
        return b"".join(self.vocab[i] for i in ids).decode("utf-8", errors="replace")

    def to_dict(self) -> dict:
        return {"type": "bpe", "merges": [list(m) for m in self.merges]}

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict()))


def tokenizer_from_dict(d: dict) -> ByteTokenizer | BPETokenizer:
    return BPETokenizer(d["merges"]) if d["type"] == "bpe" else ByteTokenizer()
