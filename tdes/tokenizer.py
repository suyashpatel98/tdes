from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .canonical import hash_object


TOKENIZER_SPEC: dict[str, Any] = {
    "schema_version": 1,
    "type": "utf8_bytes",
    "normalization": "none_after_corpus_cleaning",
    "special_tokens": {
        "PAD": 0,
        "BOS": 1,
        "EOS": 2,
        "PROMPT": 3,
        "RESPONSE": 4,
        "UNK": 5,
    },
    "byte_offset": 6,
    "byte_values": 256,
    "vocab_size": 262,
}


@dataclass(frozen=True)
class FrozenByteTokenizer:
    spec: dict[str, Any]

    @classmethod
    def create(cls) -> "FrozenByteTokenizer":
        return cls(dict(TOKENIZER_SPEC))

    @property
    def tokenizer_hash(self) -> str:
        return hash_object(self.spec)

    @property
    def vocab_size(self) -> int:
        return int(self.spec["vocab_size"])

    def special(self, name: str) -> int:
        return int(self.spec["special_tokens"][name])

    def encode(self, text: str) -> list[int]:
        offset = int(self.spec["byte_offset"])
        return [offset + value for value in text.encode("utf-8")]

    def decode(self, token_ids: list[int]) -> str:
        offset = int(self.spec["byte_offset"])
        values = bytes(token - offset for token in token_ids if token >= offset)
        return values.decode("utf-8", errors="replace")

    def artifact(self) -> dict[str, Any]:
        return {"spec": self.spec, "tokenizer_hash": self.tokenizer_hash}
