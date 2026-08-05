from __future__ import annotations

import json
import tempfile
from pathlib import Path

from tdes.corpus import build_documents
from tdes.shards import ShardRepository, build_shards
from tdes.tokenizer import FrozenByteTokenizer


ROOT = Path(__file__).resolve().parents[1]


class RepositoryFixture:
    def __init__(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.artifacts = Path(self.temporary.name)
        self.config = json.loads((ROOT / "configs" / "demo.json").read_text())
        self.tokenizer = FrozenByteTokenizer.create()
        self.documents, self.source_report = build_documents(ROOT, self.config)
        build_shards(self.artifacts, self.documents, self.tokenizer)
        self.repository = ShardRepository(self.artifacts, self.tokenizer)

    def close(self) -> None:
        self.temporary.cleanup()
