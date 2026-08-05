from __future__ import annotations

import os
import unittest

from tdes.canonical import hash_object
from tdes.shards import IntegrityError, ShardRepository
from tdes.tokenizer import FrozenByteTokenizer

from tests.support import RepositoryFixture


class TokenizerShardTests(unittest.TestCase):
    def setUp(self) -> None:
        self.fixture = RepositoryFixture()

    def tearDown(self) -> None:
        self.fixture.close()

    def test_frozen_byte_round_trip_and_hash(self) -> None:
        tokenizer = self.fixture.tokenizer
        text = "Causal data: cafe."
        self.assertEqual(tokenizer.decode(tokenizer.encode(text)), text)
        self.assertEqual(tokenizer.tokenizer_hash, hash_object(tokenizer.spec))
        self.assertEqual(tokenizer.vocab_size, 262)

    def test_every_clean_document_is_tokenized_once(self) -> None:
        records = sum(len(rows) for rows in self.fixture.repository.records_by_shard.values())
        self.assertEqual(records, len(self.fixture.documents))
        self.assertEqual(records, 82)

    def test_shard_tampering_is_detected(self) -> None:
        manifest = self.fixture.repository.manifests[0]
        path = self.fixture.artifacts / manifest["shard_path"]
        os.chmod(path, 0o644)
        with path.open("ab") as handle:
            handle.write(b"{}\n")
        with self.assertRaises(IntegrityError):
            ShardRepository(self.fixture.artifacts, FrozenByteTokenizer.create())


if __name__ == "__main__":
    unittest.main()
