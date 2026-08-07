from __future__ import annotations

import os
import unittest

from tdes.canonical import atomic_write_json, hash_object, read_json
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

    def test_tokenizer_artifact_cannot_silently_replace_runtime_spec(self) -> None:
        path = self.fixture.artifacts / "manifests" / "tokenizer.json"
        artifact = read_json(path)
        artifact["spec"]["vocab_size"] += 1
        artifact["tokenizer_hash"] = hash_object(artifact["spec"])
        atomic_write_json(path, artifact)
        with self.assertRaisesRegex(IntegrityError, "runtime tokenizer"):
            ShardRepository(self.fixture.artifacts, FrozenByteTokenizer.create())

    def test_rehashed_false_manifest_metadata_is_detected(self) -> None:
        root_path = self.fixture.artifacts / "manifests" / "root.json"
        root = read_json(root_path)
        summary = root["manifests"][0]
        manifest_path = self.fixture.artifacts / summary["manifest_path"]
        manifest = read_json(manifest_path)
        manifest["token_count"] += 1
        manifest_body = {
            key: value for key, value in manifest.items() if key != "manifest_hash"
        }
        manifest["manifest_hash"] = hash_object(manifest_body)
        atomic_write_json(manifest_path, manifest)
        summary["manifest_hash"] = manifest["manifest_hash"]
        root_body = {
            key: value for key, value in root.items() if key != "root_manifest_hash"
        }
        root["root_manifest_hash"] = hash_object(root_body)
        atomic_write_json(root_path, root)
        with self.assertRaisesRegex(IntegrityError, "token count"):
            ShardRepository(self.fixture.artifacts, FrozenByteTokenizer.create())

    def test_rehashed_firewall_policy_change_is_detected(self) -> None:
        root_path = self.fixture.artifacts / "manifests" / "root.json"
        root = read_json(root_path)
        summary = root["manifests"][0]
        manifest_path = self.fixture.artifacts / summary["manifest_path"]
        manifest = read_json(manifest_path)
        manifest["allowed_uses"] = sorted(set(manifest["allowed_uses"] + ["train"]))
        manifest_body = {
            key: value for key, value in manifest.items() if key != "manifest_hash"
        }
        manifest["manifest_hash"] = hash_object(manifest_body)
        atomic_write_json(manifest_path, manifest)
        summary["manifest_hash"] = manifest["manifest_hash"]
        root_body = {
            key: value for key, value in root.items() if key != "root_manifest_hash"
        }
        root["root_manifest_hash"] = hash_object(root_body)
        atomic_write_json(root_path, root)
        with self.assertRaisesRegex(IntegrityError, "role/use policy"):
            ShardRepository(self.fixture.artifacts, FrozenByteTokenizer.create())


if __name__ == "__main__":
    unittest.main()
