from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from tdes.canonical import atomic_write_json, hash_object, read_json, semantic_code_hash
from tdes.checkpoint import CheckpointError, load_checkpoint, write_checkpoint
from tdes.engine import TrainingEngine
from tdes.ledgers import HashLedger, LedgerError, RunLogger
from tests.support import ROOT, RepositoryFixture


class LedgerCheckpointTests(unittest.TestCase):
    def test_hash_chain_detects_row_mutation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "test.jsonl"
            ledger = HashLedger(path, "test")
            ledger.append({"value": 1})
            row = json.loads(path.read_text())
            row["value"] = 2
            path.write_text(json.dumps(row) + "\n")
            with self.assertRaises(LedgerError):
                HashLedger(path, "test")

    def test_reconcile_truncates_uncommitted_suffix(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "test.jsonl"
            ledger = HashLedger(path, "test")
            ledger.append({"value": 1})
            committed = ledger.state()
            ledger.append({"value": 2})
            restored = HashLedger.reconcile(path, "test", committed)
            self.assertEqual(len(restored.rows), 1)
            self.assertEqual(restored.rows[0]["value"], 1)

    def test_checkpoint_tampering_is_detected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.json"
            write_checkpoint(path, {"step": 3, "state": {"weight": 1.0}})
            document = json.loads(path.read_text())
            document["payload"]["step"] = 4
            path.write_text(json.dumps(document))
            with self.assertRaises(CheckpointError):
                load_checkpoint(path)

    def test_checkpoint_rejects_a_different_valid_root_manifest(self) -> None:
        fixture = RepositoryFixture()
        try:
            config_hash = hash_object(fixture.config)
            code_hash = semantic_code_hash(ROOT)
            logger = RunLogger(fixture.artifacts / "run.log")
            engine = TrainingEngine(
                ROOT,
                fixture.artifacts,
                fixture.config,
                config_hash,
                code_hash,
                "main",
                {
                    "branch_id": "main",
                    "parent_branch_id": None,
                    "parent_checkpoint_hash": None,
                    "fork_step": None,
                    "config_delta": {},
                },
                logger,
            )
            checkpoint = engine.save_checkpoint("main.bootstrap.json", "bootstrap")

            root_path = fixture.artifacts / "manifests" / "root.json"
            changed_root = read_json(root_path)
            changed_root["manifests"] = changed_root["manifests"][:-1]
            root_body = {
                key: value
                for key, value in changed_root.items()
                if key != "root_manifest_hash"
            }
            changed_root["root_manifest_hash"] = hash_object(root_body)
            atomic_write_json(root_path, changed_root)

            with self.assertRaisesRegex(ValueError, "checkpoint root manifest hash"):
                TrainingEngine.from_checkpoint(
                    ROOT,
                    fixture.artifacts,
                    Path(checkpoint["path"]),
                    fixture.config,
                    config_hash,
                    code_hash,
                    logger,
                )
        finally:
            fixture.close()


if __name__ == "__main__":
    unittest.main()
