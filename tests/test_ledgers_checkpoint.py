from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from tdes.checkpoint import CheckpointError, load_checkpoint, write_checkpoint
from tdes.ledgers import HashLedger, LedgerError


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


if __name__ == "__main__":
    unittest.main()
