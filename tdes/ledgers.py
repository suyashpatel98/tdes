from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .canonical import canonical_dumps, hash_object, read_jsonl


class LedgerError(RuntimeError):
    pass


class RunLogger:
    def __init__(self, path: Path):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, message: str) -> None:
        timestamp = datetime.now(timezone.utc).isoformat(timespec="milliseconds")
        line = f"[{timestamp}] {message}\n"
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(line)
            handle.flush()
            os.fsync(handle.fileno())


class HashLedger:
    def __init__(self, path: Path, ledger_name: str):
        self.path = path
        self.ledger_name = ledger_name
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            path.touch()
        self.rows = read_jsonl(path)
        self.validate()

    def validate(self) -> None:
        previous = "0" * 64
        for expected_offset, row in enumerate(self.rows):
            if row.get("ledger") != self.ledger_name:
                raise LedgerError(f"wrong ledger name in {self.path}")
            if row.get("ledger_offset") != expected_offset:
                raise LedgerError(f"non-contiguous offset in {self.path}")
            if row.get("previous_row_hash") != previous:
                raise LedgerError(f"broken previous hash in {self.path}")
            body = {key: value for key, value in row.items() if key != "row_hash"}
            if hash_object(body) != row.get("row_hash"):
                raise LedgerError(f"row hash mismatch in {self.path}")
            previous = row["row_hash"]

    def append(self, payload: dict[str, Any]) -> dict[str, Any]:
        forbidden = {"ledger", "ledger_offset", "previous_row_hash", "row_hash"}
        if forbidden.intersection(payload):
            raise LedgerError("payload contains reserved ledger fields")
        body = {
            "ledger": self.ledger_name,
            "ledger_offset": len(self.rows),
            "previous_row_hash": self.rows[-1]["row_hash"] if self.rows else "0" * 64,
            **payload,
        }
        row = {**body, "row_hash": hash_object(body)}
        encoded = (canonical_dumps(row) + "\n").encode("utf-8")
        with self.path.open("ab") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        self.rows.append(row)
        return row

    def state(self) -> dict[str, Any]:
        return {
            "ledger": self.ledger_name,
            "rows": len(self.rows),
            "byte_offset": self.path.stat().st_size,
            "tail_hash": self.rows[-1]["row_hash"] if self.rows else "0" * 64,
        }

    @classmethod
    def reconcile(cls, path: Path, ledger_name: str, expected: dict[str, Any]) -> "HashLedger":
        if not path.exists():
            if expected["byte_offset"] != 0:
                raise LedgerError(f"missing committed ledger: {path}")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.touch()
        size = path.stat().st_size
        if size < expected["byte_offset"]:
            raise LedgerError(f"ledger shorter than checkpoint: {path}")
        if size > expected["byte_offset"]:
            with path.open("r+b") as handle:
                handle.truncate(expected["byte_offset"])
                handle.flush()
                os.fsync(handle.fileno())
        ledger = cls(path, ledger_name)
        actual = ledger.state()
        for key in ("rows", "byte_offset", "tail_hash"):
            if actual[key] != expected[key]:
                raise LedgerError(f"checkpoint/ledger {key} mismatch: {path}")
        return ledger


class LedgerSet:
    NAMES = ("consumption", "learning", "opus", "events")

    def __init__(
        self,
        ledgers_dir: Path,
        branch_id: str,
        expected_states: dict[str, dict[str, Any]] | None = None,
    ):
        self.branch_id = branch_id
        self.ledgers_dir = ledgers_dir
        self.items: dict[str, HashLedger] = {}
        for name in self.NAMES:
            path = ledgers_dir / f"{branch_id}.{name}.jsonl"
            if expected_states is None:
                self.items[name] = HashLedger(path, name)
            else:
                self.items[name] = HashLedger.reconcile(path, name, expected_states[name])

    def __getitem__(self, name: str) -> HashLedger:
        return self.items[name]

    def states(self) -> dict[str, dict[str, Any]]:
        return {name: ledger.state() for name, ledger in self.items.items()}

    def validate(self) -> None:
        for ledger in self.items.values():
            ledger.validate()
