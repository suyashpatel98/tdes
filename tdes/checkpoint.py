from __future__ import annotations

from pathlib import Path
from typing import Any

from .canonical import atomic_write_json, hash_object, read_json


class CheckpointError(RuntimeError):
    pass


def write_checkpoint(path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    checkpoint_hash = hash_object(payload)
    document = {"payload": payload, "checkpoint_hash": checkpoint_hash}
    atomic_write_json(path, document)
    loaded = load_checkpoint(path)
    if loaded["checkpoint_hash"] != checkpoint_hash:
        raise CheckpointError("checkpoint read-after-write verification failed")
    return loaded


def load_checkpoint(path: Path) -> dict[str, Any]:
    document = read_json(path)
    if not isinstance(document, dict) or "payload" not in document:
        raise CheckpointError(f"invalid checkpoint structure: {path}")
    actual = hash_object(document["payload"])
    if actual != document.get("checkpoint_hash"):
        raise CheckpointError(f"checkpoint hash mismatch: {path}")
    return document
