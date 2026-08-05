from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable


def canonical_dumps(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def canonical_bytes(value: Any) -> bytes:
    return canonical_dumps(value).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def hash_object(value: Any) -> str:
    return sha256_bytes(canonical_bytes(value))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: JSONL row is not an object")
            rows.append(row)
    return rows


def atomic_write_bytes(path: Path, data: bytes, mode: int | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("wb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    if mode is not None:
        path.chmod(mode)


def atomic_write_json(path: Path, value: Any, mode: int | None = None) -> None:
    atomic_write_bytes(path, (canonical_dumps(value) + "\n").encode("utf-8"), mode)


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]], mode: int | None = None) -> None:
    data = "".join(canonical_dumps(row) + "\n" for row in rows).encode("utf-8")
    atomic_write_bytes(path, data, mode)


def semantic_code_hash(root: Path) -> str:
    records: list[dict[str, str]] = []
    for path in sorted(root.rglob("*.py")):
        if "submission_artifacts" in path.parts or "__pycache__" in path.parts:
            continue
        records.append({"path": str(path.relative_to(root)), "sha256": sha256_file(path)})
    return hash_object(records)


def stable_apportion(total: int, weights: dict[str, float]) -> dict[str, int]:
    if total < 0 or not weights or any(value < 0 for value in weights.values()):
        raise ValueError("invalid apportionment inputs")
    weight_sum = sum(weights.values())
    if weight_sum <= 0:
        raise ValueError("weights must have positive sum")
    exact = {key: total * value / weight_sum for key, value in weights.items()}
    result = {key: int(value) for key, value in exact.items()}
    remaining = total - sum(result.values())
    order = sorted(weights, key=lambda key: (-(exact[key] - result[key]), key))
    for key in order[:remaining]:
        result[key] += 1
    return result
