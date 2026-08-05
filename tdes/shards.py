from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from .canonical import (
    atomic_write_json,
    canonical_dumps,
    hash_object,
    read_json,
    sha256_bytes,
    sha256_file,
)
from .tokenizer import FrozenByteTokenizer


ROLE_USES = {
    "train": ["train"],
    "proxy": ["score"],
    "validation": ["validate"],
    "eval": ["evaluate"],
}


class IntegrityError(RuntimeError):
    pass


class FirewallError(RuntimeError):
    pass


def _tokenize_document(document: dict[str, Any], tokenizer: FrozenByteTokenizer) -> dict[str, Any]:
    token_fields = {
        key: tokenizer.encode(value) for key, value in document["fields"].items()
    }
    row = {
        "record_id": document["record_id"],
        "role": document["role"],
        "lane": document["lane"],
        "data_type": document["data_type"],
        "token_fields": token_fields,
        "source_content_hash": document["content_hash"],
        "source_id": document["source_id"],
        "source_index": document["source_index"],
        "metadata": document["metadata"],
    }
    row["token_record_hash"] = hash_object(row)
    return row


def build_shards(
    artifacts: Path,
    documents: list[dict[str, Any]],
    tokenizer: FrozenByteTokenizer,
) -> dict[str, Any]:
    shards_dir = artifacts / "shards"
    manifests_dir = artifacts / "manifests"
    shards_dir.mkdir(parents=True, exist_ok=True)
    manifests_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(manifests_dir / "tokenizer.json", tokenizer.artifact())

    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for document in documents:
        grouped[(document["role"], document["lane"], document["data_type"])].append(
            _tokenize_document(document, tokenizer)
        )

    manifest_summaries: list[dict[str, Any]] = []
    for (role, lane, data_type), rows in sorted(grouped.items()):
        rows.sort(key=lambda row: row["record_id"])
        shard_bytes = "".join(canonical_dumps(row) + "\n" for row in rows).encode("utf-8")
        shard_hash = sha256_bytes(shard_bytes)
        shard_id = f"{role}-{lane}-{data_type}-{shard_hash[:16]}"
        shard_name = f"{shard_id}.jsonl"
        shard_path = shards_dir / shard_name
        from .canonical import atomic_write_bytes

        atomic_write_bytes(shard_path, shard_bytes, mode=0o444)
        token_count = sum(
            len(tokens) for row in rows for tokens in row["token_fields"].values()
        )
        manifest_body = {
            "schema_version": 1,
            "shard_id": shard_id,
            "shard_path": f"shards/{shard_name}",
            "shard_sha256": shard_hash,
            "shard_bytes": len(shard_bytes),
            "record_ids": [row["record_id"] for row in rows],
            "record_hashes": [row["token_record_hash"] for row in rows],
            "record_count": len(rows),
            "token_count": token_count,
            "tokenizer_hash": tokenizer.tokenizer_hash,
            "source_aggregate_hash": hash_object(
                [row["source_content_hash"] for row in rows]
            ),
            "role": role,
            "allowed_uses": ROLE_USES[role],
            "lane": lane,
            "data_type": data_type,
        }
        manifest = {**manifest_body, "manifest_hash": hash_object(manifest_body)}
        manifest_name = f"{shard_id}.manifest.json"
        atomic_write_json(manifests_dir / manifest_name, manifest, mode=0o444)
        manifest_summaries.append(
            {
                "manifest_path": f"manifests/{manifest_name}",
                "manifest_hash": manifest["manifest_hash"],
                "shard_id": shard_id,
            }
        )

    root_body = {
        "schema_version": 1,
        "tokenizer_hash": tokenizer.tokenizer_hash,
        "manifests": manifest_summaries,
    }
    root_manifest = {**root_body, "root_manifest_hash": hash_object(root_body)}
    atomic_write_json(manifests_dir / "root.json", root_manifest)
    return root_manifest


class ShardRepository:
    def __init__(self, artifacts: Path, tokenizer: FrozenByteTokenizer):
        self.artifacts = artifacts
        self.tokenizer = tokenizer
        self.root = read_json(artifacts / "manifests" / "root.json")
        self.manifests: list[dict[str, Any]] = []
        self.records_by_shard: dict[str, list[dict[str, Any]]] = {}
        self._load_and_validate()

    def _load_and_validate(self) -> None:
        root_body = {key: value for key, value in self.root.items() if key != "root_manifest_hash"}
        if hash_object(root_body) != self.root["root_manifest_hash"]:
            raise IntegrityError("root manifest hash mismatch")
        if self.root["tokenizer_hash"] != self.tokenizer.tokenizer_hash:
            raise IntegrityError("root tokenizer hash mismatch")
        for summary in self.root["manifests"]:
            path = self.artifacts / summary["manifest_path"]
            manifest = read_json(path)
            body = {key: value for key, value in manifest.items() if key != "manifest_hash"}
            if hash_object(body) != manifest["manifest_hash"]:
                raise IntegrityError(f"manifest hash mismatch: {path}")
            if manifest["manifest_hash"] != summary["manifest_hash"]:
                raise IntegrityError(f"root/manifest mismatch: {path}")
            if manifest["tokenizer_hash"] != self.tokenizer.tokenizer_hash:
                raise IntegrityError(f"tokenizer mismatch: {path}")
            shard_path = self.artifacts / manifest["shard_path"]
            if sha256_file(shard_path) != manifest["shard_sha256"]:
                raise IntegrityError(f"shard hash mismatch: {shard_path}")
            rows: list[dict[str, Any]] = []
            with shard_path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    row = json.loads(line)
                    row_body = {
                        key: value for key, value in row.items() if key != "token_record_hash"
                    }
                    if hash_object(row_body) != row["token_record_hash"]:
                        raise IntegrityError(f"record hash mismatch: {row['record_id']}")
                    rows.append(row)
            if [row["record_id"] for row in rows] != manifest["record_ids"]:
                raise IntegrityError(f"record ordering mismatch: {shard_path}")
            self.manifests.append(manifest)
            self.records_by_shard[manifest["shard_id"]] = rows

    def manifests_for(self, use: str, lane: str | None = None) -> list[dict[str, Any]]:
        selected = []
        for manifest in self.manifests:
            if lane is not None and manifest["lane"] != lane:
                continue
            if use not in manifest["allowed_uses"]:
                raise FirewallError(
                    f"shard {manifest['shard_id']} role={manifest['role']} blocks use={use}"
                )
            selected.append(manifest)
        return selected

    def records(self, use: str, lane: str | None = None) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        for manifest in self.manifests:
            if lane is not None and manifest["lane"] != lane:
                continue
            if use in manifest["allowed_uses"]:
                for row in self.records_by_shard[manifest["shard_id"]]:
                    records.append(
                        {**row, "shard_id": manifest["shard_id"], "shard_hash": manifest["shard_sha256"]}
                    )
        return sorted(records, key=lambda row: row["record_id"])

    def require_manifest_use(self, shard_id: str, use: str) -> None:
        manifest = next(item for item in self.manifests if item["shard_id"] == shard_id)
        if use not in manifest["allowed_uses"]:
            raise FirewallError(
                f"shard {shard_id} role={manifest['role']} blocks use={use}"
            )
