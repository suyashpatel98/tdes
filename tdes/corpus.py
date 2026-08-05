from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path
from typing import Any

from .canonical import hash_object, read_json, sha256_file


def _clean_text(value: str) -> str:
    value = unicodedata.normalize("NFKC", value)
    value = value.replace("\ufeff", "")
    return " ".join(value.split()).strip()


def verify_sources(root: Path) -> dict[str, Any]:
    manifest_path = root / "corpus" / "SOURCES.json"
    manifest = read_json(manifest_path)
    checked: list[dict[str, Any]] = []
    for entry in manifest["sources"] + manifest.get("supporting_files", []):
        path = root / "corpus" / entry["path"]
        actual_size = path.stat().st_size
        actual_hash = sha256_file(path)
        if actual_size != entry["bytes"] or actual_hash != entry["sha256"]:
            raise ValueError(f"source integrity failure: {entry['path']}")
        checked.append(
            {"path": entry["path"], "bytes": actual_size, "sha256": actual_hash}
        )
    total_bytes = sum(item["bytes"] for item in checked)
    return {
        "source_manifest_hash": hash_object(manifest),
        "checked_files": checked,
        "total_bytes": total_bytes,
    }


def _alice_sentences(path: Path) -> list[tuple[int, str]]:
    text = path.read_text(encoding="utf-8-sig")
    start = text.find("*** START OF THE PROJECT GUTENBERG EBOOK")
    end = text.find("*** END OF THE PROJECT GUTENBERG EBOOK")
    if start < 0 or end < 0 or end <= start:
        raise ValueError("Project Gutenberg boundaries not found")
    body = text[start:end]
    candidates = re.split(r"(?<=[.!?])(?:[\"']*)\s+", body)
    result: list[tuple[int, str]] = []
    for index, candidate in enumerate(candidates):
        cleaned = _clean_text(candidate)
        lowered = cleaned.lower()
        if (
            28 <= len(cleaned.encode("utf-8")) <= 58
            and "chapter" not in lowered
            and "gutenberg" not in lowered
            and any(character.isalpha() for character in cleaned)
        ):
            result.append((index, cleaned))
    return result


def _code_records(path: Path) -> list[tuple[int, str]]:
    result: list[tuple[int, str]] = []
    for index, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
        cleaned = line.strip()
        if (
            10 <= len(cleaned.encode("utf-8")) <= 56
            and not cleaned.startswith("#")
            and not cleaned.startswith(('"""', "'''"))
            and any(token in cleaned for token in ("def ", "return ", "if ", "for ", "=", "raise "))
        ):
            result.append((index, cleaned))
    return result


def _instruction_records(path: Path) -> list[tuple[int, str, str, str]]:
    result: list[tuple[int, str, str, str]] = []
    with path.open("r", encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            row = json.loads(line)
            prompt = _clean_text(row["instruction"])
            response = _clean_text(row["response"])
            if (
                not row.get("context")
                and 8 <= len(prompt.encode("utf-8")) <= 36
                and 4 <= len(response.encode("utf-8")) <= 32
            ):
                result.append((index, prompt, response, row["category"]))
    return result


def build_documents(root: Path, config: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    verification = verify_sources(root)
    requested = config["corpus"]
    alice = _alice_sentences(root / "corpus" / "raw" / "alice_in_wonderland.txt")
    code = _code_records(root / "corpus" / "raw" / "cpython_statistics.py")
    instructions = _instruction_records(root / "corpus" / "raw" / "dolly.jsonl")
    alice_needed = (
        requested["general_records"]
        + requested["proxy_records"]
        + requested["validation_records"]
        + requested["eval_records"]
    )
    if len(alice) < alice_needed or len(code) < requested["code_records"]:
        raise ValueError("not enough deterministic source records")
    if len(instructions) < requested["instruction_records"]:
        raise ValueError("not enough short Dolly instruction records")

    documents: list[dict[str, Any]] = []

    def add_document(
        record_id: str,
        role: str,
        lane: str,
        data_type: str,
        fields: dict[str, str],
        source_id: str,
        source_index: int,
        extra: dict[str, Any] | None = None,
    ) -> None:
        content = {
            "record_id": record_id,
            "role": role,
            "lane": lane,
            "data_type": data_type,
            "fields": fields,
            "source_id": source_id,
            "source_index": source_index,
            "metadata": extra or {},
        }
        content["content_hash"] = hash_object(content)
        documents.append(content)

    alice_cursor = 0
    role_counts = [
        ("train", requested["general_records"]),
        ("proxy", requested["proxy_records"]),
        ("validation", requested["validation_records"]),
        ("eval", requested["eval_records"]),
    ]
    for role, count in role_counts:
        for local_index in range(count):
            source_index, text = alice[alice_cursor]
            alice_cursor += 1
            add_document(
                f"alice-{role}-{local_index:03d}",
                role,
                "general",
                "document",
                {"text": text},
                "alice_in_wonderland_gutenberg_11",
                source_index,
            )

    for local_index, (source_index, text) in enumerate(code[: requested["code_records"]]):
        add_document(
            f"cpython-train-{local_index:03d}",
            "train",
            "code",
            "document",
            {"text": text},
            "cpython_statistics_3_13_0",
            source_index,
        )

    for local_index, (source_index, prompt, response, category) in enumerate(
        instructions[: requested["instruction_records"]]
    ):
        add_document(
            f"dolly-train-{local_index:03d}",
            "train",
            "instruction",
            "prompt_completion",
            {"prompt": prompt, "response": response},
            "databricks_dolly_15k",
            source_index,
            {"category": category},
        )

    if len({row["record_id"] for row in documents}) != len(documents):
        raise ValueError("duplicate document ID")
    if len({row["content_hash"] for row in documents}) != len(documents):
        raise ValueError("duplicate cleaned document content")

    report = {
        **verification,
        "document_count": len(documents),
        "roles": {
            role: sum(row["role"] == role for row in documents)
            for role in ("train", "proxy", "validation", "eval")
        },
        "lanes": {
            lane: sum(row["lane"] == lane for row in documents)
            for lane in ("general", "code", "instruction")
        },
        "documents_hash": hash_object(documents),
    }
    return documents, report
