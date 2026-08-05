from __future__ import annotations

from pathlib import Path
from typing import Any

from .canonical import atomic_write_bytes, atomic_write_json, hash_object


DISPLAY_NAMES = {
    "tokenizer_integrity": "Tokenizer integrity",
    "evaluation_firewall": "Evaluation firewall",
    "packing_correctness": "Packing correctness",
    "mixture_compliance": "Mixture compliance",
    "opus_audit_trail": "OPUS audit trail",
    "crash_recovery": "Crash recovery",
    "replay": "Replay",
    "learning_trace": "Learning trace",
    "fork": "Branch fork",
    "throughput": "Throughput",
}


def generate_evidence(
    artifacts: Path, audit: dict[str, Any], run_identity: dict[str, Any]
) -> dict[str, Any]:
    requirements = {}
    for identifier, check in audit["checks"].items():
        requirements[identifier] = {
            "requirement": DISPLAY_NAMES[identifier],
            "result": "PASS" if check["passed"] else "FAIL",
            "passed": check["passed"],
            "evidence": check["evidence"],
            "measurements": check["details"],
            "audit_report": "reports/audit.json",
        }
    body = {
        "schema_version": 1,
        "run_identity": run_identity,
        "overall_result": "PASS" if audit["overall_passed"] else "FAIL",
        "audit_hash": audit["audit_hash"],
        "requirements": requirements,
    }
    evidence = {**body, "evidence_hash": hash_object(body)}
    atomic_write_json(artifacts / "evidence.json", evidence)

    lines = [
        "# Training Data Execution System V5 - Evidence",
        "",
        f"Overall result: **{evidence['overall_result']}**",
        "",
        "| Requirement | Result | Evidence |",
        "|---|---:|---|",
    ]
    for identifier in DISPLAY_NAMES:
        item = requirements[identifier]
        references = ", ".join(f"`{path}`" for path in item["evidence"])
        lines.append(f"| {item['requirement']} | {item['result']} | {references} |")
    lines.extend(
        [
            "",
            "This file is rendered from `evidence.json`, whose results are produced by the",
            "independent artifact audit in `reports/audit.json`.",
            "",
        ]
    )
    atomic_write_bytes(artifacts / "evidence.md", "\n".join(lines).encode("utf-8"))
    return evidence
