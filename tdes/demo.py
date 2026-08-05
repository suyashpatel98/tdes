from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from .audit import audit_all, build_performance_report
from .canonical import (
    atomic_write_json,
    hash_object,
    read_json,
    semantic_code_hash,
    write_jsonl,
)
from .corpus import build_documents
from .evidence import generate_evidence
from .ledgers import RunLogger
from .mixture import Curriculum
from .shards import FirewallError, ShardRepository, build_shards
from .tokenizer import FrozenByteTokenizer
from .worker import EXPECTED_CRASH_CODE


def _run_worker(
    root: Path,
    artifacts: Path,
    mode: str,
    checkpoint: Path | None = None,
    expected_code: int = 0,
) -> None:
    command = [
        sys.executable,
        "-m",
        "tdes.worker",
        mode,
        "--root",
        str(root),
        "--artifacts",
        str(artifacts),
    ]
    if checkpoint is not None:
        command.extend(["--checkpoint", str(checkpoint)])
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(root)
    result = subprocess.run(
        command,
        cwd=root,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != expected_code:
        logger = RunLogger(artifacts / "run.log")
        logger.log(
            f"[FAIL] worker mode={mode} expected_exit={expected_code} "
            f"actual_exit={result.returncode} stdout={result.stdout!r} stderr={result.stderr!r}"
        )
        raise RuntimeError(
            f"{mode} worker exited {result.returncode}, expected {expected_code}\n{result.stderr}"
        )


def _firewall_probes(repository: ShardRepository) -> dict[str, Any]:
    attempts = []
    for role in ("eval", "validation", "proxy"):
        manifest = next(item for item in repository.manifests if item["role"] == role)
        blocked = False
        reason = ""
        try:
            repository.require_manifest_use(manifest["shard_id"], "train")
        except FirewallError as error:
            blocked = True
            reason = str(error)
        attempts.append(
            {
                "role": role,
                "shard_id": manifest["shard_id"],
                "requested_use": "train",
                "blocked": blocked,
                "reason": reason,
            }
        )
    body = {
        "schema_version": 1,
        "attempts": attempts,
        "all_blocked": all(item["blocked"] for item in attempts),
    }
    return {**body, "firewall_report_hash": hash_object(body)}


def run_complete_demo(root: Path) -> Path:
    build = root / ".submission_artifacts.build"
    final = root / "submission_artifacts"
    if build.exists():
        shutil.rmtree(build)
    for directory in (
        "manifests",
        "shards",
        "batches",
        "ledgers",
        "checkpoints",
        "reports",
    ):
        (build / directory).mkdir(parents=True, exist_ok=True)
    logger = RunLogger(build / "run.log")
    logger.log("[PASS] demo_started")

    config = read_json(root / "configs" / "demo.json")
    config_hash = hash_object(config)
    code_hash = semantic_code_hash(root)
    tokenizer = FrozenByteTokenizer.create()
    documents, source_report = build_documents(root, config)
    write_jsonl(build / "reports" / "documents.jsonl", documents)
    atomic_write_json(build / "reports" / "source_report.json", source_report)
    logger.log(
        f"[PASS] documents_cleaned count={len(documents)} "
        f"source_bytes={source_report['total_bytes']} hash={source_report['documents_hash']}"
    )

    root_manifest = build_shards(build, documents, tokenizer)
    logger.log(
        f"[PASS] shards_created count=6 root_hash={root_manifest['root_manifest_hash']}"
    )
    repository = ShardRepository(build, tokenizer)
    logger.log(
        f"[PASS] tokenizer_hash_verified hash={tokenizer.tokenizer_hash}"
    )
    logger.log(
        f"[PASS] manifests_validated count={len(repository.manifests)}"
    )

    firewall_report = _firewall_probes(repository)
    atomic_write_json(build / "reports" / "firewall.json", firewall_report)
    if not firewall_report["all_blocked"]:
        raise RuntimeError("firewall probe failed")
    eval_attempt = next(item for item in firewall_report["attempts"] if item["role"] == "eval")
    logger.log(
        f"[PASS] eval_shard_blocked shard={eval_attempt['shard_id']}"
    )
    logger.log("[PASS] validation_and_proxy_shards_blocked")

    mixture_plan = Curriculum(config).compiled_plan()
    atomic_write_json(build / "reports" / "mixture_plan.json", mixture_plan)
    logger.log(
        f"[PASS] mixture_compiled stages={len(config['stages'])} "
        f"hash={mixture_plan['mixture_plan_hash']}"
    )
    run_identity = {
        "schema_version": 1,
        "config_hash": config_hash,
        "code_hash": code_hash,
        "tokenizer_hash": tokenizer.tokenizer_hash,
        "root_manifest_hash": root_manifest["root_manifest_hash"],
        "source_manifest_hash": source_report["source_manifest_hash"],
        "documents_hash": source_report["documents_hash"],
    }
    run_identity["run_id"] = hash_object(run_identity)[:24]
    atomic_write_json(build / "reports" / "run_identity.json", run_identity)

    _run_worker(root, build, "fresh", expected_code=EXPECTED_CRASH_CODE)
    logger.log(
        f"[PASS] crash_worker_exit_verified exit_code={EXPECTED_CRASH_CODE}"
    )
    crash_checkpoint = build / "checkpoints" / f"main.step-{config['crash_after_step']:04d}.json"
    _run_worker(root, build, "resume", crash_checkpoint)
    logger.log("[PASS] batches_packed_and_training_resumed")

    bootstrap = build / "checkpoints" / "main.bootstrap.json"
    _run_worker(root, build, "replay", bootstrap)
    logger.log("[PASS] historical_stream_replayed")
    fork_checkpoint = build / "checkpoints" / "main.step-0000.json"
    _run_worker(root, build, "fork", fork_checkpoint)
    logger.log("[PASS] branch_fork_execution_completed")

    fork_report = read_json(build / "reports" / "fork.json")
    branches_body = {
        "schema_version": 1,
        "branches": [
            {
                "branch_id": "main",
                "parent_branch_id": None,
                "config_hash": config_hash,
            },
            {
                "branch_id": "replay",
                "parent_branch_id": "main",
                "parent_checkpoint_path": "checkpoints/main.bootstrap.json",
                "purpose": "historical_reconstruction",
            },
            {
                **fork_report["branch"],
                "config_hash": fork_report["fork_config_hash"],
            },
        ],
    }
    atomic_write_json(
        build / "ledgers" / "branches.json",
        {**branches_body, "branches_hash": hash_object(branches_body)},
    )

    performance = build_performance_report(build)
    audit = audit_all(root, build, config)
    evidence = generate_evidence(build, audit, run_identity)
    if not audit["overall_passed"] or evidence["overall_result"] != "PASS":
        failed = [name for name, result in audit["checks"].items() if not result["passed"]]
        logger.log(f"[FAIL] audit_failed checks={','.join(failed)}")
        raise RuntimeError(f"artifact audit failed: {failed}")
    logger.log(
        f"[PASS] audit_completed checks={len(audit['checks'])} hash={audit['audit_hash']}"
    )
    logger.log(
        f"[PASS] performance_measured utilization="
        f"{performance['derived']['packing_utilization']:.6f} "
        f"useful_tokens_per_second="
        f"{performance['derived']['useful_loss_tokens_per_second']:.2f}"
    )
    logger.log(f"[PASS] evidence_generated hash={evidence['evidence_hash']}")
    logger.log("[PASS] demo_completed")

    if final.exists():
        shutil.rmtree(final)
    os.replace(build, final)
    return final
