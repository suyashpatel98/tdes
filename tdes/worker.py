from __future__ import annotations

import argparse
import copy
import os
from pathlib import Path
from typing import Any

from .canonical import (
    atomic_write_json,
    hash_object,
    read_json,
    read_jsonl,
    semantic_code_hash,
    sha256_file,
)
from .checkpoint import load_checkpoint
from .engine import TrainingEngine, expectation_from_prepared
from .ledgers import RunLogger


EXPECTED_CRASH_CODE = 86


def _runtime(root: Path) -> tuple[dict[str, Any], str, str]:
    config = read_json(root / "configs" / "demo.json")
    return config, hash_object(config), semantic_code_hash(root)


def run_fresh(root: Path, artifacts: Path) -> None:
    config, config_hash, code_hash = _runtime(root)
    logger = RunLogger(artifacts / "run.log")
    branch_info = {
        "branch_id": "main",
        "parent_branch_id": None,
        "parent_checkpoint_hash": None,
        "fork_step": None,
        "config_delta": {},
    }
    engine = TrainingEngine(
        root,
        artifacts,
        config,
        config_hash,
        code_hash,
        "main",
        branch_info,
        logger,
    )
    engine.save_checkpoint("main.bootstrap.json", "bootstrap")
    logger.log("[PASS] training_started branch=main")
    crash_after = int(config["crash_after_step"])
    last_checkpoint = None
    while engine.next_step <= crash_after:
        prepared = engine.prepare_step()
        logger.log(
            f"[PASS] batches_packed step={engine.next_step} "
            f"candidates={len(prepared['candidates'])}"
        )
        dispositions = sorted(
            {decision["disposition"] for decision in prepared["opus"]["decisions"]}
        )
        logger.log(
            f"[PASS] opus_decisions_recorded step={engine.next_step} "
            f"dispositions={','.join(dispositions)}"
        )
        last_checkpoint = engine.commit_step(
            prepared, f"main.step-{engine.next_step:04d}.json"
        )["checkpoint"]
    if last_checkpoint is None:
        raise RuntimeError("crash worker produced no checkpoint")
    checkpoint_path = Path(last_checkpoint["path"])
    preview, checkpoint = TrainingEngine.from_checkpoint(
        root,
        artifacts,
        checkpoint_path,
        config,
        config_hash,
        code_hash,
        logger,
    )
    expected_prepared = preview.prepare_step()
    expectation = expectation_from_prepared(
        checkpoint["checkpoint_hash"], expected_prepared
    )
    atomic_write_json(artifacts / "reports" / "crash_expectation.json", expectation)
    logger.log(
        f"[PASS] crash_simulated exit_code={EXPECTED_CRASH_CODE} "
        f"checkpoint={checkpoint_path.name} expected_batch={expectation['batch_id']}"
    )
    os._exit(EXPECTED_CRASH_CODE)


def run_resume(root: Path, artifacts: Path, checkpoint_path: Path) -> None:
    config, config_hash, code_hash = _runtime(root)
    logger = RunLogger(artifacts / "run.log")
    engine, checkpoint = TrainingEngine.from_checkpoint(
        root,
        artifacts,
        checkpoint_path,
        config,
        config_hash,
        code_hash,
        logger,
    )
    engine.ledgers["events"].append(
        {
            "schema_version": 1,
            "branch_id": "main",
            "event": "crash_observed_and_resume_started",
            "step": engine.next_step,
            "checkpoint_hash": checkpoint["checkpoint_hash"],
        }
    )
    logger.log(
        f"[PASS] run_resumed checkpoint={checkpoint_path.name} next_step={engine.next_step}"
    )
    expectation = read_json(artifacts / "reports" / "crash_expectation.json")
    expectation_body = {
        key: value for key, value in expectation.items() if key != "expectation_hash"
    }
    if hash_object(expectation_body) != expectation["expectation_hash"]:
        raise RuntimeError("crash expectation hash mismatch")
    prepared = engine.prepare_step()
    actual = expectation_from_prepared(checkpoint["checkpoint_hash"], prepared)
    if actual != expectation:
        raise RuntimeError(
            f"resume next batch mismatch: expected={expectation['batch_id']} "
            f"actual={actual['batch_id']}"
        )
    prior_rows = engine.ledgers["consumption"].rows
    if prior_rows and prior_rows[-1]["batch_id"] == actual["batch_id"]:
        raise RuntimeError("resume repeated the preceding batch")
    engine.ledgers["events"].append(
        {
            "schema_version": 1,
            "branch_id": "main",
            "event": "resume_next_batch_matched",
            "step": engine.next_step,
            "batch_id": actual["batch_id"],
            "batch_hash": actual["batch_hash"],
            "expectation_hash": actual["expectation_hash"],
        }
    )
    logger.log(
        f"[PASS] resume_next_batch_matched step={engine.next_step} "
        f"batch_id={actual['batch_id']} hash={actual['batch_hash']}"
    )
    engine.commit_step(prepared, f"main.step-{engine.next_step:04d}.json")
    while engine.next_step < config["total_steps"]:
        prepared = engine.prepare_step()
        engine.commit_step(prepared, f"main.step-{engine.next_step:04d}.json")
    logger.log(f"[PASS] main_training_completed steps={engine.next_step}")


def run_replay(root: Path, artifacts: Path, checkpoint_path: Path) -> None:
    config, config_hash, code_hash = _runtime(root)
    logger = RunLogger(artifacts / "run.log")
    checkpoint_document = load_checkpoint(checkpoint_path)
    branch_info = {
        "branch_id": "replay",
        "parent_branch_id": "main",
        "parent_checkpoint_hash": checkpoint_document["checkpoint_hash"],
        "fork_step": checkpoint_document["payload"]["next_step"],
        "config_delta": {},
        "purpose": "historical_reconstruction",
    }
    engine, _ = TrainingEngine.from_checkpoint(
        root,
        artifacts,
        checkpoint_path,
        config,
        config_hash,
        code_hash,
        logger,
        branch_override="replay",
        branch_info_override=branch_info,
        attach_parent_ledgers=False,
    )
    originals = {
        row["step"]: row
        for row in read_jsonl(artifacts / "ledgers" / "main.consumption.jsonl")
    }
    comparisons = []
    replay_steps = min(4, config["total_steps"] - engine.next_step)
    for _ in range(replay_steps):
        prepared = engine.prepare_step()
        original = originals[prepared["step"]]
        actual = prepared["batch"]
        checks = {
            "batch_id": actual["batch_id"] == original["batch_id"],
            "batch_hash": actual["batch_hash"] == original["batch_hash"],
            "candidate_ids": actual["candidate_ids"] == original["candidate_ids"],
            "candidate_hashes": actual["candidate_hashes"]
            == original["candidate_hashes"],
            "ordered_source_spans": actual["ordered_source_spans"]
            == original["ordered_source_spans"],
        }
        if not all(checks.values()):
            raise RuntimeError(f"historical replay mismatch at step {prepared['step']}: {checks}")
        comparisons.append(
            {
                "step": prepared["step"],
                "original_batch_id": original["batch_id"],
                "replay_batch_id": actual["batch_id"],
                "batch_hash": actual["batch_hash"],
                "source_span_hashes": [
                    span["span_hash"] for span in actual["ordered_source_spans"]
                ],
                "checks": checks,
            }
        )
        engine.commit_step(prepared)
    report_body = {
        "schema_version": 1,
        "checkpoint_path": str(checkpoint_path.relative_to(artifacts)),
        "checkpoint_hash": checkpoint_document["checkpoint_hash"],
        "interval": [comparisons[0]["step"], comparisons[-1]["step"]],
        "comparisons": comparisons,
        "all_matched": all(all(item["checks"].values()) for item in comparisons),
    }
    atomic_write_json(
        artifacts / "reports" / "replay.json",
        {**report_body, "report_hash": hash_object(report_body)},
    )
    engine.save_checkpoint("replay.final.json", "replay_complete")
    logger.log(
        f"[PASS] replay_hash_matched interval={report_body['interval']} "
        f"batches={len(comparisons)}"
    )


def run_fork(root: Path, artifacts: Path, checkpoint_path: Path) -> None:
    config, _, code_hash = _runtime(root)
    logger = RunLogger(artifacts / "run.log")
    checkpoint_document = load_checkpoint(checkpoint_path)
    fork_config = copy.deepcopy(config)
    old_temperature = fork_config["opus_temperature"]
    fork_config["opus_temperature"] = round(old_temperature * 1.75, 6)
    old_weights = copy.deepcopy(fork_config["stages"][0]["lane_weights"])
    fork_config["stages"][0]["lane_weights"] = {
        "general": 0.35,
        "instruction": 0.25,
        "code": 0.40,
    }
    fork_hash = hash_object(fork_config)
    config_delta = {
        "opus_temperature": {
            "from": old_temperature,
            "to": fork_config["opus_temperature"],
        },
        "stages.0.lane_weights": {
            "from": old_weights,
            "to": fork_config["stages"][0]["lane_weights"],
        },
    }
    parent_ledger = artifacts / "ledgers" / "main.consumption.jsonl"
    parent_hash_before = sha256_file(parent_ledger)
    branch_info = {
        "branch_id": "fork-temperature-mixture",
        "parent_branch_id": "main",
        "parent_checkpoint_path": str(checkpoint_path.relative_to(artifacts)),
        "parent_checkpoint_hash": checkpoint_document["checkpoint_hash"],
        "fork_step": checkpoint_document["payload"]["next_step"],
        "parent_ledger_states": checkpoint_document["payload"]["ledger_states"],
        "config_delta": config_delta,
    }
    engine, _ = TrainingEngine.from_checkpoint(
        root,
        artifacts,
        checkpoint_path,
        fork_config,
        fork_hash,
        code_hash,
        logger,
        branch_override="fork-temperature-mixture",
        branch_info_override=branch_info,
        attach_parent_ledgers=False,
    )
    engine.save_checkpoint("fork.bootstrap.json", "branch_forked")
    fork_batches = []
    for _ in range(2):
        prepared = engine.prepare_step()
        fork_batches.append(
            {
                "step": prepared["step"],
                "batch_id": prepared["batch"]["batch_id"],
                "batch_hash": prepared["batch"]["batch_hash"],
            }
        )
        engine.commit_step(
            prepared, f"fork.step-{engine.next_step:04d}.json"
        )
    parent_hash_after = sha256_file(parent_ledger)
    if parent_hash_before != parent_hash_after:
        raise RuntimeError("fork mutated the parent consumption ledger")
    main_rows = {
        row["step"]: row
        for row in read_jsonl(artifacts / "ledgers" / "main.consumption.jsonl")
    }
    diverged = any(
        item["batch_hash"] != main_rows[item["step"]]["batch_hash"] for item in fork_batches
    )
    report_body = {
        "schema_version": 1,
        "branch": branch_info,
        "fork_config_hash": fork_hash,
        "fork_batches": fork_batches,
        "parent_consumption_hash_before": parent_hash_before,
        "parent_consumption_hash_after": parent_hash_after,
        "parent_unchanged": parent_hash_before == parent_hash_after,
        "stream_diverged": diverged,
    }
    atomic_write_json(
        artifacts / "reports" / "fork.json",
        {**report_body, "report_hash": hash_object(report_body)},
    )
    logger.log(
        f"[PASS] branch_forked branch=fork-temperature-mixture "
        f"parent=main fork_step={branch_info['fork_step']} diverged={diverged}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("fresh", "resume", "replay", "fork"))
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--artifacts", required=True, type=Path)
    parser.add_argument("--checkpoint", type=Path)
    args = parser.parse_args()
    if args.mode == "fresh":
        run_fresh(args.root, args.artifacts)
    elif args.mode == "resume":
        if args.checkpoint is None:
            parser.error("resume requires --checkpoint")
        run_resume(args.root, args.artifacts, args.checkpoint)
    elif args.mode == "replay":
        if args.checkpoint is None:
            parser.error("replay requires --checkpoint")
        run_replay(args.root, args.artifacts, args.checkpoint)
    else:
        if args.checkpoint is None:
            parser.error("fork requires --checkpoint")
        run_fork(args.root, args.artifacts, args.checkpoint)


if __name__ == "__main__":
    main()
