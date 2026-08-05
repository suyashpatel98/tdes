from __future__ import annotations

import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable

from .canonical import (
    atomic_write_json,
    hash_object,
    read_json,
    read_jsonl,
    sha256_file,
)
from .checkpoint import load_checkpoint
from .corpus import build_documents
from .ledgers import HashLedger
from .packing import validate_packed
from .shards import ShardRepository
from .tokenizer import FrozenByteTokenizer


def build_performance_report(artifacts: Path) -> dict[str, Any]:
    rows = read_jsonl(artifacts / "ledgers" / "main.consumption.jsonl")
    capacity = sum(row["capacity_tokens"] for row in rows)
    non_padding = sum(row["non_padding_tokens"] for row in rows)
    useful = sum(row["loss_bearing_tokens"] for row in rows)
    packing_ns = sum(row["timing_ns"]["packing"] for row in rows)
    opus_ns = sum(row["timing_ns"]["opus"] for row in rows)
    training_ns = sum(row["timing_ns"]["training"] for row in rows)
    end_to_end_ns = sum(
        row["timing_ns"]["prepare_total"] + row["timing_ns"]["training"]
        for row in rows
    )
    stages: dict[str, dict[str, int]] = defaultdict(
        lambda: {"steps": 0, "capacity_tokens": 0, "non_padding_tokens": 0, "loss_bearing_tokens": 0}
    )
    for row in rows:
        stage = stages[row["stage"]]
        stage["steps"] += 1
        stage["capacity_tokens"] += row["capacity_tokens"]
        stage["non_padding_tokens"] += row["non_padding_tokens"]
        stage["loss_bearing_tokens"] += row["loss_bearing_tokens"]
    raw = {
        "steps": len(rows),
        "capacity_tokens": capacity,
        "non_padding_tokens": non_padding,
        "loss_bearing_tokens": useful,
        "duration_ns": {
            "packing": packing_ns,
            "opus": opus_ns,
            "training": training_ns,
            "end_to_end": end_to_end_ns,
        },
        "stages": dict(stages),
    }
    derived = {
        "packing_utilization": non_padding / capacity,
        "useful_token_ratio": useful / capacity,
        "packed_tokens_per_second": non_padding / (packing_ns / 1e9),
        "useful_loss_tokens_per_second": useful / (end_to_end_ns / 1e9),
        "training_loss_tokens_per_second": useful / (training_ns / 1e9),
        "opus_fraction_of_end_to_end": opus_ns / end_to_end_ns,
    }
    body = {"schema_version": 1, "raw": raw, "derived": derived}
    report = {**body, "performance_hash": hash_object(body)}
    atomic_write_json(artifacts / "performance.json", report)
    return report


def _result(passed: bool, evidence: list[str], details: dict[str, Any]) -> dict[str, Any]:
    return {"passed": bool(passed), "evidence": evidence, "details": details}


def _capture(
    function: Callable[[], dict[str, Any]], evidence: list[str]
) -> dict[str, Any]:
    try:
        result = function()
        return _result(True, evidence, result)
    except Exception as error:
        return _result(False, evidence, {"error": f"{type(error).__name__}: {error}"})


def audit_all(root: Path, artifacts: Path, config: dict[str, Any]) -> dict[str, Any]:
    tokenizer = FrozenByteTokenizer.create()

    def tokenizer_and_shards() -> dict[str, Any]:
        artifact = read_json(artifacts / "manifests" / "tokenizer.json")
        if artifact["tokenizer_hash"] != tokenizer.tokenizer_hash:
            raise ValueError("tokenizer artifact hash mismatch")
        regenerated, source_report = build_documents(root, config)
        generated = read_jsonl(artifacts / "reports" / "documents.jsonl")
        if regenerated != generated:
            raise ValueError("generated documents do not reproduce from raw sources")
        repository = ShardRepository(artifacts, tokenizer)
        document_by_id = {row["record_id"]: row for row in generated}
        retokenized = 0
        for records in repository.records_by_shard.values():
            for record in records:
                document = document_by_id[record["record_id"]]
                expected = {
                    key: tokenizer.encode(value) for key, value in document["fields"].items()
                }
                if expected != record["token_fields"]:
                    raise ValueError(f"retokenization mismatch: {record['record_id']}")
                retokenized += 1
        return {
            "tokenizer_hash": tokenizer.tokenizer_hash,
            "root_manifest_hash": repository.root["root_manifest_hash"],
            "manifest_count": len(repository.manifests),
            "retokenized_records": retokenized,
            "source_manifest_hash": source_report["source_manifest_hash"],
        }

    def firewall() -> dict[str, Any]:
        report = read_json(artifacts / "reports" / "firewall.json")
        if not report["all_blocked"]:
            raise ValueError("one or more firewall probes were admitted")
        repository = ShardRepository(artifacts, tokenizer)
        roles = {manifest["shard_id"]: manifest["role"] for manifest in repository.manifests}
        consumed = read_jsonl(artifacts / "ledgers" / "main.consumption.jsonl")
        consumed_shards = {
            span["shard_id"] for row in consumed for span in row["ordered_source_spans"]
        }
        invalid = {shard: roles[shard] for shard in consumed_shards if roles[shard] != "train"}
        if invalid:
            raise ValueError(f"non-train shards consumed: {invalid}")
        opus_rows = read_jsonl(artifacts / "ledgers" / "main.opus.jsonl")
        proxy_ids = {item for row in opus_rows for item in row["proxy_candidate_ids"]}
        consumed_ids = {item for row in consumed for item in row["candidate_ids"]}
        if proxy_ids.intersection(consumed_ids):
            raise ValueError("proxy candidate entered a loss-bearing batch")
        return {
            "blocked_roles": sorted(item["role"] for item in report["attempts"]),
            "consumed_shards": sorted(consumed_shards),
            "proxy_candidate_count": len(proxy_ids),
            "proxy_consumption_intersection": [],
        }

    def packing() -> dict[str, Any]:
        batch_paths = sorted((artifacts / "batches").glob("main.step-*.json"))
        checked_candidates = 0
        checked_spans = 0
        multi_segment = 0
        for path in batch_paths:
            batch = read_json(path)
            for candidate in batch["selected_candidates"]:
                validate_packed(candidate, tokenizer)
                semantic_keys = (
                    "schema_version",
                    "stream",
                    "step_created",
                    "slot_created",
                    "lane",
                    "sequence_length",
                    "tokenizer_hash",
                    "input_ids",
                    "labels",
                    "loss_mask",
                    "attention_mask",
                    "position_ids",
                    "segment_ids",
                    "source_spans",
                    "non_padding_tokens",
                    "loss_bearing_tokens",
                )
                if hash_object({key: candidate[key] for key in semantic_keys}) != candidate[
                    "packed_hash"
                ]:
                    raise ValueError(f"packed hash mismatch: {candidate['candidate_id']}")
                checked_candidates += 1
                checked_spans += len(candidate["source_spans"])
                multi_segment += int(len(candidate["source_spans"]) > 1)
            semantic = {
                key: batch[key]
                for key in (
                    "schema_version",
                    "step",
                    "candidate_ids",
                    "candidate_hashes",
                    "ordered_source_spans",
                    "input_hash",
                    "tokenizer_hash",
                    "root_manifest_hash",
                    "opus_result_hash",
                )
            }
            if hash_object(semantic) != batch["batch_hash"]:
                raise ValueError(f"batch semantic hash mismatch: {path}")
        if len(batch_paths) != config["total_steps"] or multi_segment < 1:
            raise ValueError("missing main batches or no multi-segment packing evidence")
        return {
            "batch_count": len(batch_paths),
            "checked_candidates": checked_candidates,
            "checked_source_spans": checked_spans,
            "multi_segment_candidates": multi_segment,
        }

    def mixture() -> dict[str, Any]:
        plan = read_json(artifacts / "reports" / "mixture_plan.json")
        plan_body = {key: value for key, value in plan.items() if key != "mixture_plan_hash"}
        if hash_object(plan_body) != plan["mixture_plan_hash"]:
            raise ValueError("mixture plan hash mismatch")
        opus_rows = [
            row
            for row in read_jsonl(artifacts / "ledgers" / "main.opus.jsonl")
            if row["branch_id"] == "main"
        ]
        consumption = read_jsonl(artifacts / "ledgers" / "main.consumption.jsonl")
        actual_shares: dict[str, int] = Counter()
        for step_plan in plan["steps"]:
            step = step_plan["step"]
            candidates = Counter(row["lane"] for row in opus_rows if row["step"] == step)
            if dict(candidates) != {
                key: value for key, value in step_plan["candidate_quotas"].items() if value
            }:
                raise ValueError(f"candidate quota mismatch at step {step}")
            row = next(item for item in consumption if item["step"] == step)
            for lane, required in step_plan["required_selected"].items():
                if row["selected_lane_counts"][lane] < required:
                    raise ValueError(f"protected floor missed at step {step}")
            actual_shares.update(row["selected_lane_counts"])
        return {
            "mixture_plan_hash": plan["mixture_plan_hash"],
            "steps_checked": len(plan["steps"]),
            "actual_selected_counts": dict(actual_shares),
            "protected_floor_misses": 0,
        }

    def opus() -> dict[str, Any]:
        rows = read_jsonl(artifacts / "ledgers" / "main.opus.jsonl")
        dispositions = Counter(row["disposition"] for row in rows)
        required = {"accepted", "deferred", "rejected", "protected_floor_override"}
        if not required.issubset(dispositions):
            raise ValueError(f"missing OPUS dispositions: {required - set(dispositions)}")
        for row in rows:
            decision_body = {
                key: value
                for key, value in row.items()
                if key
                not in {
                    "ledger",
                    "ledger_offset",
                    "previous_row_hash",
                    "row_hash",
                    "branch_id",
                    "opus_result_hash",
                    "decision_hash",
                }
            }
            if hash_object(decision_body) != row["decision_hash"]:
                raise ValueError(f"OPUS decision hash mismatch at row {row['ledger_offset']}")
        probability_groups: dict[tuple[int, int], list[float]] = defaultdict(list)
        for row in rows:
            for trace in row["traces"]:
                probability_groups[(row["step"], trace["selection_rank"])].append(
                    trace["probability"]
                )
        for key, probabilities in probability_groups.items():
            if not math.isclose(sum(probabilities), 1.0, rel_tol=1e-9, abs_tol=1e-9):
                raise ValueError(f"Boltzmann probabilities do not sum to one: {key}")
        overrides = [row for row in rows if row["protected_floor_event"] is not None]
        if not overrides:
            raise ValueError("no protected-floor override record")
        return {
            "decision_count": len(rows),
            "dispositions": dict(dispositions),
            "probability_groups_checked": len(probability_groups),
            "override_decision_rows": len(overrides),
        }

    def ledgers_and_learning() -> dict[str, Any]:
        ledger_counts = {}
        for name in ("consumption", "learning", "opus", "events"):
            ledger = HashLedger(artifacts / "ledgers" / f"main.{name}.jsonl", name)
            ledger_counts[name] = len(ledger.rows)
        consumption = read_jsonl(artifacts / "ledgers" / "main.consumption.jsonl")
        learning = read_jsonl(artifacts / "ledgers" / "main.learning.jsonl")
        if [row["step"] for row in consumption] != list(range(config["total_steps"])):
            raise ValueError("main consumption steps are not contiguous")
        if len({row["batch_id"] for row in consumption}) != len(consumption):
            raise ValueError("duplicate committed batch ID")
        for consumed in consumption:
            expected_global_batch = (
                consumed["microbatch_size"]
                * consumed["world_size"]
                * consumed["gradient_accumulation_steps"]
            )
            if consumed["global_batch_size"] != expected_global_batch:
                raise ValueError(
                    f"global batch formula mismatch: {consumed['batch_id']}"
                )
            linked = [row for row in learning if row["batch_id"] == consumed["batch_id"]]
            if not linked or any(
                row["consumption_row_hash"] != consumed["row_hash"] for row in linked
            ):
                raise ValueError(f"broken learning link: {consumed['batch_id']}")
            if sum(row["loss_bearing_tokens"] for row in linked) != consumed[
                "loss_bearing_tokens"
            ]:
                raise ValueError(f"learning token mismatch: {consumed['batch_id']}")
            if not math.isclose(
                sum(row["loss_sum"] for row in linked),
                consumed["loss_sum"],
                rel_tol=1e-10,
                abs_tol=1e-10,
            ):
                raise ValueError(f"learning loss mismatch: {consumed['batch_id']}")
            consumed_span_hashes = {
                span["span_hash"] for span in consumed["ordered_source_spans"]
            }
            if {row["span_hash"] for row in linked} != consumed_span_hashes:
                raise ValueError(f"learning/source span mismatch: {consumed['batch_id']}")
        return {
            "ledger_rows": ledger_counts,
            "consumption_steps": [row["step"] for row in consumption],
            "learning_records": len(learning),
            "linked_loss_tokens": sum(row["loss_bearing_tokens"] for row in learning),
        }

    def checkpoint_resume() -> dict[str, Any]:
        crash_step = config["crash_after_step"]
        checkpoint_path = artifacts / "checkpoints" / f"main.step-{crash_step:04d}.json"
        checkpoint = load_checkpoint(checkpoint_path)
        payload = checkpoint["payload"]
        if payload["next_step"] != crash_step + 1:
            raise ValueError("crash checkpoint next step mismatch")
        for name, expected in payload["ledger_states"].items():
            final_rows = read_jsonl(artifacts / "ledgers" / f"main.{name}.jsonl")
            if len(final_rows) < expected["rows"]:
                raise ValueError(f"final ledger shorter than crash checkpoint: {name}")
            tail = final_rows[expected["rows"] - 1]["row_hash"] if expected["rows"] else "0" * 64
            if tail != expected["tail_hash"]:
                raise ValueError(f"checkpoint ledger prefix mismatch: {name}")
        expectation = read_json(artifacts / "reports" / "crash_expectation.json")
        body = {key: value for key, value in expectation.items() if key != "expectation_hash"}
        if hash_object(body) != expectation["expectation_hash"]:
            raise ValueError("crash expectation hash mismatch")
        consumption = read_jsonl(artifacts / "ledgers" / "main.consumption.jsonl")
        resumed = next(row for row in consumption if row["step"] == payload["next_step"])
        if resumed["batch_id"] != expectation["batch_id"] or resumed["batch_hash"] != expectation[
            "batch_hash"
        ]:
            raise ValueError("resumed batch does not match expectation")
        if resumed["candidate_ids"] != expectation["candidate_ids"]:
            raise ValueError("resumed candidate IDs do not match expectation")
        if resumed["ordered_source_spans"] != expectation["ordered_source_spans"]:
            raise ValueError("resumed source spans do not match expectation")
        if consumption[crash_step]["batch_id"] == resumed["batch_id"]:
            raise ValueError("resume repeated previous batch")
        return {
            "checkpoint_hash": checkpoint["checkpoint_hash"],
            "checkpoint_next_step": payload["next_step"],
            "expected_batch_id": expectation["batch_id"],
            "resumed_batch_id": resumed["batch_id"],
            "ledger_prefixes_checked": len(payload["ledger_states"]),
            "skipped_or_repeated_batches": 0,
        }

    def replay() -> dict[str, Any]:
        report = read_json(artifacts / "reports" / "replay.json")
        body = {key: value for key, value in report.items() if key != "report_hash"}
        if hash_object(body) != report["report_hash"] or not report["all_matched"]:
            raise ValueError("replay report failed integrity or equality")
        if not all(all(item["checks"].values()) for item in report["comparisons"]):
            raise ValueError("replay comparison contains a mismatch")
        for name in ("consumption", "learning", "opus", "events"):
            HashLedger(artifacts / "ledgers" / f"replay.{name}.jsonl", name)
        return {
            "checkpoint_hash": report["checkpoint_hash"],
            "interval": report["interval"],
            "matched_batches": len(report["comparisons"]),
            "matched_span_hashes": sum(
                len(item["source_span_hashes"]) for item in report["comparisons"]
            ),
        }

    def fork() -> dict[str, Any]:
        report = read_json(artifacts / "reports" / "fork.json")
        body = {key: value for key, value in report.items() if key != "report_hash"}
        if hash_object(body) != report["report_hash"]:
            raise ValueError("fork report hash mismatch")
        if not report["parent_unchanged"] or not report["stream_diverged"]:
            raise ValueError("fork did not preserve parent or explicitly diverge")
        if not report["branch"]["config_delta"]:
            raise ValueError("fork configuration delta is missing")
        for name in ("consumption", "learning", "opus", "events"):
            HashLedger(
                artifacts / "ledgers" / f"fork-temperature-mixture.{name}.jsonl", name
            )
        return {
            "parent_checkpoint_hash": report["branch"]["parent_checkpoint_hash"],
            "fork_step": report["branch"]["fork_step"],
            "config_delta": report["branch"]["config_delta"],
            "fork_batches": report["fork_batches"],
            "parent_unchanged": report["parent_unchanged"],
            "stream_diverged": report["stream_diverged"],
        }

    def throughput() -> dict[str, Any]:
        report = read_json(artifacts / "performance.json")
        body = {key: value for key, value in report.items() if key != "performance_hash"}
        if hash_object(body) != report["performance_hash"]:
            raise ValueError("performance report hash mismatch")
        rows = read_jsonl(artifacts / "ledgers" / "main.consumption.jsonl")
        raw = report["raw"]
        if raw["capacity_tokens"] != sum(row["capacity_tokens"] for row in rows):
            raise ValueError("performance capacity cannot be reconstructed")
        if raw["non_padding_tokens"] != sum(row["non_padding_tokens"] for row in rows):
            raise ValueError("performance packed tokens cannot be reconstructed")
        if raw["loss_bearing_tokens"] != sum(row["loss_bearing_tokens"] for row in rows):
            raise ValueError("performance useful tokens cannot be reconstructed")
        derived = report["derived"]
        if not 0 <= derived["packing_utilization"] <= 1:
            raise ValueError("invalid packing utilization")
        if any(value <= 0 for key, value in derived.items() if "per_second" in key):
            raise ValueError("non-positive measured throughput")
        return {
            "performance_hash": report["performance_hash"],
            "raw": raw,
            "derived": derived,
        }

    checks = {
        "tokenizer_integrity": _capture(
            tokenizer_and_shards,
            ["manifests/tokenizer.json", "manifests/root.json", "reports/source_report.json"],
        ),
        "evaluation_firewall": _capture(
            firewall,
            ["reports/firewall.json", "ledgers/main.consumption.jsonl", "ledgers/main.opus.jsonl"],
        ),
        "packing_correctness": _capture(
            packing, ["batches/", "ledgers/main.consumption.jsonl"]
        ),
        "mixture_compliance": _capture(
            mixture, ["reports/mixture_plan.json", "ledgers/main.consumption.jsonl"]
        ),
        "opus_audit_trail": _capture(
            opus, ["ledgers/main.opus.jsonl", "reports/mixture_plan.json"]
        ),
        "learning_trace": _capture(
            ledgers_and_learning,
            ["ledgers/main.consumption.jsonl", "ledgers/main.learning.jsonl"],
        ),
        "crash_recovery": _capture(
            checkpoint_resume,
            [
                f"checkpoints/main.step-{config['crash_after_step']:04d}.json",
                "reports/crash_expectation.json",
                "ledgers/main.consumption.jsonl",
            ],
        ),
        "replay": _capture(
            replay, ["reports/replay.json", "ledgers/replay.consumption.jsonl"]
        ),
        "fork": _capture(
            fork,
            ["reports/fork.json", "checkpoints/fork.bootstrap.json", "ledgers/fork-temperature-mixture.consumption.jsonl"],
        ),
        "throughput": _capture(
            throughput, ["performance.json", "ledgers/main.consumption.jsonl"]
        ),
    }
    report_body = {
        "schema_version": 1,
        "checks": checks,
        "overall_passed": all(result["passed"] for result in checks.values()),
    }
    report = {**report_body, "audit_hash": hash_object(report_body)}
    atomic_write_json(artifacts / "reports" / "audit.json", report)
    return report
