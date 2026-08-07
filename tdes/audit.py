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
        if hash_object(artifact["spec"]) != artifact["tokenizer_hash"]:
            raise ValueError("tokenizer artifact does not hash its own specification")
        if artifact["tokenizer_hash"] != tokenizer.tokenizer_hash:
            raise ValueError("tokenizer artifact hash mismatch")
        regenerated, source_report = build_documents(root, config)
        generated = read_jsonl(artifacts / "reports" / "documents.jsonl")
        if regenerated != generated:
            raise ValueError("generated documents do not reproduce from raw sources")
        if read_json(artifacts / "reports" / "source_report.json") != source_report:
            raise ValueError("persisted source report does not reproduce from raw sources")
        repository = ShardRepository(artifacts, tokenizer)
        document_by_id = {row["record_id"]: row for row in generated}
        record_ids = {
            record["record_id"]
            for records in repository.records_by_shard.values()
            for record in records
        }
        if record_ids != set(document_by_id):
            raise ValueError("tokenized record IDs do not match cleaned document IDs")
        retokenized = 0
        for records in repository.records_by_shard.values():
            for record in records:
                document = document_by_id[record["record_id"]]
                expected = {
                    key: tokenizer.encode(value) for key, value in document["fields"].items()
                }
                if expected != record["token_fields"]:
                    raise ValueError(f"retokenization mismatch: {record['record_id']}")
                metadata_fields = {
                    "role": "role",
                    "lane": "lane",
                    "data_type": "data_type",
                    "source_content_hash": "content_hash",
                    "source_id": "source_id",
                    "source_index": "source_index",
                    "metadata": "metadata",
                }
                for record_field, document_field in metadata_fields.items():
                    if record[record_field] != document[document_field]:
                        raise ValueError(
                            f"tokenized metadata mismatch for {record_field}: "
                            f"{record['record_id']}"
                        )
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
        report_body = {
            key: value for key, value in report.items() if key != "firewall_report_hash"
        }
        if hash_object(report_body) != report["firewall_report_hash"]:
            raise ValueError("firewall report hash mismatch")
        if not report["all_blocked"]:
            raise ValueError("one or more firewall probes were admitted")
        attempts = {item["role"]: item for item in report["attempts"]}
        if set(attempts) != {"eval", "validation", "proxy"} or any(
            not item["blocked"] or item["requested_use"] != "train"
            for item in attempts.values()
        ):
            raise ValueError("firewall probes do not cover every non-training role")
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
        consumption_by_step = {row["step"]: row for row in consumption}
        plan_steps = {row["step"] for row in plan["steps"]}
        if {row["step"] for row in opus_rows} != plan_steps:
            raise ValueError("OPUS decision steps do not match mixture plan")
        if set(consumption_by_step) != plan_steps:
            raise ValueError("consumption steps do not match mixture plan")

        stage_counters: dict[str, dict[str, Any]] = {}
        protected_floor_misses: list[dict[str, Any]] = []
        for step_plan in plan["steps"]:
            step = step_plan["step"]
            candidates = Counter(row["lane"] for row in opus_rows if row["step"] == step)
            if dict(candidates) != {
                key: value for key, value in step_plan["candidate_quotas"].items() if value
            }:
                raise ValueError(f"candidate quota mismatch at step {step}")
            row = consumption_by_step[step]
            if row["stage"] != step_plan["stage"]:
                raise ValueError(f"curriculum stage mismatch at step {step}")
            for lane, required in step_plan["required_selected"].items():
                if row["selected_lane_counts"][lane] < required:
                    protected_floor_misses.append(
                        {
                            "step": step,
                            "lane": lane,
                            "required": required,
                            "actual": row["selected_lane_counts"][lane],
                        }
                    )

            stage = stage_counters.setdefault(
                step_plan["stage"],
                {
                    "steps": 0,
                    "planned_weights": step_plan["lane_weights"],
                    "planned_candidates": Counter(),
                    "actual_candidates": Counter(),
                    "selected_candidates": Counter(),
                    "selected_loss_bearing_tokens": Counter(),
                },
            )
            if stage["planned_weights"] != step_plan["lane_weights"]:
                raise ValueError(f"lane weights changed inside stage {step_plan['stage']}")
            stage["steps"] += 1
            stage["planned_candidates"].update(step_plan["candidate_quotas"])
            stage["actual_candidates"].update(candidates)
            stage["selected_candidates"].update(row["selected_lane_counts"])
            stage["selected_loss_bearing_tokens"].update(
                {
                    lane: sum(
                        span["loss_bearing_tokens"]
                        for span in row["ordered_source_spans"]
                        if span["lane"] == lane
                    )
                    for lane in step_plan["lane_weights"]
                }
            )

        if protected_floor_misses:
            raise ValueError(f"protected floor misses: {protected_floor_misses}")

        def shares(counts: dict[str, int]) -> dict[str, float]:
            total = sum(counts.values())
            return {lane: value / total for lane, value in sorted(counts.items())}

        stage_reports: dict[str, dict[str, Any]] = {}
        aggregate = {
            "planned_candidates": Counter(),
            "actual_candidates": Counter(),
            "selected_candidates": Counter(),
            "selected_loss_bearing_tokens": Counter(),
        }
        for name, counters in stage_counters.items():
            planned = dict(sorted(counters["planned_candidates"].items()))
            actual = dict(sorted(counters["actual_candidates"].items()))
            selected = dict(sorted(counters["selected_candidates"].items()))
            selected_tokens = dict(
                sorted(counters["selected_loss_bearing_tokens"].items())
            )
            if actual != planned:
                raise ValueError(f"aggregate candidate mixture mismatch in stage {name}")
            stage_reports[name] = {
                "steps": counters["steps"],
                "planned_lane_weights": counters["planned_weights"],
                "planned_candidate_counts": planned,
                "planned_candidate_shares": shares(planned),
                "actual_candidate_counts": actual,
                "actual_candidate_shares": shares(actual),
                "selected_candidate_counts": selected,
                "selected_candidate_shares": shares(selected),
                "selected_loss_bearing_tokens": selected_tokens,
                "selected_loss_bearing_token_shares": shares(selected_tokens),
            }
            for key in aggregate:
                aggregate[key].update(counters[key])

        planned = dict(sorted(aggregate["planned_candidates"].items()))
        actual = dict(sorted(aggregate["actual_candidates"].items()))
        selected = dict(sorted(aggregate["selected_candidates"].items()))
        selected_tokens = dict(
            sorted(aggregate["selected_loss_bearing_tokens"].items())
        )
        aggregate_report = {
            "planned_candidate_counts": planned,
            "planned_candidate_shares": shares(planned),
            "actual_candidate_counts": actual,
            "actual_candidate_shares": shares(actual),
            "selected_candidate_counts": selected,
            "selected_candidate_shares": shares(selected),
            "selected_loss_bearing_tokens": selected_tokens,
            "selected_loss_bearing_token_shares": shares(selected_tokens),
        }

        report_body = {
            "schema_version": 1,
            "mixture_plan_hash": plan["mixture_plan_hash"],
            "steps_checked": len(plan["steps"]),
            "stages": stage_reports,
            "aggregate": aggregate_report,
            "protected_floor_misses": 0,
        }
        report = {**report_body, "mixture_compliance_hash": hash_object(report_body)}
        atomic_write_json(artifacts / "reports" / "mixture_compliance.json", report)
        return report

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
        consumption = {
            row["step"]: row
            for row in read_jsonl(artifacts / "ledgers" / "main.consumption.jsonl")
        }
        for step, consumed in consumption.items():
            decisions = [row for row in rows if row["step"] == step]
            decision_by_id = {row["candidate_id"]: row for row in decisions}
            if len(decision_by_id) != len(decisions):
                raise ValueError(f"duplicate OPUS candidate decision at step {step}")
            selected_ids = {
                row["candidate_id"] for row in decisions if row["final_selected"]
            }
            if selected_ids != set(consumed["candidate_ids"]):
                raise ValueError(f"OPUS selections do not match batch at step {step}")
            for candidate_id, packed_hash in zip(
                consumed["candidate_ids"], consumed["candidate_hashes"]
            ):
                if decision_by_id[candidate_id]["packed_hash"] != packed_hash:
                    raise ValueError(f"OPUS packed hash does not match batch at step {step}")
            selected_span_hashes = Counter(
                span_hash
                for row in decisions
                if row["final_selected"]
                for span_hash in row["source_span_hashes"]
            )
            consumed_span_hashes = Counter(
                span["span_hash"] for span in consumed["ordered_source_spans"]
            )
            if selected_span_hashes != consumed_span_hashes:
                raise ValueError(f"OPUS source spans do not match batch at step {step}")
            batch = read_json(artifacts / consumed["batch_path"])
            result_hashes = {row["opus_result_hash"] for row in decisions}
            if result_hashes != {batch["opus_result_hash"]}:
                raise ValueError(f"OPUS result hash does not match batch at step {step}")
            if any(
                row["model_hash"] != consumed["model_before_hash"]
                or row["optimizer_hash"] != consumed["optimizer_before_hash"]
                or row["branch_id"] != "main"
                for row in decisions
            ):
                raise ValueError(f"OPUS state identity mismatch at step {step}")
            for row in decisions:
                selected_disposition = row["disposition"] in {
                    "accepted",
                    "protected_floor_override",
                }
                if selected_disposition != row["final_selected"]:
                    raise ValueError(
                        f"OPUS disposition/final selection mismatch at step {step}"
                    )
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
        root_manifest = read_json(artifacts / "manifests" / "root.json")
        if payload["root_manifest_hash"] != root_manifest["root_manifest_hash"]:
            raise ValueError("checkpoint root manifest does not match audited repository")
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
        replay_ledgers = {}
        for name in ("consumption", "learning", "opus", "events"):
            replay_ledgers[name] = HashLedger(
                artifacts / "ledgers" / f"replay.{name}.jsonl", name
            )
        main_by_step = {
            row["step"]: row
            for row in read_jsonl(artifacts / "ledgers" / "main.consumption.jsonl")
        }
        replay_rows = replay_ledgers["consumption"].rows
        comparison_by_step = {
            row["step"]: row for row in report["comparisons"]
        }
        if not replay_rows or set(comparison_by_step) != {
            row["step"] for row in replay_rows
        }:
            raise ValueError("replay report interval does not match replay ledger")
        if report["interval"] != [replay_rows[0]["step"], replay_rows[-1]["step"]]:
            raise ValueError("replay interval boundaries are incorrect")
        replay_keys = (
            "batch_id",
            "batch_hash",
            "candidate_ids",
            "candidate_hashes",
            "ordered_source_spans",
        )
        for replayed in replay_rows:
            original = main_by_step[replayed["step"]]
            if any(replayed[key] != original[key] for key in replay_keys):
                raise ValueError(f"replay ledger mismatch at step {replayed['step']}")
            comparison = comparison_by_step[replayed["step"]]
            if (
                comparison["original_batch_id"] != original["batch_id"]
                or comparison["replay_batch_id"] != replayed["batch_id"]
                or comparison["batch_hash"] != replayed["batch_hash"]
                or comparison["source_span_hashes"]
                != [span["span_hash"] for span in replayed["ordered_source_spans"]]
            ):
                raise ValueError(
                    f"replay report does not describe its ledger at step {replayed['step']}"
                )
        return {
            "checkpoint_hash": report["checkpoint_hash"],
            "interval": report["interval"],
            "matched_batches": len(replay_rows),
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
        fork_ledgers = {}
        for name in ("consumption", "learning", "opus", "events"):
            fork_ledgers[name] = HashLedger(
                artifacts / "ledgers" / f"fork-temperature-mixture.{name}.jsonl", name
            )
        parent_hash = sha256_file(artifacts / "ledgers" / "main.consumption.jsonl")
        if (
            report["parent_consumption_hash_before"] != parent_hash
            or report["parent_consumption_hash_after"] != parent_hash
        ):
            raise ValueError("fork report parent hashes do not match parent ledger")
        parent_checkpoint = load_checkpoint(
            artifacts / report["branch"]["parent_checkpoint_path"]
        )
        if parent_checkpoint["checkpoint_hash"] != report["branch"][
            "parent_checkpoint_hash"
        ]:
            raise ValueError("fork parent checkpoint identity mismatch")
        fork_rows = fork_ledgers["consumption"].rows
        reported_batches = [
            {
                "step": row["step"],
                "batch_id": row["batch_id"],
                "batch_hash": row["batch_hash"],
            }
            for row in fork_rows
        ]
        if reported_batches != report["fork_batches"]:
            raise ValueError("fork report does not match branch consumption ledger")
        main_by_step = {
            row["step"]: row
            for row in read_jsonl(artifacts / "ledgers" / "main.consumption.jsonl")
        }
        actual_divergence = any(
            row["batch_hash"] != main_by_step[row["step"]]["batch_hash"]
            for row in fork_rows
        )
        if not actual_divergence:
            raise ValueError("fork ledger did not diverge from parent stream")
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
        expected_stages: dict[str, dict[str, int]] = defaultdict(
            lambda: {
                "steps": 0,
                "capacity_tokens": 0,
                "non_padding_tokens": 0,
                "loss_bearing_tokens": 0,
            }
        )
        for row in rows:
            stage = expected_stages[row["stage"]]
            stage["steps"] += 1
            stage["capacity_tokens"] += row["capacity_tokens"]
            stage["non_padding_tokens"] += row["non_padding_tokens"]
            stage["loss_bearing_tokens"] += row["loss_bearing_tokens"]
        expected_raw = {
            "steps": len(rows),
            "capacity_tokens": sum(row["capacity_tokens"] for row in rows),
            "non_padding_tokens": sum(row["non_padding_tokens"] for row in rows),
            "loss_bearing_tokens": sum(row["loss_bearing_tokens"] for row in rows),
            "duration_ns": {
                "packing": sum(row["timing_ns"]["packing"] for row in rows),
                "opus": sum(row["timing_ns"]["opus"] for row in rows),
                "training": sum(row["timing_ns"]["training"] for row in rows),
                "end_to_end": sum(
                    row["timing_ns"]["prepare_total"]
                    + row["timing_ns"]["training"]
                    for row in rows
                ),
            },
            "stages": dict(expected_stages),
        }
        if raw != expected_raw:
            raise ValueError("performance raw measurements cannot be reconstructed")
        durations = expected_raw["duration_ns"]
        expected_derived = {
            "packing_utilization": expected_raw["non_padding_tokens"]
            / expected_raw["capacity_tokens"],
            "useful_token_ratio": expected_raw["loss_bearing_tokens"]
            / expected_raw["capacity_tokens"],
            "packed_tokens_per_second": expected_raw["non_padding_tokens"]
            / (durations["packing"] / 1e9),
            "useful_loss_tokens_per_second": expected_raw["loss_bearing_tokens"]
            / (durations["end_to_end"] / 1e9),
            "training_loss_tokens_per_second": expected_raw["loss_bearing_tokens"]
            / (durations["training"] / 1e9),
            "opus_fraction_of_end_to_end": durations["opus"]
            / durations["end_to_end"],
        }
        derived = report["derived"]
        if set(derived) != set(expected_derived) or any(
            not math.isclose(
                derived[key], expected, rel_tol=1e-12, abs_tol=1e-12
            )
            for key, expected in expected_derived.items()
        ):
            raise ValueError("performance derived metrics cannot be reconstructed")
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
            mixture,
            [
                "reports/mixture_plan.json",
                "reports/mixture_compliance.json",
                "ledgers/main.consumption.jsonl",
                "ledgers/main.opus.jsonl",
            ],
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
