from __future__ import annotations

import copy
import time
from pathlib import Path
from typing import Any

from .canonical import atomic_write_json, hash_object
from .checkpoint import load_checkpoint, write_checkpoint
from .ledgers import LedgerSet, RunLogger
from .mixture import Curriculum
from .model import TinyCausalLM
from .opus import OPUSSelector
from .packing import DeterministicPacker, validate_packed
from .shards import ShardRepository
from .tokenizer import FrozenByteTokenizer


class TrainingEngine:
    def __init__(
        self,
        root: Path,
        artifacts: Path,
        config: dict[str, Any],
        config_hash: str,
        code_hash: str,
        branch_id: str,
        branch_info: dict[str, Any],
        logger: RunLogger,
        state: dict[str, Any] | None = None,
        ledger_states: dict[str, dict[str, Any]] | None = None,
        attach_existing_ledgers: bool = False,
    ):
        self.root = root
        self.artifacts = artifacts
        self.config = config
        self.config_hash = config_hash
        self.code_hash = code_hash
        self.branch_id = branch_id
        self.branch_info = branch_info
        self.logger = logger
        self.tokenizer = FrozenByteTokenizer.create()
        self.repository = ShardRepository(artifacts, self.tokenizer)
        self.curriculum = Curriculum(config)
        train_records = {
            lane: self.repository.records("train", lane)
            for lane in ("general", "code", "instruction")
        }
        proxy_records = {"general": self.repository.records("score", "general")}
        if state is None:
            self.next_step = 0
            self.model = TinyCausalLM(
                self.tokenizer.vocab_size,
                config["sequence_length"],
                config["model"],
            )
            self.train_packer = DeterministicPacker(
                train_records,
                self.tokenizer,
                config["sequence_length"],
                "train",
            )
            self.proxy_packer = DeterministicPacker(
                proxy_records,
                self.tokenizer,
                config["sequence_length"],
                "proxy",
            )
            self.opus = OPUSSelector(
                projection_dimension=config["projection_dimension"],
                temperature=config["opus_temperature"],
                selection_ratio=config["selection_ratio"],
                max_deferrals=config["max_deferrals"],
                sketch_seed=config["seed"] + 13,
                rng_state=config["seed"] + 29,
            )
            self.deferred_by_lane: dict[str, list[dict[str, Any]]] = {
                lane: [] for lane in train_records
            }
        else:
            self.next_step = int(state["next_step"])
            self.model = TinyCausalLM(
                self.tokenizer.vocab_size,
                config["sequence_length"],
                config["model"],
                state["model"],
            )
            self.train_packer = DeterministicPacker(
                train_records,
                self.tokenizer,
                config["sequence_length"],
                "train",
                state["train_packer"]["cursors"],
            )
            self.proxy_packer = DeterministicPacker(
                proxy_records,
                self.tokenizer,
                config["sequence_length"],
                "proxy",
                state["proxy_packer"]["cursors"],
            )
            opus_state = copy.deepcopy(state["opus"])
            opus_state["temperature"] = config["opus_temperature"]
            self.opus = OPUSSelector.from_state(opus_state)
            self.deferred_by_lane = copy.deepcopy(state["deferred_by_lane"])
        expected = ledger_states if attach_existing_ledgers else None
        self.ledgers = LedgerSet(artifacts / "ledgers", branch_id, expected)

    @classmethod
    def from_checkpoint(
        cls,
        root: Path,
        artifacts: Path,
        checkpoint_path: Path,
        config: dict[str, Any],
        config_hash: str,
        code_hash: str,
        logger: RunLogger,
        branch_override: str | None = None,
        branch_info_override: dict[str, Any] | None = None,
        attach_parent_ledgers: bool = True,
    ) -> tuple["TrainingEngine", dict[str, Any]]:
        checkpoint = load_checkpoint(checkpoint_path)
        payload = checkpoint["payload"]
        if payload["tokenizer_hash"] != FrozenByteTokenizer.create().tokenizer_hash:
            raise ValueError("checkpoint tokenizer hash does not match runtime")
        if payload["code_hash"] != code_hash:
            raise ValueError("checkpoint code hash does not match runtime")
        branch_id = branch_override or payload["branch_id"]
        if branch_override is None and payload["config_hash"] != config_hash:
            raise ValueError("checkpoint config hash does not match runtime")
        branch_info = branch_info_override or payload["branch_info"]
        engine = cls(
            root,
            artifacts,
            config,
            config_hash,
            code_hash,
            branch_id,
            branch_info,
            logger,
            state=payload["engine_state"],
            ledger_states=payload["ledger_states"],
            attach_existing_ledgers=attach_parent_ledgers and branch_override is None,
        )
        return engine, checkpoint

    def engine_state(self) -> dict[str, Any]:
        return {
            "next_step": self.next_step,
            "model": self.model.state(),
            "train_packer": self.train_packer.state(),
            "proxy_packer": self.proxy_packer.state(),
            "opus": self.opus.state(),
            "deferred_by_lane": self.deferred_by_lane,
        }

    def checkpoint_payload(self, reason: str) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "reason": reason,
            "branch_id": self.branch_id,
            "branch_info": self.branch_info,
            "next_step": self.next_step,
            "tokenizer_hash": self.tokenizer.tokenizer_hash,
            "root_manifest_hash": self.repository.root["root_manifest_hash"],
            "config_hash": self.config_hash,
            "code_hash": self.code_hash,
            "engine_state": self.engine_state(),
            "ledger_states": self.ledgers.states(),
        }

    def save_checkpoint(self, filename: str, reason: str) -> dict[str, Any]:
        path = self.artifacts / "checkpoints" / filename
        document = write_checkpoint(path, self.checkpoint_payload(reason))
        self.logger.log(
            f"[PASS] checkpoint_saved branch={self.branch_id} next_step={self.next_step} "
            f"hash={document['checkpoint_hash']} path=checkpoints/{filename}"
        )
        return {**document, "path": path}

    def prepare_step(self) -> dict[str, Any]:
        step = self.next_step
        if step >= self.config["total_steps"]:
            raise ValueError("training is already complete")
        started = time.perf_counter_ns()
        candidates: list[dict[str, Any]] = []
        for slot, lane in enumerate(self.curriculum.lane_slots(step)):
            queue = self.deferred_by_lane[lane]
            if queue:
                candidate = queue.pop(0)
            else:
                candidate = self.train_packer.pack_next(lane, step, slot)
            validate_packed(candidate, self.tokenizer)
            candidates.append(candidate)
        packed_at = time.perf_counter_ns()
        proxies = [
            self.proxy_packer.pack_next("general", step, proxy_slot)
            for proxy_slot in range(self.config["proxy_batch_size"])
        ]
        result = self.opus.select(
            step,
            candidates,
            proxies,
            self.model,
            self.curriculum.protected_counts(step),
        )
        for lane in self.deferred_by_lane:
            self.deferred_by_lane[lane].extend(
                candidate for candidate in result["deferred"] if candidate["lane"] == lane
            )
        selected = result["selected"]
        semantic = {
            "schema_version": 1,
            "step": step,
            "candidate_ids": [candidate["candidate_id"] for candidate in selected],
            "candidate_hashes": [candidate["packed_hash"] for candidate in selected],
            "ordered_source_spans": [
                span for candidate in selected for span in candidate["source_spans"]
            ],
            "input_hash": hash_object(
                [
                    {
                        "input_ids": candidate["input_ids"],
                        "labels": candidate["labels"],
                        "loss_mask": candidate["loss_mask"],
                        "attention_mask": candidate["attention_mask"],
                        "position_ids": candidate["position_ids"],
                    }
                    for candidate in selected
                ]
            ),
            "tokenizer_hash": self.tokenizer.tokenizer_hash,
            "root_manifest_hash": self.repository.root["root_manifest_hash"],
            "opus_result_hash": result["opus_result_hash"],
        }
        batch_hash = hash_object(semantic)
        finished = time.perf_counter_ns()
        return {
            "step": step,
            "stage": self.curriculum.stage_for(step)["name"],
            "candidates": candidates,
            "proxies": proxies,
            "selected": selected,
            "opus": result,
            "batch": {
                **semantic,
                "batch_hash": batch_hash,
                "batch_id": f"batch-{batch_hash[:20]}",
                "selected_candidates": selected,
            },
            "timing_ns": {
                "packing": packed_at - started,
                "opus": finished - packed_at,
                "prepare_total": finished - started,
            },
        }

    def _assert_train_firewall(self, selected: list[dict[str, Any]]) -> None:
        for candidate in selected:
            for span in candidate["source_spans"]:
                self.repository.require_manifest_use(span["shard_id"], "train")

    def commit_step(
        self, prepared: dict[str, Any], save_checkpoint_file: str | None = None
    ) -> dict[str, Any]:
        if prepared["step"] != self.next_step:
            raise ValueError("prepared step does not match engine cursor")
        if any(
            row["branch_id"] == self.branch_id and row["step"] == self.next_step
            for row in self.ledgers["consumption"].rows
        ):
            raise ValueError("duplicate branch optimizer step")
        self._assert_train_firewall(prepared["selected"])
        train_started = time.perf_counter_ns()
        training = self.model.train_batch(prepared["selected"])
        train_finished = time.perf_counter_ns()
        step = self.next_step
        batch = prepared["batch"]
        batch_path = self.artifacts / "batches" / f"{self.branch_id}.step-{step:04d}.json"
        atomic_write_json(
            batch_path,
            {
                **batch,
                "branch_id": self.branch_id,
                "stage": prepared["stage"],
                "model_before_hash": training["model_before_hash"],
            },
        )
        for decision in prepared["opus"]["decisions"]:
            self.ledgers["opus"].append(
                {
                    "branch_id": self.branch_id,
                    "opus_result_hash": prepared["opus"]["opus_result_hash"],
                    **decision,
                }
            )
        non_padding = sum(
            candidate["non_padding_tokens"] for candidate in prepared["selected"]
        )
        capacity = len(prepared["selected"]) * self.config["sequence_length"]
        consumption = self.ledgers["consumption"].append(
            {
                "schema_version": 1,
                "branch_id": self.branch_id,
                "step": step,
                "optimizer_step": training["optimizer_step"],
                "stage": prepared["stage"],
                "batch_id": batch["batch_id"],
                "batch_hash": batch["batch_hash"],
                "batch_path": str(batch_path.relative_to(self.artifacts)),
                "candidate_ids": batch["candidate_ids"],
                "candidate_hashes": batch["candidate_hashes"],
                "ordered_source_spans": batch["ordered_source_spans"],
                "capacity_tokens": capacity,
                "non_padding_tokens": non_padding,
                "loss_bearing_tokens": training["loss_bearing_tokens"],
                "microbatch_size": self.config["microbatch_size"],
                "world_size": self.config["world_size"],
                "gradient_accumulation_steps": self.config[
                    "gradient_accumulation_steps"
                ],
                "global_batch_size": len(prepared["selected"]),
                "planned_lane_quotas": self.curriculum.candidate_quotas(step),
                "protected_counts": self.curriculum.protected_counts(step),
                "selected_lane_counts": {
                    lane: sum(candidate["lane"] == lane for candidate in prepared["selected"])
                    for lane in ("general", "code", "instruction")
                },
                "model_before_hash": training["model_before_hash"],
                "model_after_hash": training["model_after_hash"],
                "optimizer_before_hash": training["optimizer_before_hash"],
                "optimizer_after_hash": training["optimizer_after_hash"],
                "loss_sum": training["loss_sum"],
                "loss_mean": training["loss_mean"],
                "gradient_nonzero_parameters": training[
                    "gradient_nonzero_parameters"
                ],
                "tokenizer_hash": self.tokenizer.tokenizer_hash,
                "root_manifest_hash": self.repository.root["root_manifest_hash"],
                "config_hash": self.config_hash,
                "code_hash": self.code_hash,
                "timing_ns": {
                    **prepared["timing_ns"],
                    "training": train_finished - train_started,
                },
            }
        )
        for candidate, result in zip(prepared["selected"], training["candidate_results"]):
            for metric in result["segment_metrics"]:
                self.ledgers["learning"].append(
                    {
                        "schema_version": 1,
                        "branch_id": self.branch_id,
                        "step": step,
                        "batch_id": batch["batch_id"],
                        "batch_hash": batch["batch_hash"],
                        "candidate_id": candidate["candidate_id"],
                        "consumption_row_hash": consumption["row_hash"],
                        "model_before_hash": training["model_before_hash"],
                        "model_after_hash": training["model_after_hash"],
                        **metric,
                    }
                )
        self.ledgers["events"].append(
            {
                "schema_version": 1,
                "branch_id": self.branch_id,
                "event": "step_committed",
                "step": step,
                "batch_id": batch["batch_id"],
                "batch_hash": batch["batch_hash"],
                "consumption_row_hash": consumption["row_hash"],
            }
        )
        self.next_step += 1
        checkpoint = None
        if save_checkpoint_file is not None:
            checkpoint = self.save_checkpoint(save_checkpoint_file, "step_committed")
        self.logger.log(
            f"[PASS] batch_trained branch={self.branch_id} step={step} "
            f"batch_id={batch['batch_id']} loss={training['loss_mean']:.6f}"
        )
        return {
            "consumption": consumption,
            "training": training,
            "checkpoint": checkpoint,
        }


def expectation_from_prepared(
    checkpoint_hash: str, prepared: dict[str, Any]
) -> dict[str, Any]:
    batch = prepared["batch"]
    body = {
        "schema_version": 1,
        "checkpoint_hash": checkpoint_hash,
        "step": prepared["step"],
        "batch_id": batch["batch_id"],
        "batch_hash": batch["batch_hash"],
        "candidate_ids": batch["candidate_ids"],
        "candidate_hashes": batch["candidate_hashes"],
        "ordered_source_spans": batch["ordered_source_spans"],
    }
    return {**body, "expectation_hash": hash_object(body)}
