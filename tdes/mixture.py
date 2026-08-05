from __future__ import annotations

import math
from typing import Any

from .canonical import hash_object, stable_apportion


class Curriculum:
    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.stages = config["stages"]
        self._validate()

    def _validate(self) -> None:
        expected_start = 0
        lanes: set[str] | None = None
        for stage in self.stages:
            if stage["start_step"] != expected_start or stage["end_step"] <= expected_start:
                raise ValueError("curriculum stages must be contiguous")
            expected_start = stage["end_step"]
            stage_lanes = set(stage["lane_weights"])
            if lanes is not None and stage_lanes != lanes:
                raise ValueError("lane set changes across stages")
            lanes = stage_lanes
            for lane, floor in stage.get("protected_floors", {}).items():
                if lane not in stage_lanes or not 0 <= floor <= 1:
                    raise ValueError("invalid protected floor")
        if expected_start != self.config["total_steps"]:
            raise ValueError("curriculum does not cover all steps")

    def stage_for(self, step: int) -> dict[str, Any]:
        for stage in self.stages:
            if stage["start_step"] <= step < stage["end_step"]:
                return stage
        raise ValueError(f"step outside curriculum: {step}")

    def candidate_quotas(self, step: int) -> dict[str, int]:
        stage = self.stage_for(step)
        total = self.config["candidate_buffer_size"]
        quotas = stable_apportion(total, stage["lane_weights"])
        selected = int(total * self.config["selection_ratio"])
        for lane, floor in stage.get("protected_floors", {}).items():
            required = math.ceil(selected * floor)
            if quotas[lane] < required:
                donors = sorted(
                    (key for key in quotas if key != lane),
                    key=lambda key: (-quotas[key], key),
                )
                while quotas[lane] < required:
                    donor = next(key for key in donors if quotas[key] > 0)
                    quotas[donor] -= 1
                    quotas[lane] += 1
        return quotas

    def lane_slots(self, step: int) -> list[str]:
        stage = self.stage_for(step)
        quotas = self.candidate_quotas(step)
        slots: list[str] = []
        used = {lane: 0 for lane in quotas}
        total = sum(quotas.values())
        for position in range(total):
            eligible = [lane for lane in quotas if used[lane] < quotas[lane]]
            lane = max(
                eligible,
                key=lambda item: (
                    stage["lane_weights"][item] * (position + 1) - used[item],
                    -sorted(quotas).index(item),
                ),
            )
            slots.append(lane)
            used[lane] += 1
        return slots

    def protected_counts(self, step: int) -> dict[str, int]:
        stage = self.stage_for(step)
        selected = int(
            self.config["candidate_buffer_size"] * self.config["selection_ratio"]
        )
        return {
            lane: math.ceil(selected * floor)
            for lane, floor in stage.get("protected_floors", {}).items()
        }

    def compiled_plan(self) -> dict[str, Any]:
        steps = []
        for step in range(self.config["total_steps"]):
            stage = self.stage_for(step)
            steps.append(
                {
                    "step": step,
                    "stage": stage["name"],
                    "lane_weights": stage["lane_weights"],
                    "protected_floors": stage.get("protected_floors", {}),
                    "candidate_quotas": self.candidate_quotas(step),
                    "lane_slots": self.lane_slots(step),
                    "required_selected": self.protected_counts(step),
                }
            )
        body = {"schema_version": 1, "steps": steps}
        return {**body, "mixture_plan_hash": hash_object(body)}
