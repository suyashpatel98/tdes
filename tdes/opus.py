from __future__ import annotations

import math
from collections import defaultdict
from typing import Any

from .canonical import hash_object
from .model import TinyCausalLM


class XorShift64:
    def __init__(self, state: int):
        self.state = state & ((1 << 64) - 1)
        if self.state == 0:
            self.state = 0x9E3779B97F4A7C15

    def uniform(self) -> float:
        value = self.state
        value ^= (value << 13) & ((1 << 64) - 1)
        value ^= value >> 7
        value ^= (value << 17) & ((1 << 64) - 1)
        self.state = value & ((1 << 64) - 1)
        return self.state / float(1 << 64)


class CountSketch:
    def __init__(self, dimension: int, seed: int):
        if dimension <= 0:
            raise ValueError("projection dimension must be positive")
        self.dimension = dimension
        self.seed = seed & ((1 << 64) - 1)

    def _bucket(self, index: int) -> int:
        value = (index ^ self.seed) * 0x9E3779B185EBCA87
        value ^= value >> 33
        return (value & ((1 << 64) - 1)) % self.dimension

    def _sign(self, index: int) -> float:
        value = (index + self.seed * 0xC2B2AE3D27D4EB4F) & ((1 << 64) - 1)
        value ^= value >> 29
        return 1.0 if value & 1 else -1.0

    def project(self, values: dict[int, float]) -> list[float]:
        result = [0.0] * self.dimension
        for index, value in values.items():
            result[self._bucket(index)] += self._sign(index) * value
        return result


def _dot(left: list[float], right: list[float]) -> float:
    return sum(a * b for a, b in zip(left, right))


def _add_in_place(target: list[float], source: list[float]) -> None:
    for index, value in enumerate(source):
        target[index] += value


class OPUSSelector:
    def __init__(
        self,
        projection_dimension: int,
        temperature: float,
        selection_ratio: float,
        max_deferrals: int,
        sketch_seed: int,
        rng_state: int,
    ):
        if temperature <= 0 or not 0 < selection_ratio <= 1:
            raise ValueError("invalid OPUS configuration")
        self.projection_dimension = projection_dimension
        self.temperature = temperature
        self.selection_ratio = selection_ratio
        self.max_deferrals = max_deferrals
        self.sketch_seed = sketch_seed
        self.rng = XorShift64(rng_state)

    def state(self) -> dict[str, Any]:
        return {
            "projection_dimension": self.projection_dimension,
            "temperature": self.temperature,
            "selection_ratio": self.selection_ratio,
            "max_deferrals": self.max_deferrals,
            "sketch_seed": self.sketch_seed,
            "rng_state": self.rng.state,
        }

    @classmethod
    def from_state(cls, state: dict[str, Any]) -> "OPUSSelector":
        return cls(**state)

    def select(
        self,
        step: int,
        candidates: list[dict[str, Any]],
        proxies: list[dict[str, Any]],
        model: TinyCausalLM,
        protected_counts: dict[str, int],
    ) -> dict[str, Any]:
        if not candidates or not proxies:
            raise ValueError("OPUS requires candidates and a proxy batch")
        target_size = int(len(candidates) * self.selection_ratio)
        if target_size <= 0:
            raise ValueError("selection ratio produced an empty update batch")
        sketch = CountSketch(self.projection_dimension, self.sketch_seed)
        model_hash = model.model_hash()
        optimizer_hash = model.optimizer_hash()

        proxy_results = [model.loss_and_gradient(proxy) for proxy in proxies]
        proxy_gradient: dict[int, float] = defaultdict(float)
        for result in proxy_results:
            for index, value in result["gradient"].items():
                proxy_gradient[index] += value / len(proxy_results)
        proxy_sketch = sketch.project(dict(proxy_gradient))
        proxy_hash = hash_object(
            {
                "candidate_ids": [proxy["candidate_id"] for proxy in proxies],
                "packed_hashes": [proxy["packed_hash"] for proxy in proxies],
                "gradient_hash": hash_object(
                    {str(key): proxy_gradient[key] for key in sorted(proxy_gradient)}
                ),
            }
        )

        candidate_results: dict[str, dict[str, Any]] = {}
        features: dict[str, list[float]] = {}
        for candidate in candidates:
            result = model.loss_and_gradient(candidate)
            candidate_results[candidate["candidate_id"]] = result
            features[candidate["candidate_id"]] = sketch.project(
                model.precondition_gradient(result["gradient"])
            )

        eta = float(model.optimizer_config["learning_rate"])
        history = [0.0] * self.projection_dimension
        remaining = list(candidates)
        selected: list[dict[str, Any]] = []
        traces: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for rank in range(target_size):
            scored: list[tuple[dict[str, Any], float, float, float]] = []
            for candidate in remaining:
                feature = features[candidate["candidate_id"]]
                alignment = eta * _dot(feature, proxy_sketch)
                redundancy = eta * eta * _dot(feature, history)
                utility = alignment - redundancy
                scored.append((candidate, utility, alignment, redundancy))
            maximum = max(item[1] / self.temperature for item in scored)
            weights = [math.exp(item[1] / self.temperature - maximum) for item in scored]
            total_weight = sum(weights)
            probabilities = [weight / total_weight for weight in weights]
            draw = self.rng.uniform()
            cumulative = 0.0
            chosen_index = len(scored) - 1
            for index, probability in enumerate(probabilities):
                cumulative += probability
                if draw < cumulative:
                    chosen_index = index
                    break
            for index, (candidate, utility, alignment, redundancy) in enumerate(scored):
                traces[candidate["candidate_id"]].append(
                    {
                        "selection_rank": rank,
                        "utility": utility,
                        "alignment": alignment,
                        "redundancy_penalty": redundancy,
                        "probability": probabilities[index],
                        "rng_draw": draw if index == chosen_index else None,
                    }
                )
            chosen = scored[chosen_index][0]
            selected.append(chosen)
            _add_in_place(history, features[chosen["candidate_id"]])
            remaining = [
                candidate
                for candidate in remaining
                if candidate["candidate_id"] != chosen["candidate_id"]
            ]

        initially_selected = {candidate["candidate_id"] for candidate in selected}
        overrides: list[dict[str, Any]] = []
        for protected_lane, required in sorted(protected_counts.items()):
            while sum(candidate["lane"] == protected_lane for candidate in selected) < required:
                eligible = [
                    candidate
                    for candidate in candidates
                    if candidate["lane"] == protected_lane
                    and candidate["candidate_id"]
                    not in {item["candidate_id"] for item in selected}
                ]
                if not eligible:
                    raise ValueError(f"no candidate can satisfy protected floor: {protected_lane}")
                admitted = max(
                    eligible,
                    key=lambda candidate: (
                        traces[candidate["candidate_id"]][-1]["utility"],
                        candidate["candidate_id"],
                    ),
                )
                evictable = []
                for candidate in selected:
                    lane = candidate["lane"]
                    lane_required = protected_counts.get(lane, 0)
                    lane_selected = sum(item["lane"] == lane for item in selected)
                    if lane != protected_lane and lane_selected > lane_required:
                        evictable.append(candidate)
                if not evictable:
                    raise ValueError("protected floor reconciliation has no evictable candidate")
                evicted = min(
                    evictable,
                    key=lambda candidate: (
                        traces[candidate["candidate_id"]][-1]["utility"],
                        candidate["candidate_id"],
                    ),
                )
                position = next(
                    index
                    for index, candidate in enumerate(selected)
                    if candidate["candidate_id"] == evicted["candidate_id"]
                )
                selected[position] = admitted
                overrides.append(
                    {
                        "lane": protected_lane,
                        "required": required,
                        "admitted_candidate_id": admitted["candidate_id"],
                        "evicted_candidate_id": evicted["candidate_id"],
                    }
                )

        final_selected = {candidate["candidate_id"] for candidate in selected}
        admitted_by_override = {
            event["admitted_candidate_id"] for event in overrides
        }
        evicted_by_override = {
            event["evicted_candidate_id"] for event in overrides
        }
        decisions: list[dict[str, Any]] = []
        deferred: list[dict[str, Any]] = []
        for candidate in candidates:
            candidate_id = candidate["candidate_id"]
            if candidate_id in final_selected:
                if candidate_id in admitted_by_override:
                    disposition = "protected_floor_override"
                    reason = "admitted_to_meet_lane_floor"
                else:
                    disposition = "accepted"
                    reason = "selected_by_boltzmann"
            elif candidate["deferral_count"] < self.max_deferrals:
                disposition = "deferred"
                reason = (
                    "displaced_by_protected_floor"
                    if candidate_id in evicted_by_override
                    else "not_selected_retry_available"
                )
                deferred.append({**candidate, "deferral_count": candidate["deferral_count"] + 1})
            else:
                disposition = "rejected"
                reason = "deferral_budget_exhausted"
            linked_override = next(
                (
                    event
                    for event in overrides
                    if candidate_id
                    in (event["admitted_candidate_id"], event["evicted_candidate_id"])
                ),
                None,
            )
            decision_body = {
                "schema_version": 1,
                "step": step,
                "candidate_id": candidate_id,
                "packed_hash": candidate["packed_hash"],
                "lane": candidate["lane"],
                "source_span_hashes": [
                    span["span_hash"] for span in candidate["source_spans"]
                ],
                "model_hash": model_hash,
                "optimizer_hash": optimizer_hash,
                "proxy_hash": proxy_hash,
                "proxy_candidate_ids": [proxy["candidate_id"] for proxy in proxies],
                "projection_dimension": self.projection_dimension,
                "sketch_seed": self.sketch_seed,
                "temperature": self.temperature,
                "traces": traces[candidate_id],
                "selected_by_opus": candidate_id in initially_selected,
                "final_selected": candidate_id in final_selected,
                "prior_deferral_count": candidate["deferral_count"],
                "disposition": disposition,
                "reason": reason,
                "protected_floor_event": linked_override,
                "scoring_loss": candidate_results[candidate_id]["loss_mean"],
            }
            decisions.append(
                {**decision_body, "decision_hash": hash_object(decision_body)}
            )

        result_body = {
            "schema_version": 1,
            "step": step,
            "model_hash": model_hash,
            "optimizer_hash": optimizer_hash,
            "proxy_hash": proxy_hash,
            "candidate_ids": [candidate["candidate_id"] for candidate in candidates],
            "selected_candidate_ids": [candidate["candidate_id"] for candidate in selected],
            "protected_counts": protected_counts,
            "overrides": overrides,
            "rng_state_after": self.rng.state,
            "decisions": decisions,
        }
        return {
            **result_body,
            "selected": selected,
            "deferred": deferred,
            "opus_result_hash": hash_object(
                {key: value for key, value in result_body.items() if key != "decisions"}
            ),
        }
