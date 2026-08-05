from __future__ import annotations

import math
from collections import Counter, defaultdict
from typing import Any

from .canonical import hash_object


class TinyCausalLM:
    """A sparse causal context LM with position bias and real AdamW updates."""

    def __init__(
        self,
        vocab_size: int,
        sequence_length: int,
        optimizer_config: dict[str, float],
        state: dict[str, Any] | None = None,
    ):
        self.vocab_size = vocab_size
        self.sequence_length = sequence_length
        self.optimizer_config = optimizer_config
        self.position_offset = vocab_size * vocab_size
        if state is None:
            self.weights: dict[int, float] = {}
            self.first_moment: dict[int, float] = {}
            self.second_moment: dict[int, float] = {}
            self.optimizer_step = 0
        else:
            self.weights = {int(key): float(value) for key, value in state["weights"].items()}
            self.first_moment = {
                int(key): float(value) for key, value in state["first_moment"].items()
            }
            self.second_moment = {
                int(key): float(value) for key, value in state["second_moment"].items()
            }
            self.optimizer_step = int(state["optimizer_step"])

    def _weight_index(self, token: int, output: int) -> int:
        return token * self.vocab_size + output

    def _position_index(self, position: int, output: int) -> int:
        return self.position_offset + position * self.vocab_size + output

    def state(self) -> dict[str, Any]:
        def encode(values: dict[int, float]) -> dict[str, float]:
            return {str(key): values[key] for key in sorted(values)}

        return {
            "schema_version": 1,
            "model_type": "sparse_causal_context_lm",
            "vocab_size": self.vocab_size,
            "sequence_length": self.sequence_length,
            "weights": encode(self.weights),
            "first_moment": encode(self.first_moment),
            "second_moment": encode(self.second_moment),
            "optimizer_step": self.optimizer_step,
        }

    def model_hash(self) -> str:
        return hash_object(
            {
                "weights": {str(key): self.weights[key] for key in sorted(self.weights)},
                "vocab_size": self.vocab_size,
                "sequence_length": self.sequence_length,
            }
        )

    def optimizer_hash(self) -> str:
        return hash_object(
            {
                "first_moment": {
                    str(key): self.first_moment[key] for key in sorted(self.first_moment)
                },
                "second_moment": {
                    str(key): self.second_moment[key] for key in sorted(self.second_moment)
                },
                "step": self.optimizer_step,
                "config": self.optimizer_config,
            }
        )

    def loss_and_gradient(
        self, candidate: dict[str, Any], need_gradient: bool = True
    ) -> dict[str, Any]:
        gradient: dict[int, float] = defaultdict(float)
        losses_by_segment: dict[int, list[float]] = defaultdict(list)
        total_loss = 0.0
        loss_tokens = 0
        for row_index, enabled in enumerate(candidate["loss_mask"]):
            if not enabled:
                continue
            visible_tokens = [
                candidate["input_ids"][column]
                for column, allowed in enumerate(candidate["attention_mask"][row_index])
                if allowed
            ]
            if not visible_tokens:
                raise ValueError("loss-bearing token has no causal context")
            counts = Counter(visible_tokens)
            inverse_count = 1.0 / len(visible_tokens)
            position = candidate["position_ids"][row_index]
            logits = [0.0] * self.vocab_size
            for output in range(self.vocab_size):
                value = self.weights.get(self._position_index(position, output), 0.0)
                for token, count in counts.items():
                    value += (
                        count
                        * inverse_count
                        * self.weights.get(self._weight_index(token, output), 0.0)
                    )
                logits[output] = value
            maximum = max(logits)
            exponentials = [math.exp(value - maximum) for value in logits]
            denominator = sum(exponentials)
            probabilities = [value / denominator for value in exponentials]
            target = candidate["labels"][row_index]
            token_loss = -math.log(max(probabilities[target], 1e-300))
            total_loss += token_loss
            loss_tokens += 1
            segment_id = candidate["segment_ids"][row_index]
            losses_by_segment[segment_id].append(token_loss)
            if need_gradient:
                probabilities[target] -= 1.0
                for output, error in enumerate(probabilities):
                    gradient[self._position_index(position, output)] += error
                    for token, count in counts.items():
                        gradient[self._weight_index(token, output)] += (
                            error * count * inverse_count
                        )
        if loss_tokens <= 0:
            raise ValueError("candidate has no loss-bearing tokens")
        if need_gradient:
            inverse_tokens = 1.0 / loss_tokens
            gradient = {
                index: value * inverse_tokens
                for index, value in gradient.items()
                if value != 0.0
            }
        segment_metrics = []
        for segment_id, span in enumerate(candidate["source_spans"]):
            values = losses_by_segment.get(segment_id, [])
            segment_metrics.append(
                {
                    "span_hash": span["span_hash"],
                    "record_id": span["record_id"],
                    "lane": span["lane"],
                    "data_type": span["data_type"],
                    "loss_bearing_tokens": len(values),
                    "loss_sum": sum(values),
                    "loss_mean": sum(values) / len(values) if values else 0.0,
                }
            )
        return {
            "loss_sum": total_loss,
            "loss_mean": total_loss / loss_tokens,
            "loss_tokens": loss_tokens,
            "gradient": gradient,
            "segment_metrics": segment_metrics,
        }

    def precondition_gradient(self, gradient: dict[int, float]) -> dict[int, float]:
        learning_rate = float(self.optimizer_config["learning_rate"])
        beta1 = float(self.optimizer_config["beta1"])
        beta2 = float(self.optimizer_config["beta2"])
        epsilon = float(self.optimizer_config["epsilon"])
        next_step = self.optimizer_step + 1
        first_correction = 1.0 - beta1**next_step
        second_correction = 1.0 - beta2**self.optimizer_step if self.optimizer_step else 1.0
        coefficient = learning_rate * (1.0 - beta1) / first_correction
        result: dict[int, float] = {}
        for index, value in gradient.items():
            variance = self.second_moment.get(index, 0.0) / second_correction
            result[index] = coefficient * value / (math.sqrt(variance) + epsilon)
        return result

    def _apply_gradient(self, gradient: dict[int, float]) -> None:
        cfg = self.optimizer_config
        learning_rate = float(cfg["learning_rate"])
        beta1 = float(cfg["beta1"])
        beta2 = float(cfg["beta2"])
        epsilon = float(cfg["epsilon"])
        weight_decay = float(cfg["weight_decay"])
        self.optimizer_step += 1
        first_correction = 1.0 - beta1**self.optimizer_step
        second_correction = 1.0 - beta2**self.optimizer_step
        if weight_decay:
            decay = 1.0 - learning_rate * weight_decay
            for index in list(self.weights):
                self.weights[index] *= decay
        for index, value in gradient.items():
            first = beta1 * self.first_moment.get(index, 0.0) + (1.0 - beta1) * value
            second = beta2 * self.second_moment.get(index, 0.0) + (1.0 - beta2) * value * value
            self.first_moment[index] = first
            self.second_moment[index] = second
            update = (first / first_correction) / (
                math.sqrt(second / second_correction) + epsilon
            )
            self.weights[index] = self.weights.get(index, 0.0) - learning_rate * update

    def train_batch(self, candidates: list[dict[str, Any]]) -> dict[str, Any]:
        if not candidates:
            raise ValueError("cannot train an empty batch")
        before_model = self.model_hash()
        before_optimizer = self.optimizer_hash()
        candidate_results = [self.loss_and_gradient(candidate) for candidate in candidates]
        total_tokens = sum(result["loss_tokens"] for result in candidate_results)
        combined: dict[int, float] = defaultdict(float)
        for result in candidate_results:
            weight = result["loss_tokens"] / total_tokens
            for index, value in result["gradient"].items():
                combined[index] += value * weight
        self._apply_gradient(dict(combined))
        return {
            "model_before_hash": before_model,
            "optimizer_before_hash": before_optimizer,
            "model_after_hash": self.model_hash(),
            "optimizer_after_hash": self.optimizer_hash(),
            "optimizer_step": self.optimizer_step,
            "loss_sum": sum(result["loss_sum"] for result in candidate_results),
            "loss_bearing_tokens": total_tokens,
            "loss_mean": sum(result["loss_sum"] for result in candidate_results) / total_tokens,
            "candidate_results": candidate_results,
            "gradient_nonzero_parameters": len(combined),
        }
