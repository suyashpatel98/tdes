from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .canonical import hash_object
from .tokenizer import FrozenByteTokenizer


@dataclass
class Segment:
    tokens: list[int]
    loss_targets: list[int]
    record: dict[str, Any]
    occurrence: int
    field_spans: list[dict[str, Any]]
    truncated: bool


def _document_segment(
    record: dict[str, Any], tokenizer: FrozenByteTokenizer, limit: int, occurrence: int
) -> Segment:
    source = record["token_fields"]["text"]
    used = source[: max(0, limit - 2)]
    tokens = [tokenizer.special("BOS"), *used, tokenizer.special("EOS")]
    loss_targets = [1] * (len(tokens) - 1) + [0]
    return Segment(
        tokens=tokens,
        loss_targets=loss_targets,
        record=record,
        occurrence=occurrence,
        field_spans=[
            {
                "field": "text",
                "source_start": 0,
                "source_end": len(used),
                "segment_start": 1,
                "segment_end": 1 + len(used),
            }
        ],
        truncated=len(used) != len(source),
    )


def _prompt_completion_segment(
    record: dict[str, Any], tokenizer: FrozenByteTokenizer, limit: int, occurrence: int
) -> Segment:
    prompt = record["token_fields"]["prompt"]
    response = record["token_fields"]["response"]
    budget = max(2, limit - 4)
    if len(prompt) + len(response) > budget:
        response_budget = min(len(response), max(1, budget // 2))
        prompt_budget = min(len(prompt), budget - response_budget)
        if prompt_budget + response_budget < budget:
            response_budget = min(len(response), budget - prompt_budget)
        used_prompt = prompt[:prompt_budget]
        used_response = response[:response_budget]
    else:
        used_prompt = prompt
        used_response = response
    response_marker = 2 + len(used_prompt)
    tokens = [
        tokenizer.special("BOS"),
        tokenizer.special("PROMPT"),
        *used_prompt,
        tokenizer.special("RESPONSE"),
        *used_response,
        tokenizer.special("EOS"),
    ]
    loss_targets = []
    for input_position in range(len(tokens)):
        target_position = input_position + 1
        loss_targets.append(
            int(target_position < len(tokens) and target_position > response_marker)
        )
    prompt_start = 2
    response_start = response_marker + 1
    return Segment(
        tokens=tokens,
        loss_targets=loss_targets,
        record=record,
        occurrence=occurrence,
        field_spans=[
            {
                "field": "prompt",
                "source_start": 0,
                "source_end": len(used_prompt),
                "segment_start": prompt_start,
                "segment_end": prompt_start + len(used_prompt),
            },
            {
                "field": "response",
                "source_start": 0,
                "source_end": len(used_response),
                "segment_start": response_start,
                "segment_end": response_start + len(used_response),
            },
        ],
        truncated=len(used_prompt) != len(prompt) or len(used_response) != len(response),
    )


def make_segment(
    record: dict[str, Any], tokenizer: FrozenByteTokenizer, limit: int, occurrence: int
) -> Segment:
    if record["data_type"] == "document":
        return _document_segment(record, tokenizer, limit, occurrence)
    if record["data_type"] == "prompt_completion":
        return _prompt_completion_segment(record, tokenizer, limit, occurrence)
    raise ValueError(f"unknown data type: {record['data_type']}")


class DeterministicPacker:
    def __init__(
        self,
        records_by_lane: dict[str, list[dict[str, Any]]],
        tokenizer: FrozenByteTokenizer,
        sequence_length: int,
        stream_name: str,
        cursors: dict[str, int] | None = None,
    ):
        self.records_by_lane = {
            lane: sorted(records, key=lambda row: row["record_id"])
            for lane, records in records_by_lane.items()
        }
        self.tokenizer = tokenizer
        self.sequence_length = sequence_length
        self.stream_name = stream_name
        self.cursors = dict(cursors or {lane: 0 for lane in records_by_lane})

    def state(self) -> dict[str, Any]:
        return {"cursors": dict(self.cursors), "stream_name": self.stream_name}

    def _peek(self, lane: str) -> tuple[dict[str, Any], int]:
        records = self.records_by_lane[lane]
        if not records:
            raise ValueError(f"empty lane: {lane}")
        absolute = self.cursors.get(lane, 0)
        return records[absolute % len(records)], absolute // len(records)

    def pack_next(self, lane: str, step: int, slot: int) -> dict[str, Any]:
        segments: list[Segment] = []
        used = 0
        while used < self.sequence_length:
            record, occurrence = self._peek(lane)
            segment = make_segment(
                record, self.tokenizer, self.sequence_length - used, occurrence
            )
            if segments and used + len(segment.tokens) > self.sequence_length:
                break
            if len(segment.tokens) < 2:
                raise ValueError(f"empty segment: {record['record_id']}")
            segments.append(segment)
            used += len(segment.tokens)
            self.cursors[lane] = self.cursors.get(lane, 0) + 1
            if used == self.sequence_length:
                break
            next_record, next_occurrence = self._peek(lane)
            next_segment = make_segment(
                next_record, self.tokenizer, self.sequence_length, next_occurrence
            )
            if used + len(next_segment.tokens) > self.sequence_length:
                break

        length = self.sequence_length
        pad = self.tokenizer.special("PAD")
        input_ids = [pad] * length
        labels = [pad] * length
        loss_mask = [0] * length
        position_ids = [0] * length
        segment_ids = [-1] * length
        attention_mask = [[0] * length for _ in range(length)]
        source_spans: list[dict[str, Any]] = []
        cursor = 0
        for segment_index, segment in enumerate(segments):
            start = cursor
            end = start + len(segment.tokens)
            for local, token in enumerate(segment.tokens):
                packed = start + local
                input_ids[packed] = token
                position_ids[packed] = local
                segment_ids[packed] = segment_index
                for visible in range(local + 1):
                    attention_mask[packed][start + visible] = 1
                if local + 1 < len(segment.tokens):
                    labels[packed] = segment.tokens[local + 1]
                    loss_mask[packed] = segment.loss_targets[local]
            field_spans = []
            for span in segment.field_spans:
                field_spans.append(
                    {
                        **span,
                        "packed_start": start + span["segment_start"],
                        "packed_end": start + span["segment_end"],
                    }
                )
            span_body = {
                "record_id": segment.record["record_id"],
                "occurrence": segment.occurrence,
                "shard_id": segment.record["shard_id"],
                "shard_hash": segment.record["shard_hash"],
                "token_record_hash": segment.record["token_record_hash"],
                "source_content_hash": segment.record["source_content_hash"],
                "data_type": segment.record["data_type"],
                "lane": segment.record["lane"],
                "packed_start": start,
                "packed_end": end,
                "field_spans": field_spans,
                "truncated": segment.truncated,
                "loss_bearing_tokens": sum(loss_mask[start:end]),
            }
            source_spans.append({**span_body, "span_hash": hash_object(span_body)})
            cursor = end

        semantic = {
            "schema_version": 1,
            "stream": self.stream_name,
            "step_created": step,
            "slot_created": slot,
            "lane": lane,
            "sequence_length": length,
            "tokenizer_hash": self.tokenizer.tokenizer_hash,
            "input_ids": input_ids,
            "labels": labels,
            "loss_mask": loss_mask,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "segment_ids": segment_ids,
            "source_spans": source_spans,
            "non_padding_tokens": cursor,
            "loss_bearing_tokens": sum(loss_mask),
        }
        packed_hash = hash_object(semantic)
        candidate_id = f"{self.stream_name}-{packed_hash[:20]}"
        candidate = {
            **semantic,
            "packed_hash": packed_hash,
            "candidate_id": candidate_id,
            "deferral_count": 0,
        }
        validate_packed(candidate, self.tokenizer)
        return candidate


def validate_packed(candidate: dict[str, Any], tokenizer: FrozenByteTokenizer) -> None:
    length = candidate["sequence_length"]
    arrays = ("input_ids", "labels", "loss_mask", "position_ids", "segment_ids")
    if any(len(candidate[name]) != length for name in arrays):
        raise ValueError("packed vector shape mismatch")
    if len(candidate["attention_mask"]) != length or any(
        len(row) != length for row in candidate["attention_mask"]
    ):
        raise ValueError("attention mask shape mismatch")
    pad = tokenizer.special("PAD")
    for row_index in range(length):
        segment = candidate["segment_ids"][row_index]
        if segment < 0:
            if candidate["input_ids"][row_index] != pad:
                raise ValueError("non-pad token outside a segment")
            if candidate["loss_mask"][row_index] != 0 or any(
                candidate["attention_mask"][row_index]
            ):
                raise ValueError("padding contributes attention or loss")
            continue
        for column, allowed in enumerate(candidate["attention_mask"][row_index]):
            expected = int(
                candidate["segment_ids"][column] == segment
                and candidate["position_ids"][column]
                <= candidate["position_ids"][row_index]
                and candidate["segment_ids"][column] >= 0
            )
            if allowed != expected:
                raise ValueError("attention mask violates block-causal policy")
        if candidate["loss_mask"][row_index]:
            next_index = row_index + 1
            if (
                next_index >= length
                or candidate["segment_ids"][next_index] != segment
                or candidate["labels"][row_index] != candidate["input_ids"][next_index]
            ):
                raise ValueError("loss target crosses a segment boundary")
    if sum(candidate["loss_mask"]) != candidate["loss_bearing_tokens"]:
        raise ValueError("loss token count mismatch")
    for span in candidate["source_spans"]:
        body = {key: value for key, value in span.items() if key != "span_hash"}
        if hash_object(body) != span["span_hash"]:
            raise ValueError("source span hash mismatch")


def candidate_semantic_hash(candidate: dict[str, Any]) -> str:
    ignored = {"deferral_count", "packed_hash", "candidate_id"}
    return hash_object({key: value for key, value in candidate.items() if key not in ignored})
