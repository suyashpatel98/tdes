from __future__ import annotations

import copy
import unittest

from tdes.packing import DeterministicPacker, validate_packed

from tests.support import RepositoryFixture


class PackingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.fixture = RepositoryFixture()

    @classmethod
    def tearDownClass(cls) -> None:
        cls.fixture.close()

    def pack(self, lane: str):
        packer = DeterministicPacker(
            {lane: self.fixture.repository.records("train", lane)},
            self.fixture.tokenizer,
            self.fixture.config["sequence_length"],
            "test",
        )
        return packer.pack_next(lane, 0, 0)

    def test_document_masks_are_block_causal(self) -> None:
        candidate = self.pack("code")
        validate_packed(candidate, self.fixture.tokenizer)
        self.assertGreaterEqual(len(candidate["source_spans"]), 2)
        first_end = candidate["source_spans"][0]["packed_end"]
        second_start = candidate["source_spans"][1]["packed_start"]
        self.assertEqual(candidate["attention_mask"][second_start][first_end - 1], 0)
        self.assertEqual(candidate["position_ids"][second_start], 0)
        self.assertEqual(candidate["loss_mask"][first_end - 1], 0)

    def test_instruction_prompt_has_no_loss(self) -> None:
        candidate = self.pack("instruction")
        span = candidate["source_spans"][0]
        prompt = next(item for item in span["field_spans"] if item["field"] == "prompt")
        response = next(item for item in span["field_spans"] if item["field"] == "response")
        for position in range(prompt["packed_start"] - 1, prompt["packed_end"]):
            self.assertEqual(candidate["loss_mask"][position], 0)
        self.assertEqual(candidate["loss_mask"][response["packed_start"] - 1], 1)

    def test_cross_segment_attention_tampering_fails(self) -> None:
        candidate = self.pack("code")
        changed = copy.deepcopy(candidate)
        second_start = changed["source_spans"][1]["packed_start"]
        changed["attention_mask"][second_start][0] = 1
        with self.assertRaises(ValueError):
            validate_packed(changed, self.fixture.tokenizer)

    def test_position_id_tampering_fails(self) -> None:
        candidate = self.pack("code")
        changed = copy.deepcopy(candidate)
        changed["position_ids"][2] = changed["position_ids"][1]
        with self.assertRaisesRegex(ValueError, "position IDs"):
            validate_packed(changed, self.fixture.tokenizer)

    def test_prompt_loss_mask_tampering_fails(self) -> None:
        candidate = self.pack("instruction")
        changed = copy.deepcopy(candidate)
        prompt = changed["source_spans"][0]["field_spans"][0]
        changed["loss_mask"][prompt["packed_start"]] = 1
        changed["loss_bearing_tokens"] += 1
        with self.assertRaisesRegex(ValueError, "data-type policy"):
            validate_packed(changed, self.fixture.tokenizer)


if __name__ == "__main__":
    unittest.main()
