from __future__ import annotations

import unittest

from tdes.mixture import Curriculum
from tdes.model import TinyCausalLM
from tdes.opus import OPUSSelector
from tdes.packing import DeterministicPacker

from tests.support import RepositoryFixture


class MixtureOPUSTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.fixture = RepositoryFixture()

    @classmethod
    def tearDownClass(cls) -> None:
        cls.fixture.close()

    def _selection(self):
        config = self.fixture.config
        curriculum = Curriculum(config)
        train_packer = DeterministicPacker(
            {
                lane: self.fixture.repository.records("train", lane)
                for lane in ("general", "code", "instruction")
            },
            self.fixture.tokenizer,
            config["sequence_length"],
            "train",
        )
        candidates = [
            train_packer.pack_next(lane, 0, slot)
            for slot, lane in enumerate(curriculum.lane_slots(0))
        ]
        proxy_packer = DeterministicPacker(
            {"general": self.fixture.repository.records("score", "general")},
            self.fixture.tokenizer,
            config["sequence_length"],
            "proxy",
        )
        proxies = [proxy_packer.pack_next("general", 0, slot) for slot in range(2)]
        model = TinyCausalLM(
            self.fixture.tokenizer.vocab_size,
            config["sequence_length"],
            config["model"],
        )
        selector = OPUSSelector(
            config["projection_dimension"],
            config["opus_temperature"],
            config["selection_ratio"],
            config["max_deferrals"],
            config["seed"] + 13,
            config["seed"] + 29,
        )
        return selector.select(0, candidates, proxies, model, curriculum.protected_counts(0))

    def test_curriculum_stage_and_quota_compilation(self) -> None:
        curriculum = Curriculum(self.fixture.config)
        self.assertEqual(curriculum.stage_for(0)["name"], "foundation")
        self.assertEqual(curriculum.stage_for(3)["name"], "reasoning")
        self.assertEqual(curriculum.candidate_quotas(0), {"general": 3, "instruction": 2, "code": 1})
        self.assertEqual(sum(curriculum.candidate_quotas(4).values()), 6)

    def test_opus_is_deterministic_and_meets_floor(self) -> None:
        first = self._selection()
        second = self._selection()
        self.assertEqual(first["selected_candidate_ids"], second["selected_candidate_ids"])
        self.assertEqual(first["opus_result_hash"], second["opus_result_hash"])
        selected_code = sum(item["lane"] == "code" for item in first["selected"])
        self.assertGreaterEqual(selected_code, 1)
        self.assertIn("deferred", {item["disposition"] for item in first["decisions"]})


if __name__ == "__main__":
    unittest.main()
