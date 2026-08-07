from __future__ import annotations

import json
import subprocess
import sys
import unittest
from pathlib import Path

from tdes.canonical import hash_object


ROOT = Path(__file__).resolve().parents[1]


class EndToEndDemoTests(unittest.TestCase):
    def test_complete_demo_regenerates_passing_evidence(self) -> None:
        result = subprocess.run(
            [sys.executable, "run_demo.py"],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        artifacts = ROOT / "submission_artifacts"
        evidence = json.loads((artifacts / "evidence.json").read_text())
        audit = json.loads((artifacts / "reports" / "audit.json").read_text())
        self.assertEqual(evidence["overall_result"], "PASS")
        self.assertTrue(audit["overall_passed"])
        self.assertTrue(all(item["passed"] for item in audit["checks"].values()))
        mixture = json.loads(
            (artifacts / "reports" / "mixture_compliance.json").read_text()
        )
        mixture_body = {
            key: value
            for key, value in mixture.items()
            if key != "mixture_compliance_hash"
        }
        self.assertEqual(
            mixture["mixture_compliance_hash"], hash_object(mixture_body)
        )
        for stage in mixture["stages"].values():
            self.assertEqual(
                stage["planned_candidate_counts"], stage["actual_candidate_counts"]
            )
        log = (artifacts / "run.log").read_text()
        for marker in (
            "[PASS] checkpoint_saved",
            "[PASS] crash_simulated",
            "[PASS] resume_next_batch_matched",
            "[PASS] replay_hash_matched",
            "[PASS] branch_forked",
        ):
            self.assertIn(marker, log)
        ordered_markers = (
            "[PASS] shards_created",
            "[PASS] manifests_validated",
            "[PASS] eval_shard_blocked",
            "[PASS] mixture_compiled",
            "[PASS] batches_packed",
            "[PASS] opus_decisions_recorded",
            "[PASS] checkpoint_saved",
            "[PASS] crash_simulated",
            "[PASS] run_resumed",
            "[PASS] replay_hash_matched",
            "[PASS] branch_forked",
            "[PASS] audit_completed",
            "[PASS] performance_measured",
        )
        positions = [log.index(marker) for marker in ordered_markers]
        self.assertEqual(positions, sorted(positions))


if __name__ == "__main__":
    unittest.main()
