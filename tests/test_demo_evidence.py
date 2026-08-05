from __future__ import annotations

import json
import subprocess
import sys
import unittest
from pathlib import Path


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
        log = (artifacts / "run.log").read_text()
        for marker in (
            "[PASS] checkpoint_saved",
            "[PASS] crash_simulated",
            "[PASS] resume_next_batch_matched",
            "[PASS] replay_hash_matched",
            "[PASS] branch_forked",
        ):
            self.assertIn(marker, log)


if __name__ == "__main__":
    unittest.main()
