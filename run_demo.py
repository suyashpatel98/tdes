from __future__ import annotations

from pathlib import Path

from tdes.demo import run_complete_demo


def main() -> None:
    root = Path(__file__).resolve().parent
    artifacts = run_complete_demo(root)
    print(f"Training Data Execution System demo: PASS")
    print(f"Artifacts: {artifacts}")
    print(f"Evidence: {artifacts / 'evidence.md'}")


if __name__ == "__main__":
    main()
