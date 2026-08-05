# Training Data Execution System V5 - Evidence

Overall result: **PASS**

| Requirement | Result | Evidence |
|---|---:|---|
| Tokenizer integrity | PASS | `manifests/tokenizer.json`, `manifests/root.json`, `reports/source_report.json` |
| Evaluation firewall | PASS | `reports/firewall.json`, `ledgers/main.consumption.jsonl`, `ledgers/main.opus.jsonl` |
| Packing correctness | PASS | `batches/`, `ledgers/main.consumption.jsonl` |
| Mixture compliance | PASS | `reports/mixture_plan.json`, `ledgers/main.consumption.jsonl` |
| OPUS audit trail | PASS | `ledgers/main.opus.jsonl`, `reports/mixture_plan.json` |
| Crash recovery | PASS | `checkpoints/main.step-0002.json`, `reports/crash_expectation.json`, `ledgers/main.consumption.jsonl` |
| Replay | PASS | `reports/replay.json`, `ledgers/replay.consumption.jsonl` |
| Learning trace | PASS | `ledgers/main.consumption.jsonl`, `ledgers/main.learning.jsonl` |
| Branch fork | PASS | `reports/fork.json`, `checkpoints/fork.bootstrap.json`, `ledgers/fork-temperature-mixture.consumption.jsonl` |
| Throughput | PASS | `performance.json`, `ledgers/main.consumption.jsonl` |

This file is rendered from `evidence.json`, whose results are produced by the
independent artifact audit in `reports/audit.json`.
