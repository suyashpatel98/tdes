# Training Data Execution System V5

A small, complete training-data execution system focused on correctness, reproducibility,
auditability, and measurable data efficiency.

## Run

Python 3.11 or newer is required. The implementation uses only the standard library.

```bash
python3 run_demo.py
```

This command cleans the fixed corpus snapshot, rebuilds every artifact, launches a training
worker, deliberately kills it after a checkpoint, resumes in a new worker, replays an earlier
interval, forks a new branch, and independently audits the result. It should finish in roughly
10 seconds on a typical laptop.

Run the automated invariants with:

```bash
python3 -m unittest discover -s tests -v
```

## Architecture

```text
vendored raw sources
  -> deterministic cleaning and role assignment
  -> frozen UTF-8 byte tokenizer
  -> content-addressed immutable JSONL shards and manifests
  -> staged lane mixture and protected floors
  -> type-aware fixed-length packing
  -> OPUS candidate scoring and selection
  -> global batch and real AdamW model update
  -> hash-chained consumption, learning, OPUS, and event ledgers
  -> atomic checkpoint
  -> process crash and exact resume
  -> historical replay and explicit branch fork
  -> independent audit and generated evidence
```

The main components are under `tdes/`:

- `corpus.py`, `tokenizer.py`, and `shards.py` build and verify the immutable data layer.
- `packing.py` creates block-causal masks, position IDs, shifted labels, loss masks, and exact
  source-token spans for document and prompt-completion records.
- `mixture.py` compiles curriculum stages, deterministic lane quotas, and protected floors.
- `model.py` implements a small causal context language model with learned position weights,
  masked cross entropy, per-sample gradients, and AdamW optimizer state.
- `opus.py` implements Optimizer-induced Projected Utility Selection.
- `ledgers.py` and `checkpoint.py` provide hash chains, atomic persistence, and offset-based
  recovery.
- `worker.py` contains isolated fresh, resume, replay, and fork worker modes.
- `audit.py` independently reconstructs all reported claims.

## Model

The demo trains a small autoregressive causal context model from scratch. At each
loss-bearing position it aggregates only tokens visible through the packer's block-causal
attention mask, adds a learned position bias, and predicts the next byte token. Training uses
real cross-entropy gradients and AdamW updates.

This is intentionally not a Transformer. The assignment concerns the data execution path,
and the compact model makes every gradient, loss attribution, optimizer state, and OPUS
calculation inspectable without external dependencies.

The selected global batch has three sequences and is represented as:

```text
microbatch size 1 * world size 1 * 3 accumulation steps = global batch size 3
```

The implementation combines those three microbatch gradients before one optimizer update,
which is mathematically equivalent to gradient accumulation for this single-process demo.

## Corpus

The repository vendors a 13 MB fixed raw snapshot, below the assignment's size limits:

- *Alice's Adventures in Wonderland* from Project Gutenberg for general text and the
  proxy/validation/evaluation partitions;
- CPython 3.13 `Lib/statistics.py` for the code lane; and
- Databricks Dolly 15k for the prompt-completion lane.

`corpus/SOURCES.json` records exact URLs, licenses, byte counts, and SHA-256 hashes. License
texts or metadata are stored alongside the sources. The demo refuses to run if any raw file
does not match its recorded size and hash.

Cleaning is deterministic: Unicode is normalized, whitespace is canonicalized, bounded
records are selected by stable source positions, and train/proxy/validation/eval roles are
assigned before tokenization. The generated demonstration uses 82 records. The partitions
contain distinct source records and their roles are carried into separate manifests.

## Packing

Two policies are implemented:

- `document`: `BOS + text + EOS`; every valid next-token target bears loss.
- `prompt_completion`: `BOS + PROMPT + prompt + RESPONSE + response + EOS`; prompt targets
  bear no loss, while response and terminal targets do.

Short records can share one physical sequence. Attention is causal only inside each packed
segment, position IDs restart at segment boundaries, padding cannot be attended to, and labels
never cross a boundary. Every segment stores its shard, record, field-token ranges, packed
ranges, occurrence, truncation state, and hash.

## Curriculum and OPUS

The six-step demo has `foundation` and `reasoning` curriculum stages with different weights
for `general`, `instruction`, and `code`. Stable integer apportionment produces each candidate
buffer. The code lane has a protected selected-batch floor.

OPUS follows Algorithm 1 and Equations 23-26 of `2602.05400v2.pdf`:

1. Compute genuine per-candidate and proxy gradients.
2. Freeze the AdamW diagonal preconditioner at the start of the step.
3. Project optimizer-preconditioned candidate updates and the mean proxy gradient through a
   deterministic CountSketch.
4. Score proxy alignment minus selected-history redundancy.
5. Select sequentially without replacement using seeded Boltzmann probabilities.

The assignment-specific lifecycle wraps the paper's selected/unselected result:

- selected candidates are `accepted`;
- unselected candidates are `deferred` while retry budget remains;
- candidates exhausting that budget are `rejected`; and
- a deterministic swap is recorded as `protected_floor_override` when needed to satisfy a
  lane floor.

Every utility component, probability, RNG draw, proxy identity, model/optimizer hash,
disposition, and floor swap is recorded. The generated run naturally contains all four
dispositions.

## Firewalls

Each shard manifest has one role and an allowed-use list:

| Role | Allowed use |
|---|---|
| train | optimizer candidates and updates |
| proxy | OPUS gradient scoring only |
| validation | validation only |
| eval | evaluation only |

The demo actively attempts to submit proxy, validation, and eval shards for training and
requires each attempt to fail. The final audit also proves that no such shard or proxy
candidate appears in the consumption ledger.

## Ledgers and checkpoints

All JSONL ledgers have contiguous offsets and SHA-256 chains. Consumption rows identify the
exact batch and source spans. Learning rows contain actual sample-level loss and link back to
the consumption-row hash. OPUS rows preserve candidate decisions. Event rows preserve run
state transitions.

Checkpoints include:

- sparse model parameters and AdamW moments;
- mixture, packer, proxy, deferred-queue, CountSketch, and Boltzmann RNG state;
- the next optimizer step;
- exact row/byte offsets and tail hashes for every ledger; and
- tokenizer, root-manifest, config, code, branch, and parent identities.

Writes use flush, `fsync`, and atomic rename. On recovery, ledgers are truncated to the
checkpoint's committed byte offsets before state is restored.

## Crash, replay, and fork proof

After main step 2, the worker atomically saves its checkpoint. A separate engine loaded from
that checkpoint previews the expected next batch and persists its ID, full hash, candidate
hashes, and source spans. The training worker then calls `os._exit(86)`.

The parent requires exit code 86 and launches a fresh process. That process restores only
persisted state and must reproduce the preview exactly before committing step 3. It also
checks that the preceding batch was not repeated.

Replay starts from the bootstrap checkpoint and reconstructs steps 0-3, including model
updates needed for dynamic OPUS. Batch IDs, full hashes, candidate hashes, and ordered source
spans are compared with the historical main ledger.

The fork starts from the checkpoint after main step 0. It records a new OPUS temperature and
lane mixture as an explicit config delta, runs two updates on a new hash chain, proves stream
divergence, and verifies that the parent consumption ledger is byte-for-byte unchanged.

## Generated artifacts

`python3 run_demo.py` replaces `submission_artifacts/` only after the fresh build passes:

```text
submission_artifacts/
  run.log
  evidence.json
  evidence.md
  manifests/
  shards/
  batches/
  ledgers/
  checkpoints/
  reports/
  performance.json
```

`performance.json` contains raw durations and token counters plus reconstructible packing
utilization and useful loss-bearing tokens per second. `reports/audit.json` records the
independent validators. `evidence.json` is built from those validator results, and
`evidence.md` is rendered from the JSON bundle. A mandatory audit failure makes the command
exit nonzero.

## Important invariants

- Semantic identities exclude timestamps and wall-clock timings.
- Every tokenizer, shard, manifest, batch, ledger row, checkpoint, report, and evidence bundle
  has a recomputable SHA-256 identity.
- Only `train` shards can reach an optimizer update.
- Padding and prompt-only tokens never contribute to loss.
- Packed segments cannot attend to or predict across boundaries.
- A committed `(branch, step)` is unique and steps are contiguous.
- Checkpoint ledger tails are prefixes of the final ledgers.
- Resume and replay equality includes source spans, not only batch IDs.
- Performance ratios are recomputed from consumption rows during audit.
