# Training Data Execution System V5 - Implementation Plan

## 1. Goal and completion criteria

Build a small, real, deterministic training system whose primary product is evidence about
the data path, not model quality. One command will regenerate all artifacts and exercise:

```text
documents -> immutable tokenized shards -> manifests -> staged mixture
-> OPUS candidate selection -> packing -> global batches -> model updates
-> consumption/learning ledgers -> checkpoint -> real worker-process crash
-> resume -> replay -> branch fork -> independent audit
```

The implementation is complete when `python3 run_demo.py` exits successfully after:

1. generating a fresh `submission_artifacts/` tree;
2. training a small language model on actually packed data;
3. crashing a worker process after an atomic checkpoint;
4. resuming in a new process and matching the independently recorded next batch;
5. reconstructing an earlier interval from an earlier checkpoint and matching every batch;
6. forking a new ledger branch from an earlier checkpoint;
7. deriving all PASS/FAIL evidence from artifact validators; and
8. passing focused automated tests for the important invariants.

No evidence result will be a hardcoded `PASS`. Each result will be emitted from an audit
function that reads the generated artifacts and records concrete supporting references.

## 2. Proposed technology and scope

- Python 3.11+ and the standard library only, so the demonstration needs no package install.
- A deterministic frozen tokenizer and a tiny causal context language model implemented in
  Python. The model will perform real cross-entropy training with AdamW state; it will be
  intentionally small so the complete demonstration remains fast.
- JSON, canonical JSONL, and content-addressed files for inspectable machine-readable
  artifacts. SHA-256 will be used for source, tokenizer, shard, batch, config, code, ledger,
  checkpoint, and model-state identities.
- `unittest` for automated tests, avoiding a test-runner dependency.
- Fixed logical seeds with every RNG state persisted. Wall-clock timing will be measured but
  will never affect scheduling, selection, hashes, or evidence outcomes.

The tiny model will consume the packer's causal attention visibility and position IDs rather
than merely generating unused mask arrays. It will use the visible prior tokens as context,
add a learned position bias, predict the next token, and update only loss-bearing positions.
This keeps the math auditable while proving the mask and position metadata are operational.

## 3. Repository layout

```text
run_demo.py                       one-command parent orchestrator
README.md                         architecture, commands, invariants, artifact guide
configs/demo.json                 curriculum, lanes, OPUS, packing, and training config
corpus/documents.jsonl            small versioned source corpus with split/type/lane metadata
tdes/
  canonical.py                    canonical JSON, hashing, file identity, atomic writes
  tokenizer.py                    frozen tokenizer and integrity checks
  shards.py                       tokenization, content-addressed shards, manifests
  firewall.py                     allowed-use enforcement for train/proxy/validation/eval
  mixture.py                      stages, lane quotas, weights, floors, candidate stream
  packing.py                      type-aware packing, masks, positions, provenance spans
  model.py                        causal model, cross entropy, sample gradients, AdamW
  opus.py                         paper-based projected utility selection and decisions
  ledgers.py                      append-only hash-chained ledgers and branch metadata
  checkpoint.py                   atomic checkpoints and recovery reconciliation
  replay.py                       historical reconstruction and branch forking
  audit.py                        independent artifact and invariant validators
  evidence.py                     evidence.json/evidence.md generation from audit results
  worker.py                       fresh-run, crash, resume, replay, and fork worker modes
  demo.py                         corpus-to-audit orchestration
tests/
  test_tokenizer_shards.py
  test_packing.py
  test_mixture_opus.py
  test_firewall.py
  test_ledgers_checkpoint.py
  test_resume_replay.py
  test_demo_evidence.py
2602.05400v2.pdf                  assignment reference for OPUS
```

Generated output will include the required structure and a few additional inspectable
directories:

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

Generation will occur in a staging directory. A fully audited run will replace the previous
artifact directory, preventing a failed run from being mistaken for fresh evidence.

## 4. Source data, tokenizer, shards, and manifests

### Source documents

The checked-in corpus will be small but contain enough records to exercise:

- multiple training lanes, such as `general`, `code`, and `instruction`;
- `document` and `prompt_completion` packing policies;
- curriculum transitions and a low-weight lane with a protected floor;
- a proxy pool used only for OPUS scoring;
- validation and evaluation records that must never become training candidates; and
- deterministic deferral and eventual rejection cases in the OPUS lifecycle.

Each source record will have a stable document ID, lane, split, data type, text fields, and
source content hash. The demo will make an explicit attempted eval-shard injection and prove
that it is blocked before candidate selection and packing.

### Frozen tokenizer

Use a checked, canonical tokenizer specification with fixed special IDs (`PAD`, `UNK`,
`BOS`, `EOS`, prompt/response delimiters) and a fixed token vocabulary. Its canonical
SHA-256 will be embedded in every shard manifest, batch descriptor, checkpoint, and run
identity. The tokenizer is never fit or extended during execution.

Audits will independently:

- recompute the tokenizer hash;
- ensure every manifest uses that hash;
- re-tokenize source records and compare token IDs; and
- prove that a changed tokenizer or shard byte causes validation failure.

### Immutable tokenized shards

Records will be partitioned by allowed use, lane, and data type. Shard filenames will contain
their SHA-256 and their canonical bytes will never be modified in place. After creation they
will be opened read-only; logical immutability is enforced by content hash regardless of file
mode portability.

Each manifest will record schema version, shard ID/path/hash/size, ordered record IDs,
record count, token count, tokenizer hash, source aggregate hash, split, allowed uses, lane,
data type, creation run ID, and manifest hash. A root manifest index will hash the ordered
set of manifests.

## 5. Firewall model

Every manifest will declare one of these roles:

- `train`: eligible for candidate generation and optimizer updates;
- `proxy`: eligible only for OPUS proxy-gradient calculation;
- `validation`: eligible only for validation metrics;
- `eval`: eligible only for final evaluation.

The firewall will be enforced at manifest loading, candidate construction, packing, and
immediately before the optimizer update. Proxy losses may be computed for scoring, but
proxy/eval/validation tokens must have zero update eligibility and may not appear in the
consumption ledger. Block events will include the rejected shard, role, requested use, and
reason. This yields both prevention and auditable proof.

## 6. Curriculum, mixture, lanes, and protected floors

The config will define at least two optimizer-step stages with different lane weights. A
deterministic weighted-deficit scheduler will compile candidate-buffer quotas for each stage.
Integer apportionment will use a documented stable tie-breaker, making the compiled stream
independent of dictionary order or worker count.

For every stage the compiled plan will record:

- start/end optimizer steps;
- lane weights and minimum protected shares;
- candidate slots and update slots;
- per-lane planned counts/tokens; and
- the seed and scheduler-state hash.

Protected floors apply to the selected update batch, not just the candidate buffer. After
normal OPUS selection, a deterministic reconciliation pass will detect an under-floor lane,
evict the lowest-marginal selected item from an over-floor lane, and admit the best eligible
protected-lane item. Both decisions will be retained and marked as one linked
`protected_floor_override` event. The audit will compare planned, selected, and consumed
shares per stage and prove every floor was met.

## 7. OPUS implementation

OPUS means **Optimizer-induced Projected Utility Selection**, following Algorithm 1 and
Equations 23-26 in `2602.05400v2.pdf`.

At optimizer step `t`:

1. Read a deterministic candidate buffer `B_t` of `N` packed samples.
2. Draw a deterministic proxy mini-batch from the firewalled proxy pool.
3. Freeze the AdamW optimizer geometry at the start of the step and construct its diagonal
   preconditioner `P_t` from the saved second-moment state.
4. Compute genuine per-sample gradient factors from the tiny model. The main weight
   gradient is an outer product of context activations and logit error, providing the paper's
   ghost-gradient factorization without materializing a candidate-by-parameter tensor.
5. Apply a seeded CountSketch projection to the optimizer-preconditioned candidate update
   and to the unpreconditioned mean proxy gradient.
6. Sequentially score remaining candidates using:

   ```text
   utility(z) = eta * <phi(z), proxy_sketch>
                - eta^2 * <phi(z), selected_history>
   ```

7. Select `K = floor(rho * N)` without replacement using numerically stable Boltzmann
   probabilities `softmax(utility / temperature)` and a persisted deterministic RNG.
8. Reconcile protected floors, then train only on the final selected samples.

The OPUS ledger will preserve candidate buffer ID, sample/batch ID, model and optimizer
state hashes, proxy IDs/hash, sketch seed/dimension, utility components, temperature,
probability, RNG draw, selection rank, lane, prior deferral count, final disposition, reason,
and any linked floor override.

### Assignment disposition lifecycle

The paper defines selected and unselected candidates, but the assignment additionally asks
for acceptance, rejection, deferral, and protected-floor override. The proposed explicit
stream lifecycle is:

- `accepted`: selected by the paper's Boltzmann loop;
- `deferred`: unselected, still within a configured retry budget, and returned to its lane's
  deterministic deferred queue;
- `rejected`: unselected after exhausting that retry budget, permanently retired with its
  final score history retained; and
- `protected_floor_override`: not selected globally, but admitted by the floor reconciliation
  rule, linked to the displaced candidate.

This lifecycle wraps OPUS without changing its utility calculation. The demo corpus and
seeds will be selected during development to naturally produce every disposition; the audit
will derive their presence from decision records rather than fabricate events.

## 8. Packing and batch correctness

The packer will emit fixed-length sequences and preserve token-to-source provenance.

Packing policies:

- `document`: `BOS + body + EOS`; all valid next-token targets bear loss;
- `prompt_completion`: prompt and response delimiters are retained, but only response and
  terminal targets bear loss; prompt targets are masked out.

Multiple short samples may share a physical sequence. Each segment gets:

- a block-causal attention mask that cannot see padding or another packed segment;
- position IDs reset to zero at its boundary;
- labels shifted only within the segment, never across boundaries;
- a loss mask consistent with its data-type policy; and
- source spans containing shard hash, record ID, source token start/end, packed start/end,
  and segment hash.

The batch descriptor will contain fixed-shape input IDs, labels, loss masks, attention masks,
position IDs, segment IDs, source spans, lane/type metadata, and aggregate hashes. The batch
ID is the SHA-256 of canonical semantic content, so hashes do not depend on timestamps or
file paths.

Packing tests and audits will prove causal visibility, segment isolation, padding isolation,
position resets, label shifting, prompt masking, source-span completeness, and batch-hash
stability. Reports will expose capacity tokens, non-padding tokens, loss-bearing tokens,
padding tokens, and utilization.

## 9. Real training and learning trace

For each selected global batch the model will perform a real forward pass, masked
cross-entropy calculation, backward gradient calculation, AdamW update, and state hash.
There is one process and one logical GPU in the demo, but batch metadata will still explicitly
record microbatch size, world size, gradient-accumulation steps, and the resulting global
batch size.

Sample-level loss will be calculated by aggregating token losses through segment/source-span
metadata. This makes each learning record traceable to exact source tokens while keeping the
ledger small. A learning row will record batch ID/hash, optimizer step, model before/after
hashes, source span, loss-bearing token count, summed/mean loss, lane, data type, and linked
consumption row hash.

The audit will recompute aggregate batch loss from sample records and prove that masked,
padding, proxy, validation, and eval tokens contributed no optimizer loss.

## 10. Ledgers and branch lineage

All ledgers will use canonical JSONL with monotonically increasing offsets and a SHA-256
hash chain (`previous_row_hash`, `row_hash`). At minimum:

- `consumption.jsonl`: one committed optimizer-update row with batch identity, ordered
  source spans, token counts, manifest/tokenizer/config hashes, branch, and step;
- `learning.jsonl`: sample-level actual losses linked to consumption and source spans;
- `opus.jsonl`: every candidate decision and protected-floor reconciliation;
- `events.jsonl`: checkpoint, crash, resume, replay, fork, firewall, and audit events; and
- `branches.json`: immutable parent checkpoint/branch/offset and explicit config delta.

Duplicate `(branch_id, optimizer_step)` and duplicate committed batch IDs on the same branch
will be rejected. Ledger tail hashes and exact byte/row offsets will be placed in checkpoints.
The audit will validate schemas, hash chains, cross-ledger references, monotonicity, unique
commits, and branch ancestry.

## 11. Atomic checkpoints, deliberate crash, and resume

Each checkpoint will be written to a temporary file, flushed and fsynced, atomically renamed,
then hashed. It will contain:

- model parameters and AdamW moments/step;
- model-before/after identity and optimizer-state hash;
- curriculum stage and deterministic lane/deferred-queue cursors;
- packer state and partially consumed record offsets, if any;
- OPUS CountSketch seed, Boltzmann RNG state, proxy cursor, and disposition state;
- tokenizer, root-manifest, code, config, and run hashes;
- branch ID, next optimizer step, and exact ledger offsets/tail hashes.

The parent `run_demo.py` process will launch a training worker. After a configured committed
step, the worker will:

1. save and validate the checkpoint;
2. load a separate in-memory preview from that persisted checkpoint and derive the expected
   next batch descriptor without committing it;
3. atomically write the expected next batch ID/hash/spans;
4. flush the log; and
5. terminate itself with `os._exit()` and a dedicated expected crash code.

The parent will require that exact crash exit code, then launch a new resume worker. The new
process will load only persisted state, reconcile/truncate any ledger suffix beyond checkpoint
offsets, construct its next batch, and compare ID, full hash, and source spans with the preview
before committing. It will also prove the prior batch is not repeated and the expected next
step is not skipped.

Using separate OS processes prevents accidental reliance on surviving in-memory state.

## 12. Replay and fork

### Historical replay

Load an earlier checkpoint into an isolated replay branch, reconstruct several historical
steps using immutable shards/manifests plus restored model, optimizer, mixture, packer,
proxy, CountSketch, deferred-queue, and RNG state, and perform the same updates needed to
advance dynamic OPUS state. For every step compare reconstructed batch ID, batch hash,
ordered source spans, and span hashes with the original consumption ledger. Replay output
will be separate and cannot mutate the historical branch.

### Fork

Load the same earlier checkpoint under a new branch ID. Record its parent checkpoint,
parent ledger offsets/tails, fork reason, and a canonical config delta (proposed: change OPUS
temperature and one post-fork lane weight). Execute at least two fork updates while leaving
the parent ledgers/checkpoints unchanged. The audit will prove common ancestry, explicit
divergence, valid independent hash chains, and preservation of the parent history.

## 13. Performance and efficiency evidence

Use `time.perf_counter_ns()` around tokenization, packing, OPUS scoring, forward/backward,
and end-to-end training. `performance.json` will report raw durations and reconstructible
counters rather than only derived claims:

- batch capacity, source tokens, non-padding tokens, and loss-bearing tokens;
- packing utilization = non-padding tokens / capacity tokens;
- useful-token ratio = loss-bearing tokens / capacity tokens;
- packed tokens/second and useful loss-bearing tokens/second;
- update tokens/second and OPUS scoring overhead; and
- per-stage/per-lane totals.

The audit will recompute every ratio and token total from batch and ledger artifacts. A
performance PASS means counters reconcile, durations and rates are positive, and utilization
is within `[0, 1]`; it will not depend on a machine-specific speed threshold.

## 14. Audit and generated evidence

The final audit will read artifacts through fresh readers and produce structured results for:

- tokenizer and shard integrity;
- evaluation/validation/proxy firewall;
- packing, masks, labels, positions, and provenance;
- curriculum boundaries, mixture shares, and protected floors;
- OPUS math inputs, deterministic selection records, and all dispositions;
- consumption/learning ledger integrity and loss/source linkage;
- checkpoint/ledger offset agreement;
- exact resume next-batch match with no duplicate or skipped commit;
- exact historical replay match;
- fork ancestry and explicit configuration delta; and
- packing utilization and throughput reconciliation.

`evidence.json` will contain requirement ID, PASS/FAIL, measured values, audit function,
and artifact references (file plus row/record/hash). `evidence.md` will be rendered from that
JSON into the requested requirement/result/evidence table. The run will fail nonzero if any
mandatory audit fails.

`run.log` will be timestamped and contain the complete event order and required lines,
including:

```text
[PASS] tokenizer_hash_verified
[PASS] eval_shard_blocked
[PASS] checkpoint_saved
[PASS] resume_next_batch_matched
[PASS] replay_hash_matched
```

The remaining requested events (shards, manifests, mixture, packing, OPUS decisions, crash,
resume, fork, audit, and performance) will likewise be emitted by the component that actually
performed or verified them.

## 15. Automated test plan

Focused tests will use temporary directories and small fixtures:

1. Tokenizer/shards: stable tokenization; content-addressed manifest validation; tokenizer,
   source, shard, and manifest tamper detection.
2. Firewall: eval/validation/proxy injection blocked at every boundary; proxy scoring allowed
   but proxy optimizer updates denied.
3. Packing: exact expected IDs/labels/loss masks/attention blocks/positions/spans for document
   and prompt-completion samples; no cross-segment target or attention leakage.
4. Mixture: deterministic stage transitions, quota apportionment, protected-floor compliance,
   and planned-versus-consumed reconciliation.
5. OPUS: preconditioned projected utility against a direct tiny reference calculation;
   deterministic Boltzmann sampling; acceptance, deferral, rejection, and floor override.
6. Ledgers: valid chain/reference checks; rejection of mutation, broken tails, duplicates, and
   out-of-order steps.
7. Checkpoint/recovery: atomic round trip; restored model/optimizer/RNG/cursors; reconciliation
   of an uncommitted ledger suffix; exact next-batch match.
8. Replay/fork: historical interval equality and immutable parent lineage.
9. End to end: run a shortened demo, require all evidence checks, then independently
   cross-check evidence references against generated artifacts.

Commands documented in the README will be:

```bash
python3 run_demo.py
python3 -m unittest discover -s tests -v
```

## 16. Implementation sequence

1. Add project skeleton, canonical serialization/hashing, tokenizer, source fixtures, shard
   writer, manifests, and integrity/firewall tests.
2. Implement type-aware packing and provenance with exact mask/position/label tests.
3. Implement curriculum mixture compilation, candidate queues, floors, and reports.
4. Implement the tiny model, masked learning, AdamW state, and per-sample loss traces.
5. Implement paper-based OPUS scoring, CountSketch, seeded Boltzmann selection, lifecycle,
   protected-floor reconciliation, and reference-math tests.
6. Implement hash-chained ledgers and atomic checkpoints with cross-reference audits.
7. Implement multi-process crash orchestration, resume reconciliation, historical replay, and
   explicit branch fork.
8. Implement performance collection, independent audit, evidence generation, and full log.
9. Run all unit/integration tests and the complete demo repeatedly; compare deterministic
   artifact hashes while excluding timestamps/timings from semantic identities.
10. Write the README, inspect the final artifact tree, and verify every evidence reference by
    reading its target artifact.

## 17. Decisions requested before implementation

Approval of this plan will be treated as approval of the following defaults unless you request
changes:

1. **OPUS lifecycle extension:** use paper-faithful utility and Boltzmann selection, with the
   retry-budget defer/reject lifecycle and post-selection protected-floor reconciliation
   described in Section 7. The paper itself only defines selected versus unselected samples;
   the extra states are an explicit assignment-facing policy, not claimed as part of the paper.
2. **Dependencies:** use standard-library-only Python for a one-command run on the current
   environment, rather than requiring PyTorch/NumPy. The model will be genuinely trained but
   intentionally small.
3. **Crash strength:** use a child process terminated by `os._exit()` and a clean resume child,
   rather than catching a simulated exception in one process.
4. **Fork delta:** change OPUS temperature and a future-stage lane weight after the fork so the
   divergence is deliberate and fully recorded.
5. **Repository PDF:** retain `2602.05400v2.pdf` as a local design reference unless it should be
   excluded from the eventual repository because of size or licensing.

No implementation beyond this plan will begin until the plan is approved.
