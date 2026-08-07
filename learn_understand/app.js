const P = (html) => ({ type: "p", html });
const CODE = (text) => ({ type: "code", text });
const NOTE = (html) => ({ type: "note", html });
const PROOF = (html) => ({ type: "proof", html });
const LINK = (label, href) => ({ type: "link", label, href });
const NODE = (title, preview, content = [], children = []) => ({ title, preview, content, children });

const stages = [
  {
    id: "assignment",
    title: "The assignment",
    shortTitle: "Assignment",
    category: "Orientation",
    icon: "clipboard-check",
    summary: "Build a small training system that can prove exactly what data it used, how the model learned from it, and whether a crashed run can be reconstructed.",
    takeaway: "The product is not just a trained model. The product is a trained model plus proof of its complete data history.",
    flow: ["data", "training", "ledgers", "proof"],
    details: [
      NODE("What are we building?", "A complete, inspectable path from documents to an audited training run.", [
        P("The assignment asks for a <strong>Training Data Execution System</strong>. Its job is to turn source documents into model updates while preserving enough identity and state to explain every update later."),
        CODE("documents → tokenized shards → manifests → mixture → packing\n→ batches → training → ledgers → checkpoint → crash\n→ resume → replay → audit"),
        NOTE("Model quality is not the main objective. Correctness, reproducibility, auditability, recovery, and measurable efficiency are the objectives.")
      ], [
        NODE("The data side", "Documents become immutable tokens, scheduled samples, and packed sequences.", [
          P("The system must answer: <strong>Which source text became which tokens?</strong> Which lane supplied it? Which packing rule was used? Which batch finally consumed it?")
        ]),
        NODE("The training side", "Selected batches produce real loss, gradients, optimizer updates, and checkpoints.", [
          P("The demonstration trains a real small model. Each update has a model-before hash, model-after hash, optimizer state, exact batch identity, and source-level loss records.")
        ]),
        NODE("The proof side", "Independent validators regenerate claims instead of trusting log messages.", [
          P("A requirement passes only when an audit reads the generated artifacts and verifies it. A literal hardcoded <code>PASS</code> is not evidence.")
        ])
      ]),
      NODE("What defines one experiment?", "Weights alone do not determine what happens after restoration.", [
        CODE("experiment = model state + optimizer state + data-stream state + code/config"),
        P("Changing any one of these silently weakens a comparison. Restoring old weights with a different sample order is a different experiment, even if the filename says it is the same checkpoint.")
      ], [
        NODE("Model and optimizer", "Parameters say what the model knows; optimizer moments affect the next update.", [
          P("AdamW stores moving averages in addition to model weights. Two runs with identical weights but different AdamW moments can produce different next updates.")
        ]),
        NODE("Data-stream state", "Lane cursors, deferred candidates, proxy position, and RNG position determine the next batch.", [
          P("The next batch is a consequence of saved stream state. If that state is missing, documents can be repeated, skipped, or selected differently.")
        ]),
        NODE("Code and configuration", "Packing rules, mixture weights, and selection temperature are part of the experiment.", [
          P("The checkpoint records code and configuration hashes. Restoration rejects an unexplained mismatch on the historical branch.")
        ])
      ]),
      NODE("What must be submitted?", "One command regenerates both the run and the evidence about it.", [
        CODE("python3 run_demo.py"),
        P("The command creates logs, evidence, manifests, shards, packed batches, ledgers, checkpoints, replay and fork reports, and a performance report."),
        PROOF("The finished demo reports ten independent audit groups as PASS and the automated suite contains fourteen invariant tests."),
        LINK("Open generated evidence", "../submission_artifacts/evidence.md")
      ])
    ]
  },
  {
    id: "design",
    title: "Core design choices",
    shortTitle: "Design",
    category: "Orientation",
    icon: "drafting-compass",
    summary: "Keep every component small enough to inspect, but real enough that loss, gradients, optimizer state, selection, recovery, and replay are genuine.",
    takeaway: "Small is a scope choice, not a simulation: the model trains and every data-system invariant is enforced.",
    flow: ["standard Python", "small model", "fixed corpus", "deterministic run"],
    details: [
      NODE("Why standard-library Python?", "The entire demonstration runs without installing a machine-learning framework.", [
        P("The current environment did not include NumPy, PyTorch, or pytest. A standard-library implementation keeps <code>python3 run_demo.py</code> portable and removes network installation as a failure mode."),
        P("Canonical JSON, SHA-256, atomic file operations, subprocess workers, sparse model parameters, and <code>unittest</code> are all available in Python itself.")
      ]),
      NODE("What model is trained?", "A compact causal context language model predicts the next byte token.", [
        P("At each loss-bearing position, the model aggregates only tokens visible through the block-causal attention mask, adds a learned position bias, and predicts the next token."),
        P("Training uses real softmax cross-entropy, sparse gradients, and AdamW moments. It is intentionally not a Transformer because the assignment evaluates the data execution path rather than benchmark quality."),
        LINK("Inspect the model", "../tdes/model.py")
      ], [
        NODE("Why the model is sufficient", "It exercises every contract that the data system must satisfy.", [
          P("The model consumes attention masks and position IDs, respects loss masks, emits sample-level losses, supplies per-candidate gradients to OPUS, and restores parameters plus AdamW state from checkpoints.")
        ])
      ]),
      NODE("How is determinism achieved?", "State reproduces the stream; hashes prove the reproduction matched.", [
        P("Saved cursors and random state cause the same next operations. Semantic hashes then verify that tokens, masks, source spans, and ordering are identical."),
        NOTE("A hash does not make an operation deterministic. It detects whether deterministic reconstruction succeeded.")
      ])
    ]
  },
  {
    id: "documents",
    title: "Documents and corpus",
    shortTitle: "Documents",
    category: "Data foundation",
    icon: "files",
    summary: "Verify a small fixed corpus, clean it deterministically, and assign each record an identity, lane, type, and permitted role before tokenization.",
    takeaway: "A reproducible stream starts with immutable source bytes and stable document identities.",
    flow: ["13 MB raw", "verify hashes", "clean", "82 records"],
    details: [
      NODE("Which sources are used?", "Three small public datasets provide prose, code, and instruction data.", [
        P("The vendored snapshot contains Project Gutenberg's <em>Alice's Adventures in Wonderland</em>, CPython's <code>statistics.py</code>, and Databricks Dolly 15k."),
        P("The raw files total 13,343,489 bytes. Runtime does not depend on downloading them again."),
        LINK("Open source manifest", "../corpus/SOURCES.json")
      ], [
        NODE("General lane", "Short sentence-like records come from Alice in Wonderland.", [P("The cleaner removes Gutenberg framing, normalizes text, filters stable sentence positions, and allocates distinct records across train, proxy, validation, and eval roles.")]),
        NODE("Code lane", "Eligible logical lines come from CPython statistics.py.", [P("Stable source-order filtering keeps compact lines containing structures such as <code>def</code>, <code>return</code>, <code>if</code>, <code>for</code>, assignment, or <code>raise</code>.")]),
        NODE("Instruction lane", "Short prompt-response examples come from Dolly.", [P("Records with no external context and bounded prompt/response byte lengths are selected in stable source order. Their original category remains metadata.")])
      ]),
      NODE("How are source bytes verified?", "Every raw file has a recorded size and SHA-256 identity.", [
        P("Before cleaning, the system recalculates every raw file's byte count and hash and compares both with <code>corpus/SOURCES.json</code>. A mismatch stops execution."),
        PROOF("The source report records the verified files, total bytes, source-manifest hash, document count, role counts, lane counts, and aggregate documents hash."),
        LINK("Open source report", "../submission_artifacts/reports/source_report.json")
      ]),
      NODE("What does one document record contain?", "Clean content stays linked to its exact upstream location.", [
        CODE('{\n  "record_id": "dolly-train-000",\n  "role": "train",\n  "lane": "instruction",\n  "data_type": "prompt_completion",\n  "fields": {"prompt": "...", "response": "..."},\n  "source_id": "databricks_dolly_15k",\n  "source_index": 230,\n  "content_hash": "..."\n}'),
        P("The final set contains 66 training records, 8 proxy records, 4 validation records, and 4 evaluation records."),
        LINK("Open generated documents", "../submission_artifacts/reports/documents.jsonl")
      ])
    ]
  },
  {
    id: "tokenizer",
    title: "Frozen tokenizer",
    shortTitle: "Tokenizer",
    category: "Data foundation",
    icon: "binary",
    summary: "Convert cleaned UTF-8 bytes into fixed token IDs using a tokenizer that never learns, expands, or changes during the run.",
    takeaway: "The same text bytes always become the same token IDs, independent of corpus order or training state.",
    flow: ["text", "UTF-8 bytes", "+6 offset", "token IDs"],
    visual: "tokenizer",
    details: [
      NODE("What is tokenization?", "A model processes integer IDs rather than strings.", [
        P("The tokenizer turns text into a sequence of integers. For this system, each UTF-8 byte maps to exactly one ordinary token."),
        CODE("token ID = UTF-8 byte value + 6"),
        P("The six lower IDs are reserved for <code>PAD</code>, <code>BOS</code>, <code>EOS</code>, <code>PROMPT</code>, <code>RESPONSE</code>, and <code>UNK</code>. The total vocabulary size is 262.")
      ], [
        NODE("Why bytes instead of words?", "Every valid UTF-8 string is representable without fitting a vocabulary.", [P("A byte tokenizer has no unknown word problem, needs no training phase, and makes exact retokenization easy to audit. Unicode characters simply occupy multiple UTF-8 byte tokens.")]),
        NODE("What do special tokens do?", "They mark padding, sequence boundaries, and prompt/response structure.", [P("The packer inserts the special IDs according to each data type. They are part of the frozen tokenizer specification and therefore part of its hash.")])
      ]),
      NODE("What does frozen mean?", "No vocabulary rule can change after the experiment is defined.", [
        P("The tokenizer never learns new entries, reorders IDs, or depends on corpus statistics. Its canonical specification has a SHA-256 hash embedded in manifests, batches, checkpoints, and the run identity."),
        PROOF("During audit, all 82 source documents are tokenized again and compared with the stored token records."),
        LINK("Open tokenizer artifact", "../submission_artifacts/manifests/tokenizer.json")
      ])
    ]
  },
  {
    id: "shards",
    title: "Shards and manifests",
    shortTitle: "Shards",
    category: "Data foundation",
    icon: "database",
    summary: "Group tokenized records into content-addressed files, then describe every file with a manifest and one root dataset identity.",
    takeaway: "A root-manifest hash identifies the exact tokenizer, shards, record ordering, roles, and token counts used by the run.",
    flow: ["token records", "6 shards", "6 manifests", "root hash"],
    details: [
      NODE("What is a shard?", "A shard is an ordered JSONL file containing multiple tokenized records.", [
        P("Records are grouped by <strong>role + lane + data type</strong>. The demo produces three training shards and one each for proxy, validation, and evaluation data."),
        P("Each shard byte sequence receives SHA-256. The first part of that hash appears in its filename, making the file content-addressed."),
        CODE("train-general-document-901ec1be60f93306.jsonl"),
        LINK("Browse generated shards", "../submission_artifacts/shards/")
      ]),
      NODE("What is a manifest?", "A compact contract describes what a shard contains and how it may be used.", [
        P("A manifest stores shard path/hash/size, ordered record IDs and hashes, record and token counts, tokenizer hash, source aggregate hash, role, allowed uses, lane, data type, and its own manifest hash."),
        NOTE("The manifest lets a reader reject the wrong file or tokenizer before training consumes any record."),
        LINK("Browse generated manifests", "../submission_artifacts/manifests/")
      ], [
        NODE("Root manifest", "One higher-level hash commits to the ordered set of manifests.", [
          CODE("root hash\n  → manifest hashes\n    → shard hashes\n      → token-record hashes\n        → source-document hashes"),
          LINK("Open root manifest", "../submission_artifacts/manifests/root.json")
        ]),
        NODE("Immutability", "Read-only permissions discourage edits; hashes detect edits.", [P("Shard files are written read-only. Even if permissions are changed, a modified byte changes the shard hash and causes repository validation to fail.")])
      ]),
      NODE("What is validated before training?", "Every identity layer is recomputed, not trusted.", [
        P("Opening the repository verifies the root hash, tokenizer identity, each manifest, every shard byte hash, every token-record hash, and declared record ordering."),
        PROOF("The tamper test appends bytes to a shard and requires repository construction to raise an integrity error."),
        LINK("Inspect shard validation", "../tdes/shards.py")
      ])
    ]
  },
  {
    id: "firewall",
    title: "Data firewalls",
    shortTitle: "Firewall",
    category: "Data control",
    icon: "shield-check",
    summary: "Prevent proxy, validation, and evaluation records from entering loss-bearing training batches while still permitting their intended read-only uses.",
    takeaway: "A split label is not enough; authorization is checked at the boundary where data is requested and again before an update.",
    flow: ["manifest role", "requested use", "allow / block", "audit"],
    details: [
      NODE("What are the four roles?", "Every shard declares one purpose and a narrow allowed-use list.", [
        CODE("train      → model updates\nproxy      → OPUS scoring only\nvalidation → validation metrics only\neval       → final evaluation only"),
        P("Proxy loss can be calculated to produce a desired gradient direction, but proxy tokens cannot contribute to an optimizer update.")
      ]),
      NODE("Where is the firewall enforced?", "The system checks use when records are requested and immediately before training.", [
        P("The shard repository exposes explicit requested uses such as <code>train</code> or <code>score</code>. The engine also rechecks every selected source span before calling the model update."),
        NOTE("Defense at multiple boundaries prevents a later scheduler or selector bug from silently bypassing the split policy."),
        LINK("Inspect firewall enforcement", "../tdes/shards.py")
      ]),
      NODE("How is the firewall proven?", "The demo deliberately tries to train on all three protected roles.", [
        P("Eval, validation, and proxy shard injection attempts must each raise a firewall error. The final audit then scans the entire consumption ledger and proves every consumed shard has role <code>train</code>."),
        PROOF("The audit also intersects proxy candidate IDs with consumed candidate IDs and requires the intersection to be empty."),
        LINK("Open firewall report", "../submission_artifacts/reports/firewall.json")
      ])
    ]
  },
  {
    id: "curriculum",
    title: "Lanes and curriculum",
    shortTitle: "Curriculum",
    category: "Scheduling",
    icon: "route",
    summary: "Organize training sources into lanes, change their proportions across stages, and guarantee that a protected lane receives a minimum selected share.",
    takeaway: "A lane says what kind of data is next; a curriculum stage says how much of each lane should be considered now.",
    flow: ["stage", "lane weights", "6 candidate slots", "floor check"],
    visual: "mixture",
    details: [
      NODE("What is a lane?", "A lane is an independently ordered stream for one data category.", [
        CODE("general:     G1, G2, G3, ...\ncode:        C1, C2, C3, ...\ninstruction: I1, I2, I3, ..."),
        P("Each lane has a cursor. Reading candidates advances that lane's cursor, while unselected retryable candidates move to an explicit deferred queue.")
      ], [
        NODE("Why independent cursors matter", "A checkpoint must know the next new record in every lane.", [P("A single global index cannot describe three independently weighted streams. Saved lane cursors prevent silent skip or repetition after resume.")])
      ]),
      NODE("What is a curriculum stage?", "A stage is a range of optimizer steps sharing one mixture policy.", [
        CODE("steps 0–2  foundation: 55% general, 30% instruction, 15% code\nsteps 3–5  reasoning:  30% general, 35% instruction, 35% code"),
        P("Stable integer apportionment converts percentages into six concrete candidate slots. Ties use deterministic ordering."),
        LINK("Open compiled mixture plan", "../submission_artifacts/reports/mixture_plan.json")
      ]),
      NODE("What is a protected floor?", "A minimum selected share prevents OPUS from starving an important lane.", [
        P("The code lane must contribute at least one of the three selected candidates. If normal OPUS selection contains no code candidate, reconciliation admits the best eligible code candidate and evicts the weakest candidate from an overrepresented lane."),
        PROOF("The generated run naturally records one <code>protected_floor_override</code> at step 2. Both the admitted and displaced candidate records reference the same event."),
        LINK("Inspect curriculum logic", "../tdes/mixture.py")
      ])
    ]
  },
  {
    id: "packing",
    title: "Packing and masks",
    shortTitle: "Packing",
    category: "Batch construction",
    icon: "layout-grid",
    summary: "Fit short records into fixed-length sequences without allowing attention, labels, positions, or loss to leak across record boundaries.",
    takeaway: "Packing saves empty space, but every token must still behave as if its source sample were isolated.",
    flow: ["records", "segments", "64-token sequence", "masks + spans"],
    visual: "packing",
    details: [
      NODE("Why pack records?", "Models train on fixed shapes even when source examples have different lengths.", [
        P("A physical sequence has 64 token positions. Placing multiple short samples into one sequence reduces padding and increases useful work per batch."),
        P("The final main run packs 873 non-padding tokens into 1,152 available positions, giving roughly 75.8% utilization.")
      ]),
      NODE("Which packing policies exist?", "Documents and prompt-response records have different loss rules.", [
        CODE("document:\n  BOS + body + EOS\n  all valid next-token targets bear loss\n\nprompt_completion:\n  BOS + PROMPT + prompt + RESPONSE + response + EOS\n  only response and EOS targets bear loss"),
        P("Long fields are deterministically truncated while preserving delimiters and some response content.")
      ], [
        NODE("Attention mask", "A token can see only earlier or current tokens inside its own segment.", [P("The mask is block-causal. A row in segment B cannot attend to any column from segment A, even though both occupy one physical sequence.")]),
        NODE("Position IDs", "Positions restart from zero at every segment boundary.", [P("Reset positions preserve per-sample semantics and make the second packed record start at position zero rather than continuing after the first.")]),
        NODE("Labels and loss mask", "Next-token targets stop at boundaries, and prompt/padding positions contribute no loss.", [P("The final token of a segment cannot predict the first token of the next segment. Padding has zero attention and zero loss.")])
      ]),
      NODE("How is provenance preserved?", "Every packed segment maps back to exact shard and field-token ranges.", [
        P("A source span records shard ID/hash, token-record hash, record ID, occurrence, lane, type, packed start/end, field source ranges, truncation, loss-token count, and a span hash."),
        PROOF("The audit validates all selected masks and semantic hashes. The generated main batches contain 23 source spans, including four multi-segment candidates."),
        LINK("Inspect packing implementation", "../tdes/packing.py")
      ])
    ]
  },
  {
    id: "opus",
    title: "OPUS selection",
    shortTitle: "OPUS",
    category: "Dynamic selection",
    icon: "scan-search",
    summary: "Use the current model, AdamW state, and a proxy gradient to choose a useful, non-redundant subset from each candidate buffer.",
    takeaway: "OPUS is not another neural network; it is a step-wise algorithm that scores how candidate updates align with a desired learning direction.",
    flow: ["6 candidates", "gradient utility", "Boltzmann draw", "3 selected"],
    visual: "opus",
    details: [
      NODE("What problem does OPUS solve?", "The data stream offers six candidates, but one update can consume only three.", [
        P("Static filtering asks whether a sample looks good in general. OPUS asks whether the sample is useful for the model <strong>at this optimizer step</strong>."),
        NOTE("A candidate rejected now could become useful later because the model, proxy gradient, and optimizer geometry change over time.")
      ]),
      NODE("How is utility calculated?", "Compare each optimizer-shaped candidate update with the proxy direction, then subtract redundancy.", [
        CODE("utility(z) = learning_rate × alignment(candidate, proxy)\n             − learning_rate² × redundancy(candidate, selected_history)"),
        P("Alignment rewards candidates expected to reduce proxy loss. The redundancy term discourages selecting several candidates that would move parameters in nearly the same direction.")
      ], [
        NODE("Proxy gradient", "A protected in-distribution proxy batch describes the desired learning direction.", [P("The model calculates proxy loss and gradient for scoring only. The proxy records never enter the update batch.")]),
        NODE("Candidate gradient", "Each packed candidate produces a real per-sample gradient from the current model.", [P("This gradient answers: if this sample trained the model now, which parameter direction would it request?")]),
        NODE("Optimizer-induced geometry", "AdamW rescales gradient coordinates using its stored moments.", [P("OPUS freezes the current diagonal preconditioner and scores the candidate's effective optimizer-shaped update instead of using only the raw gradient.")]),
        NODE("CountSketch", "Large sparse gradients are projected into a deterministic 64-dimensional sketch.", [P("Inner products in sketch space approximate alignment and redundancy efficiently. The sketch seed is saved so replay uses the same projection.")])
      ]),
      NODE("How are final candidates chosen?", "Sequential Boltzmann sampling favors utility without becoming a brittle top-k rule.", [
        P("At each of three rounds, utilities become probabilities through softmax with a temperature. A saved pseudorandom draw selects one remaining candidate, then its feature joins selected history before the next round."),
        P("After selection, candidates become accepted, deferred, rejected after their retry budget, or admitted through a protected-floor override."),
        PROOF("The main OPUS ledger contains 36 decisions: 17 accepted, 12 deferred, 6 rejected, and 1 protected-floor admission."),
        LINK("Inspect OPUS implementation", "../tdes/opus.py")
      ])
    ]
  },
  {
    id: "training",
    title: "Batches and training",
    shortTitle: "Training",
    category: "Model update",
    icon: "cpu",
    summary: "Combine three selected packed candidates into one global batch, calculate masked loss, accumulate gradients, and perform one real AdamW update.",
    takeaway: "One optimizer step is defined by all selected microbatches together, not by any one packed sequence.",
    flow: ["3 microbatches", "masked loss", "mean gradient", "AdamW update"],
    details: [
      NODE("Microbatch versus global batch", "The global batch is every sample contributing to one optimizer update.", [
        CODE("microbatch size 1 × world size 1 × accumulation steps 3\n= global batch size 3"),
        P("The implementation computes each selected candidate's gradient and combines them using their loss-token counts before one optimizer update. This is mathematically equivalent to the configured single-process accumulation.")
      ]),
      NODE("What does the model calculate?", "Visible context and position generate logits for the next byte token.", [
        P("For every loss-bearing row, the model reads token IDs allowed by the attention mask, forms a context representation, adds learned position weights, computes softmax probabilities, and takes negative log probability of the target label."),
        P("Prompt-only and padding positions are skipped because their loss-mask value is zero.")
      ], [
        NODE("Sample-level losses", "Token losses are aggregated back through segment IDs and source-span hashes.", [P("Each learning record contains source identity, loss-bearing token count, summed loss, and mean loss, linked to the exact consumption row and model-before/model-after hashes.")]),
        NODE("AdamW state", "First and second moments plus the optimizer step affect parameter updates.", [P("The checkpoint saves sparse weights and both moment maps. Restoring weights without those moments would not reproduce the next update.")])
      ]),
      NODE("What identifies a batch?", "Tokens, masks, ordering, source spans, tokenizer, manifest root, and OPUS result form one semantic hash.", [
        P("The batch ID is derived from the semantic SHA-256. Wall-clock timings and file paths do not participate, so replay on a different machine can still match."),
        PROOF("The main branch contains six unique contiguous committed batch IDs, one for each optimizer step."),
        LINK("Browse packed batches", "../submission_artifacts/batches/")
      ])
    ]
  },
  {
    id: "ledgers",
    title: "Consumption and learning ledgers",
    shortTitle: "Ledgers",
    category: "Traceability",
    icon: "book-open-check",
    summary: "Write append-only, hash-chained records of what the model consumed, what loss each source produced, and why OPUS made every decision.",
    takeaway: "The consumption ledger answers what was used; the learning ledger answers what loss was attributed to each source span.",
    flow: ["batch commit", "consumption row", "learning rows", "hash chain"],
    details: [
      NODE("Consumption ledger", "One row describes each committed optimizer update.", [
        P("A consumption row records branch, step, batch ID/hash, ordered candidates and source spans, stage, lane counts, capacity/non-padding/loss tokens, model and optimizer hashes, configuration identities, and measured durations."),
        LINK("Open main consumption ledger", "../submission_artifacts/ledgers/main.consumption.jsonl")
      ]),
      NODE("Learning ledger", "One or more rows connect actual loss back to source records.", [
        P("Learning rows record candidate and source-span identity, loss-bearing token count, sum and mean loss, before/after model hashes, and the parent consumption-row hash."),
        PROOF("The generated run contains 23 learning rows covering 717 loss-bearing tokens. The audit recomputes every batch loss and token count from these rows."),
        LINK("Open main learning ledger", "../submission_artifacts/ledgers/main.learning.jsonl")
      ]),
      NODE("Why hash-chain the ledgers?", "Each row commits to itself and the preceding row.", [
        CODE("row 0 hash = hash(row 0 body + zero previous hash)\nrow 1 hash = hash(row 1 body + row 0 hash)\nrow 2 hash = hash(row 2 body + row 1 hash)"),
        P("Editing, removing, inserting, or reordering a historical row breaks the chain. Offsets must also remain contiguous."),
        PROOF("Automated tests mutate a ledger row and require validation to fail."),
        LINK("Inspect ledger implementation", "../tdes/ledgers.py")
      ])
    ]
  },
  {
    id: "checkpoint",
    title: "Atomic checkpoints",
    shortTitle: "Checkpoint",
    category: "Persistence",
    icon: "save",
    summary: "Persist model, optimizer, stream, selector, branch, and exact ledger positions as one verified recovery boundary.",
    takeaway: "A complete checkpoint says both what the model is and exactly where the data execution system must continue.",
    flow: ["committed ledgers", "serialize state", "fsync + rename", "verify hash"],
    details: [
      NODE("What is saved?", "Every state value capable of changing the next batch or next update.", [
        P("The payload includes sparse model weights, AdamW moments and step, training and proxy packer cursors, deferred queues, OPUS settings and RNG state, next optimizer step, branch lineage, and tokenizer/manifest/config/code hashes."),
        P("It also stores exact row counts, byte offsets, and tail hashes for the consumption, learning, OPUS, and event ledgers.")
      ], [
        NODE("Why save RNG state?", "OPUS must receive the exact next pseudorandom draw.", [P("Restarting only from the original seed would replay earlier random values. Saving the current 64-bit generator state resumes at the next value.")]),
        NODE("Why save deferred queues?", "A read but unselected sample is no longer at its lane cursor.", [P("Without the queue it would disappear after restoration or be replaced by a new record, changing the candidate buffer.")]),
        NODE("Why tie checkpoints to ledgers?", "Recovery needs a trusted transaction boundary.", [P("Anything beyond the checkpoint's ledger byte offsets is uncommitted suffix data and is truncated during restoration.")])
      ]),
      NODE("What makes the write atomic?", "A valid old checkpoint or a valid new checkpoint exists, never a trusted half-file.", [
        P("The system writes a temporary file, flushes it, calls <code>fsync</code>, atomically renames it to the final path, reloads it, and verifies the checkpoint hash."),
        NOTE("Atomic rename protects against interruption during checkpoint writing. Read-after-write hashing protects against malformed content."),
        LINK("Inspect checkpoint implementation", "../tdes/checkpoint.py")
      ]),
      NODE("How many checkpoints are generated?", "Bootstrap, every main step, replay completion, and fork boundaries remain inspectable.", [
        P("The main branch stores a bootstrap checkpoint and one checkpoint after each of six steps. Separate files preserve the replay and fork histories."),
        LINK("Browse checkpoints", "../submission_artifacts/checkpoints/")
      ])
    ]
  },
  {
    id: "recovery",
    title: "Crash and exact resume",
    shortTitle: "Crash + resume",
    category: "Recovery",
    icon: "refresh-ccw",
    summary: "Terminate the training worker after a committed checkpoint, start a fresh process, and prove that its next reconstructed batch is exactly the expected one.",
    takeaway: "Recovery may recompute uncommitted work, but it must never repeat or skip a committed batch.",
    flow: ["step 2 saved", "worker exits 86", "fresh worker", "step 3 matches"],
    visual: "recovery",
    details: [
      NODE("How is the crash made real?", "The training worker process destroys its in-memory state with os._exit(86).", [
        P("After committing step 2, the worker saves and verifies <code>main.step-0002.json</code>. A separate engine loaded from that persisted checkpoint previews the expected step-3 batch and writes its identity."),
        P("The original training worker flushes its log and calls <code>os._exit(86)</code>. The parent process requires exactly exit code 86 before it launches recovery.")
      ]),
      NODE("What happens on resume?", "A new process trusts only checkpointed state and committed ledger prefixes.", [
        P("The resume worker loads model, AdamW, packer cursors, deferred queues, proxy position, OPUS RNG, and next step. It truncates any ledger bytes beyond the checkpoint offsets."),
        P("Before training, it reconstructs the next batch and compares batch ID, full hash, candidate IDs/hashes, and ordered source spans with the persisted expectation.")
      ], [
        NODE("No repeat", "The reconstructed batch must differ from the preceding committed batch.", [P("The worker explicitly checks that step 3's batch ID is not step 2's batch ID.")]),
        NODE("No skip", "Committed main steps must remain the contiguous sequence 0 through 5.", [P("The consumption ledger enforces one unique <code>(branch, step)</code>, and the audit verifies its complete ordered sequence.")])
      ]),
      NODE("What was proven in the generated run?", "The persisted expectation and resumed step 3 are byte-semantically identical.", [
        PROOF("Expected and resumed batch ID: <code>batch-4f7891bdf6d93683c3a3</code> in the demonstrated run. A new run may produce the same semantic identity because timings are excluded."),
        LINK("Open crash expectation", "../submission_artifacts/reports/crash_expectation.json"),
        LINK("Open execution log", "../submission_artifacts/run.log")
      ])
    ]
  },
  {
    id: "replay",
    title: "Replay and branch fork",
    shortTitle: "Replay + fork",
    category: "Reconstruction",
    icon: "git-fork",
    summary: "Reconstruct historical batches from an earlier checkpoint, then create a separate branch whose intentional configuration differences are explicit.",
    takeaway: "Replay proves the old stream; a fork preserves common history while making future differences visible and attributable.",
    flow: ["old checkpoint", "replay 0–3", "compare hashes", "fork new branch"],
    details: [
      NODE("What is historical replay?", "Start from old persisted state and independently regenerate an earlier interval.", [
        P("Replay loads the bootstrap checkpoint and reconstructs steps 0 through 3. Because OPUS depends on the changing model and optimizer, replay also performs the historical model updates needed to reach each next decision."),
        P("At every step it compares batch ID, full batch hash, candidate IDs, candidate hashes, ordered source spans, and each span hash with the original main ledger."),
        PROOF("All four reconstructed historical batches and fourteen source spans match the original run."),
        LINK("Open replay report", "../submission_artifacts/reports/replay.json")
      ]),
      NODE("What is a branch fork?", "Load common history, declare a new configuration, and write independent future ledgers.", [
        P("The fork begins after main step 0. It records its parent branch, parent checkpoint path/hash, fork step, parent ledger states, reason, and canonical configuration delta."),
        CODE("OPUS temperature: 0.35 → 0.6125\nfoundation lane weights:\n  general 0.55 → 0.35\n  instruction 0.30 → 0.25\n  code 0.15 → 0.40"),
        P("The fork executes two updates under its own branch ID and hash chains.")
      ], [
        NODE("Parent preservation", "The main consumption ledger must remain byte-for-byte unchanged.", [P("The fork hashes the parent ledger before and after its work and requires equality.")]),
        NODE("Explicit divergence", "At least one fork batch must differ from the corresponding main batch.", [P("The changed temperature and lane mixture produce a deliberate stream difference recorded with its cause.")])
      ]),
      NODE("Why keep replay and fork separate?", "Replay tests sameness; fork documents intentional difference.", [
        P("A replay with different batches is a failure. A fork with an explicit configuration delta is a valid new experiment. Both remain traceable to the same earlier checkpoint."),
        LINK("Open fork report", "../submission_artifacts/reports/fork.json"),
        LINK("Open branch lineage", "../submission_artifacts/ledgers/branches.json")
      ])
    ]
  },
  {
    id: "audit",
    title: "Audit, evidence, and performance",
    shortTitle: "Audit",
    category: "Independent proof",
    icon: "badge-check",
    summary: "Reopen generated artifacts through fresh readers, recompute every claim, and render a machine-readable and human-readable evidence bundle.",
    takeaway: "The log says what happened; the audit proves whether the persisted artifacts support that story.",
    flow: ["generated files", "10 validators", "evidence JSON", "PASS / FAIL"],
    details: [
      NODE("What does the audit check?", "Ten independent groups cover the complete system contract.", [
        CODE("1. tokenizer and shard integrity\n2. evaluation firewall\n3. packing correctness\n4. mixture and protected floors\n5. OPUS audit trail\n6. learning trace and ledgers\n7. crash recovery\n8. replay\n9. fork lineage\n10. throughput reconstruction"),
        P("Each validator catches exceptions and emits measured details, artifact references, and a boolean result. Overall PASS requires every group to pass."),
        LINK("Open audit report", "../submission_artifacts/reports/audit.json")
      ]),
      NODE("How is evidence generated?", "Evidence JSON is derived from audit results, then Markdown is rendered from that JSON.", [
        P("Each requirement records PASS/FAIL, measured values, the audit report, and supporting artifact paths. The evidence bundle itself has a semantic hash."),
        NOTE("No component is allowed to claim success merely because a log message exists. The validator must reconstruct the claim from manifests, batches, ledgers, checkpoints, or reports."),
        LINK("Open machine-readable evidence", "../submission_artifacts/evidence.json"),
        LINK("Open evidence summary", "../submission_artifacts/evidence.md")
      ]),
      NODE("How is efficiency measured?", "Raw token counts and real monotonic durations make reported ratios reconstructible.", [
        P("The report stores capacity, non-padding tokens, loss-bearing tokens, packing/OPUS/training/end-to-end nanoseconds, stage totals, utilization, useful-token ratio, and several token-per-second rates."),
        CODE("packing utilization = non-padding tokens / capacity tokens\nuseful token ratio = loss-bearing tokens / capacity tokens\nuseful tokens/s = loss-bearing tokens / end-to-end seconds"),
        PROOF("The audit recalculates counts and ratios from main consumption rows and rejects non-positive timings or impossible utilization."),
        LINK("Open performance report", "../submission_artifacts/performance.json")
      ], [
        NODE("Automated tests", "Adversarial tests verify failure behavior as well as the happy path.", [P("Tests cover shard tampering, tokenizer identity, packing leakage, prompt masking, mixture quotas, deterministic OPUS, firewalls, ledger mutation, suffix truncation, checkpoint tampering, and full demo regeneration.")])
      ])
    ]
  }
];

const elements = {
  stageNav: document.getElementById("stageNav"),
  stageSearch: document.getElementById("stageSearch"),
  pipelineMap: document.getElementById("pipelineMap"),
  breadcrumbCurrent: document.getElementById("breadcrumbCurrent"),
  stepNumber: document.getElementById("stepNumber"),
  stageCategory: document.getElementById("stageCategory"),
  stageTitle: document.getElementById("stageTitle"),
  stageSummary: document.getElementById("stageSummary"),
  stageTakeaway: document.getElementById("stageTakeaway"),
  stageFlow: document.getElementById("stageFlow"),
  stepSymbol: document.getElementById("stepSymbol"),
  conceptTree: document.getElementById("conceptTree"),
  interactiveBand: document.getElementById("interactiveBand"),
  previousStep: document.getElementById("previousStep"),
  nextStep: document.getElementById("nextStep"),
  lessonPosition: document.getElementById("lessonPosition"),
  understoodButton: document.getElementById("understoodButton"),
  progressLabel: document.getElementById("progressLabel"),
  progressFill: document.getElementById("progressFill")
};

let currentIndex = Math.max(0, stages.findIndex((stage) => `#${stage.id}` === window.location.hash));
const completed = new Set(JSON.parse(localStorage.getItem("tdex-understood") || "[]"));

function icon(name, extraClass = "") {
  return `<i data-lucide="${name}"${extraClass ? ` class="${extraClass}"` : ""}></i>`;
}

function refreshIcons() {
  if (window.lucide) {
    window.lucide.createIcons({ attrs: { "stroke-width": 1.8 } });
  }
}

function saveProgress() {
  localStorage.setItem("tdex-understood", JSON.stringify([...completed]));
}

function updateProgress() {
  elements.progressLabel.textContent = `${completed.size} / ${stages.length}`;
  elements.progressFill.style.width = `${(completed.size / stages.length) * 100}%`;
}

function renderNavigation(filter = "") {
  const normalized = filter.trim().toLowerCase();
  elements.stageNav.innerHTML = "";
  let matches = 0;
  stages.forEach((stage, index) => {
    const searchable = JSON.stringify(stage).toLowerCase();
    if (normalized && !searchable.includes(normalized)) return;
    matches += 1;
    const button = document.createElement("button");
    button.type = "button";
    button.className = `stage-nav-button${index === currentIndex ? " active" : ""}${completed.has(stage.id) ? " done" : ""}`;
    button.innerHTML = `
      <span class="stage-nav-number">${String(index + 1).padStart(2, "0")}</span>
      <span class="stage-nav-copy"><strong>${stage.shortTitle}</strong><span>${stage.category}</span></span>
      <span class="stage-nav-status">${completed.has(stage.id) ? icon("check") : ""}</span>`;
    button.addEventListener("click", () => selectStage(index));
    elements.stageNav.appendChild(button);
  });
  if (!matches) {
    elements.stageNav.innerHTML = '<div class="nav-empty">No matching concepts</div>';
  }
  refreshIcons();
}

function renderPipeline() {
  elements.pipelineMap.innerHTML = "";
  stages.forEach((stage, index) => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = `pipeline-node${index === currentIndex ? " active" : ""}${completed.has(stage.id) ? " done" : ""}`;
    button.title = `Step ${index + 1}: ${stage.title}`;
    button.setAttribute("aria-label", `Step ${index + 1}: ${stage.title}`);
    button.innerHTML = `<span class="pipeline-node-icon">${icon(stage.icon)}</span><span class="pipeline-node-label">${stage.shortTitle}</span>`;
    button.addEventListener("click", () => selectStage(index));
    elements.pipelineMap.appendChild(button);
    if (index < stages.length - 1) {
      const link = document.createElement("span");
      link.className = "pipeline-link";
      link.setAttribute("aria-hidden", "true");
      elements.pipelineMap.appendChild(link);
    }
  });
  requestAnimationFrame(() => {
    elements.pipelineMap.querySelector(".pipeline-node.active")?.scrollIntoView({ behavior: "smooth", block: "nearest", inline: "center" });
  });
}

function renderContentBlock(block) {
  if (block.type === "p") {
    const element = document.createElement("div");
    element.className = "content-block";
    element.innerHTML = block.html;
    return element;
  }
  if (block.type === "code") {
    const element = document.createElement("pre");
    element.className = "code-block";
    element.textContent = block.text;
    return element;
  }
  if (block.type === "note") {
    const element = document.createElement("div");
    element.className = "note-block";
    element.innerHTML = `<strong>Important:</strong> ${block.html}`;
    return element;
  }
  if (block.type === "proof") {
    const element = document.createElement("div");
    element.className = "proof-block";
    element.innerHTML = `${icon("badge-check")}<div><strong>Generated proof</strong><br>${block.html}</div>`;
    return element;
  }
  const element = document.createElement("div");
  element.className = "content-block";
  element.innerHTML = `<a class="artifact-link" href="${block.href}" target="_blank" rel="noreferrer">${icon("external-link")}<span>${block.label}</span></a>`;
  return element;
}

function renderNode(node, path) {
  const details = document.createElement("details");
  details.className = "concept-node";
  const indexLabel = path.map((part) => String(part + 1).padStart(2, "0")).join(".");
  const summary = document.createElement("summary");
  summary.innerHTML = `
    <span class="concept-index">${indexLabel}</span>
    <span class="concept-summary-copy"><span class="concept-title">${node.title}</span><span class="concept-preview">${node.preview}</span></span>
    <span class="concept-chevron">${icon("chevron-right")}</span>`;
  details.appendChild(summary);
  const body = document.createElement("div");
  body.className = "concept-body";
  node.content.forEach((block) => body.appendChild(renderContentBlock(block)));
  if (node.children?.length) {
    const children = document.createElement("div");
    children.className = "concept-children";
    node.children.forEach((child, childIndex) => children.appendChild(renderNode(child, [...path, childIndex])));
    body.appendChild(children);
  }
  details.appendChild(body);
  return details;
}

function renderFlow(stage) {
  elements.stageFlow.innerHTML = "";
  stage.flow.forEach((item, index) => {
    const wrapper = document.createElement("div");
    wrapper.className = "flow-item";
    wrapper.innerHTML = `<span class="flow-box">${item}</span>${index < stage.flow.length - 1 ? icon("arrow-right", "flow-arrow") : ""}`;
    elements.stageFlow.appendChild(wrapper);
  });
}

function renderTokenizerDemo() {
  elements.interactiveBand.innerHTML = `
    <div class="interactive-heading"><div><span class="visual-label">Live model input</span><h2>Turn text into frozen byte tokens</h2><p>The tokenizer adds six to each UTF-8 byte. Sequence delimiters are shown separately.</p></div></div>
    <div class="interactive-grid">
      <div class="tool-pane"><h3>Text</h3><textarea class="token-input" id="tokenInput" maxlength="80">Data!</textarea><div class="byte-row" id="byteOutput"></div></div>
      <div class="tool-pane"><h3>Token sequence</h3><div class="token-output" id="tokenOutput"></div><p>Vocabulary: 6 special IDs + 256 byte IDs = 262.</p></div>
    </div>`;
  const input = document.getElementById("tokenInput");
  const update = () => {
    const bytes = [...new TextEncoder().encode(input.value)];
    document.getElementById("byteOutput").textContent = `UTF-8 bytes: [${bytes.join(", ")}]`;
    document.getElementById("tokenOutput").innerHTML = [
      '<span class="token-chip special">BOS·1</span>',
      ...bytes.map((byte) => `<span class="token-chip" title="byte ${byte}">${byte + 6}</span>`),
      '<span class="token-chip special">EOS·2</span>'
    ].join("");
  };
  input.addEventListener("input", update);
  update();
}

const mixtures = {
  foundation: { label: "Foundation", weights: { general: 55, instruction: 30, code: 15 }, slots: ["general", "instruction", "general", "code", "general", "instruction"] },
  reasoning: { label: "Reasoning", weights: { general: 30, instruction: 35, code: 35 }, slots: ["code", "instruction", "general", "code", "instruction", "general"] }
};

function renderMixtureDemo() {
  elements.interactiveBand.innerHTML = `
    <div class="interactive-heading"><div><span class="visual-label">Curriculum compiler</span><h2>Stage weights become candidate slots</h2><p>Percentages are apportioned into a deterministic six-candidate buffer.</p></div></div>
    <div class="interactive-grid">
      <div class="tool-pane">
        <div class="segmented-control" id="mixtureControl"><button class="segment-button active" data-stage="foundation" type="button">Foundation</button><button class="segment-button" data-stage="reasoning" type="button">Reasoning</button></div>
        <div class="mixture-bar" id="mixtureBar"></div>
        <div class="legend-row"><span class="legend-item"><i class="legend-swatch" style="background:var(--teal)"></i>General</span><span class="legend-item"><i class="legend-swatch" style="background:var(--blue)"></i>Instruction</span><span class="legend-item"><i class="legend-swatch" style="background:var(--coral)"></i>Code</span></div>
      </div>
      <div class="tool-pane"><h3>Compiled candidate buffer</h3><div class="slot-row" id="mixtureSlots"></div><p id="mixtureDescription"></p></div>
    </div>`;
  const update = (key) => {
    const data = mixtures[key];
    document.querySelectorAll("#mixtureControl button").forEach((button) => button.classList.toggle("active", button.dataset.stage === key));
    document.getElementById("mixtureBar").innerHTML = Object.entries(data.weights).map(([lane, weight]) => `<div class="mixture-segment ${lane}" style="width:${weight}%">${weight}%</div>`).join("");
    const labels = { general: "G", instruction: "I", code: "C" };
    document.getElementById("mixtureSlots").innerHTML = data.slots.map((lane) => `<span class="slot ${lane}" title="${lane}">${labels[lane]}</span>`).join("");
    const counts = data.slots.reduce((result, lane) => ({ ...result, [lane]: (result[lane] || 0) + 1 }), {});
    document.getElementById("mixtureDescription").textContent = `${data.label}: ${counts.general} general, ${counts.instruction} instruction, ${counts.code} code. Selected code floor: at least 1 of 3.`;
  };
  document.querySelectorAll("#mixtureControl button").forEach((button) => button.addEventListener("click", () => update(button.dataset.stage)));
  update("foundation");
}

function renderPackingDemo() {
  const tokens = [
    { value: "BOS", group: "segment-a" }, { value: "A", group: "segment-a" }, { value: "EOS", group: "segment-a" },
    { value: "BOS", group: "segment-b" }, { value: "B", group: "segment-b" }, { value: "EOS", group: "segment-b" },
    { value: "PAD", group: "padding" }, { value: "PAD", group: "padding" }
  ];
  const groups = [0, 0, 0, 1, 1, 1, -1, -1];
  let cells = "";
  for (let row = 0; row < 8; row += 1) {
    for (let column = 0; column < 8; column += 1) {
      const allowed = groups[row] >= 0 && groups[row] === groups[column] && column <= row;
      cells += `<span class="mask-cell${allowed ? ` allowed-${groups[row] === 0 ? "a" : "b"}` : ""}" title="row ${row}, column ${column}: ${allowed ? "visible" : "blocked"}"></span>`;
    }
  }
  elements.interactiveBand.innerHTML = `
    <div class="interactive-heading"><div><span class="visual-label">Packing microscope</span><h2>Two records, one physical sequence</h2><p>Colors identify segments. The attention matrix remains block-causal.</p></div></div>
    <div class="packing-demo">
      <div class="tool-pane"><h3>Packed token positions</h3><div class="packed-sequence">${tokens.map((token) => `<span class="packed-token ${token.group}">${token.value}</span>`).join("")}</div><div class="legend-row"><span class="legend-item"><i class="legend-swatch" style="background:var(--teal)"></i>Segment A</span><span class="legend-item"><i class="legend-swatch" style="background:var(--blue)"></i>Segment B</span><span class="legend-item"><i class="legend-swatch" style="background:#d9e0e3"></i>Padding</span></div></div>
      <div class="tool-pane"><h3>8 × 8 attention mask</h3><div class="mask-grid">${cells}</div><p>Filled cells are visible. Segment B cannot see segment A.</p></div>
    </div>`;
}

const opusCandidates = [
  { id: "A", name: "Arithmetic proof", lane: "instruction", alignment: 92, overlap: { A: 0, B: 58, C: 14, D: 12, E: 5, F: 8 } },
  { id: "B", name: "Similar arithmetic", lane: "instruction", alignment: 88, overlap: { A: 70, B: 0, C: 25, D: 10, E: 4, F: 10 } },
  { id: "C", name: "New math problem", lane: "general", alignment: 76, overlap: { A: 14, B: 25, C: 0, D: 7, E: 4, F: 3 } },
  { id: "D", name: "Code reasoning", lane: "code", alignment: 63, overlap: { A: 12, B: 10, C: 7, D: 0, E: 4, F: 2 } },
  { id: "E", name: "Unrelated prose", lane: "general", alignment: 31, overlap: { A: 5, B: 4, C: 4, D: 4, E: 0, F: 6 } },
  { id: "F", name: "Repeated prose", lane: "general", alignment: 24, overlap: { A: 8, B: 10, C: 3, D: 2, E: 15, F: 0 } }
];

function renderOpusDemo() {
  elements.interactiveBand.innerHTML = `
    <div class="interactive-heading"><div><span class="visual-label">Selection lab</span><h2>Build one update batch</h2><p>Utility changes after each selection because redundant directions receive a penalty.</p></div></div>
    <div class="opus-layout">
      <div>
        <div class="proxy-direction"><span>Proxy direction</span><strong>Improve mathematical reasoning</strong></div>
        <div class="opus-controls"><button class="command-button primary" id="opusNext" type="button">${icon("dices")}<span>Draw next</span></button><button class="command-button" id="opusReset" type="button">${icon("rotate-ccw")}<span>Reset</span></button></div>
        <div class="opus-status" id="opusStatus">Round 1 of 3 · no candidates selected</div>
      </div>
      <div class="candidate-list" id="candidateList"></div>
    </div>`;
  let selected = [];
  const draws = [0.21, 0.68, 0.42];
  const utilities = () => opusCandidates.map((candidate) => ({
    ...candidate,
    utility: candidate.alignment - selected.reduce((sum, picked) => sum + candidate.overlap[picked], 0)
  }));
  const probabilities = (rows) => {
    const remaining = rows.filter((row) => !selected.includes(row.id));
    const max = Math.max(...remaining.map((row) => row.utility / 18));
    const weights = remaining.map((row) => Math.exp(row.utility / 18 - max));
    const total = weights.reduce((sum, value) => sum + value, 0);
    return Object.fromEntries(remaining.map((row, index) => [row.id, weights[index] / total]));
  };
  const paint = (message) => {
    const rows = utilities();
    const probs = probabilities(rows);
    document.getElementById("candidateList").innerHTML = rows.map((row) => {
      const isSelected = selected.includes(row.id);
      const probability = probs[row.id] || 0;
      return `<div class="candidate-row${isSelected ? " selected" : ""}">
        <div class="candidate-name"><strong>${row.id} · ${row.name}</strong><span>${row.lane} · alignment ${row.alignment} · utility ${Math.round(row.utility)}</span></div>
        <div class="utility-meter"><div class="utility-fill" style="width:${Math.max(0, Math.min(100, row.utility))}%"></div></div>
        <div class="candidate-score">${isSelected ? "SELECTED" : `${Math.round(probability * 100)}%`}</div>
      </div>`;
    }).join("");
    document.getElementById("opusStatus").textContent = message || (selected.length ? `Selected: ${selected.join(", ")} · round ${Math.min(selected.length + 1, 3)} of 3` : "Round 1 of 3 · no candidates selected");
    document.getElementById("opusNext").disabled = selected.length >= 3;
  };
  document.getElementById("opusNext").addEventListener("click", () => {
    const rows = utilities();
    const probs = probabilities(rows);
    const draw = draws[selected.length];
    let cumulative = 0;
    let chosen = rows.find((row) => !selected.includes(row.id));
    for (const row of rows.filter((item) => !selected.includes(item.id))) {
      cumulative += probs[row.id];
      if (draw < cumulative) { chosen = row; break; }
    }
    selected.push(chosen.id);
    paint(selected.length === 3 ? `Final update batch: ${selected.join(", ")} · last RNG draw ${draw.toFixed(2)}` : `Selected ${chosen.id} with RNG draw ${draw.toFixed(2)} · redundancy now updates`);
  });
  document.getElementById("opusReset").addEventListener("click", () => { selected = []; paint(); });
  paint();
}

function renderRecoveryDemo() {
  elements.interactiveBand.innerHTML = `
    <div class="interactive-heading"><div><span class="visual-label">Recovery boundary</span><h2>Persisted state survives; volatile work disappears</h2><p>The restored worker reconstructs the next batch solely from the checkpoint.</p></div></div>
    <div class="recovery-layout">
      <div class="state-column saved"><h3>${icon("hard-drive")}Checkpoint after step 2</h3><div class="state-list"><div class="state-item"><span>Next optimizer step</span><code>3</code></div><div class="state-item"><span>Lane cursors</span><code>saved</code></div><div class="state-item"><span>Deferred queue</span><code>saved</code></div><div class="state-item"><span>OPUS RNG state</span><code>saved</code></div><div class="state-item"><span>Ledger tail hashes</span><code>saved</code></div></div></div>
      <div class="crash-divider">${icon("zap")}</div>
      <div class="state-column volatile" id="volatileState"><h3>${icon("memory-stick")}Worker memory</h3><div class="state-list"><div class="state-item"><span>Partially built batch</span><code>temporary</code></div><div class="state-item"><span>Advanced cursors</span><code>temporary</code></div><div class="state-item"><span>Consumed RNG draws</span><code>temporary</code></div><div class="state-item"><span>Uncommitted ledger bytes</span><code>temporary</code></div></div></div>
    </div>
    <div class="opus-controls"><button class="command-button primary" id="crashButton" type="button">${icon("power")}<span>Crash worker</span></button><button class="command-button" id="restoreButton" type="button" disabled>${icon("refresh-ccw")}<span>Restore checkpoint</span></button></div>
    <div class="resume-proof" id="resumeProof">PASS · reconstructed step-3 batch ID, hash, candidates, and source spans match the persisted expectation.</div>`;
  const crash = document.getElementById("crashButton");
  const restore = document.getElementById("restoreButton");
  crash.addEventListener("click", () => {
    document.getElementById("volatileState").classList.add("crashed");
    crash.disabled = true;
    restore.disabled = false;
  });
  restore.addEventListener("click", () => {
    document.getElementById("resumeProof").classList.add("visible");
    restore.disabled = true;
  });
}

function renderInteractive(type) {
  elements.interactiveBand.hidden = !type;
  if (!type) {
    elements.interactiveBand.innerHTML = "";
    return;
  }
  if (type === "tokenizer") renderTokenizerDemo();
  if (type === "mixture") renderMixtureDemo();
  if (type === "packing") renderPackingDemo();
  if (type === "opus") renderOpusDemo();
  if (type === "recovery") renderRecoveryDemo();
}

function renderStage() {
  const stage = stages[currentIndex];
  document.title = `${stage.title} · TDEx V5 Learning Map`;
  elements.breadcrumbCurrent.textContent = stage.shortTitle;
  elements.stepNumber.textContent = `Step ${currentIndex + 1} of ${stages.length}`;
  elements.stageCategory.textContent = stage.category;
  elements.stageTitle.textContent = stage.title;
  elements.stageSummary.textContent = stage.summary;
  elements.stageTakeaway.textContent = stage.takeaway;
  elements.stepSymbol.innerHTML = icon(stage.icon);
  elements.lessonPosition.textContent = `${currentIndex + 1} / ${stages.length}`;
  elements.previousStep.disabled = currentIndex === 0;
  elements.nextStep.disabled = currentIndex === stages.length - 1;
  elements.nextStep.querySelector("span").textContent = currentIndex === stages.length - 1 ? "Complete" : "Next step";
  const isDone = completed.has(stage.id);
  elements.understoodButton.classList.toggle("done", isDone);
  elements.understoodButton.innerHTML = `${icon(isDone ? "circle-check" : "circle")}<span>${isDone ? "Understood" : "Mark understood"}</span>`;
  renderFlow(stage);
  renderInteractive(stage.visual);
  elements.conceptTree.innerHTML = "";
  stage.details.forEach((node, index) => elements.conceptTree.appendChild(renderNode(node, [index])));
  renderNavigation(elements.stageSearch.value);
  renderPipeline();
  updateProgress();
  refreshIcons();
}

function selectStage(index) {
  currentIndex = Math.max(0, Math.min(stages.length - 1, index));
  history.replaceState(null, "", `#${stages[currentIndex].id}`);
  renderStage();
  document.body.classList.remove("sidebar-open");
  window.scrollTo({ top: 0, behavior: "smooth" });
}

elements.previousStep.addEventListener("click", () => selectStage(currentIndex - 1));
elements.nextStep.addEventListener("click", () => {
  if (currentIndex < stages.length - 1) selectStage(currentIndex + 1);
});
elements.understoodButton.addEventListener("click", () => {
  const id = stages[currentIndex].id;
  if (completed.has(id)) completed.delete(id); else completed.add(id);
  saveProgress();
  renderStage();
});
elements.stageSearch.addEventListener("input", (event) => renderNavigation(event.target.value));
document.getElementById("expandAll").addEventListener("click", () => document.querySelectorAll("#conceptTree details").forEach((detail) => { detail.open = true; }));
document.getElementById("collapseAll").addEventListener("click", () => document.querySelectorAll("#conceptTree details").forEach((detail) => { detail.open = false; }));
document.getElementById("menuButton").addEventListener("click", () => document.body.classList.add("sidebar-open"));
document.getElementById("sidebarClose").addEventListener("click", () => document.body.classList.remove("sidebar-open"));
document.getElementById("sidebarScrim").addEventListener("click", () => document.body.classList.remove("sidebar-open"));
window.addEventListener("hashchange", () => {
  const index = stages.findIndex((stage) => `#${stage.id}` === window.location.hash);
  if (index >= 0 && index !== currentIndex) { currentIndex = index; renderStage(); }
});

renderStage();
