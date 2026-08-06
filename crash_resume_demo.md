# Understanding Steps, Stages, and Crash Recovery

This tutorial explains the difference between a training **step** and a curriculum
**stage**, how data moves through lanes, and what happens if training crashes while a step
is running.

## Step Versus Stage

A **step** is one optimizer update.

A **stage** is a range of steps that uses the same curriculum configuration. A stage usually
contains multiple steps.

Consider this hypothetical schedule:

| Training step | Curriculum stage | Candidate mixture |
|---|---|---|
| Step 1 | Foundation | 3 general, 2 instruction, 1 code |
| Step 2 | Foundation | 3 general, 2 instruction, 1 code |
| Step 3 | Reasoning | 2 general, 2 instruction, 2 code |
| Step 4 | Reasoning | 2 general, 2 instruction, 2 code |

The implementation uses zero-based numbering:

```text
Steps 0, 1, 2 -> foundation
Steps 3, 4, 5 -> reasoning
```

In ordinary human counting, those are training steps 1-3 and 4-6.

## Run 1: Foundation Steps

Suppose the lanes initially contain:

```text
General:     G1, G2, G3, G4, G5, ...
Instruction: I1, I2, I3, I4, I5, ...
Code:        C1, C2, C3, C4, C5, ...
```

At step 1, the foundation mixture requests:

```text
G1, G2, G3
I1, I2
C1
```

These six samples form the OPUS candidate buffer.

Suppose OPUS selects:

```text
G1, I1, C1
```

The model performs one update using those three selected samples.

The remaining candidates may be deferred:

```text
G2, G3, I2
```

The lane cursor has already moved past these records because they were read into the
candidate buffer. The deferred queue preserves them separately.

After step 1:

```text
next new general sample: G4
next new instruction sample: I3
next new code sample: C2

deferred: G2, G3, I2
```

## Run 2: Another Foundation Step

Step 2 still uses the foundation mixture:

```text
3 general
2 instruction
1 code
```

The scheduler considers deferred candidates first, then reads new data:

```text
General candidates:     G2, G3, G4
Instruction candidates: I2, I3
Code candidate:          C2
```

Suppose OPUS selects:

```text
G2, G4, C2
```

The model performs its second update.

After this committed update, a checkpoint might say:

```text
next step: 3

next new general: G5
next new instruction: I4
next new code: C3

deferred:
I3

model state: after step 2
optimizer state: after step 2
RNG state: ready for step 3
```

## Run 3: Crash During the Reasoning Step

Step 3 begins the reasoning stage.

Its candidate mixture is now:

```text
2 general
2 instruction
2 code
```

Using the checkpoint state, the candidate buffer might be:

```text
General:     G5, G6
Instruction: I3, I4
Code:        C3, C4
```

Suppose OPUS starts scoring these candidates, but the process crashes before step 3 is
committed.

The in-memory state may have temporarily advanced:

```text
general cursor moved toward G7
instruction cursor moved toward I5
code cursor moved toward C5
RNG generated some numbers
```

None of that state is trusted because step 3 did not produce a committed checkpoint.

Recovery loads the checkpoint after step 2:

```text
next new general: G5
next new instruction: I4
next new code: C3
deferred: I3
RNG state: original pre-step-3 state
```

It therefore reconstructs:

```text
G5, G6, I3, I4, C3, C4
```

OPUS receives the same model, optimizer, proxy data, and random numbers, so it makes the same
selection. Step 3 is then performed again from the beginning and committed once.

Some computation may be repeated, but no committed model update or ledger batch is
duplicated.

## Can Data Repeat Between Steps?

Yes, but only for explicit reasons:

1. A deferred candidate can be scored again later. That does not mean it was used for
   training earlier.
2. After a small dataset is exhausted, its lane can begin another epoch.
3. A configuration may intentionally allow repeated sampling.

The system records an occurrence number, so repeated source data remains distinguishable:

```text
G1, occurrence 0
G1, occurrence 1
```

Silent or accidental repetition during crash recovery is not allowed.
