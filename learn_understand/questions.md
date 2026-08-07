# Understanding Check: Steps 1-9

Answer in your own words. Focus especially on the order of operations and on what object
enters and leaves each stage. It is fine to state uncertainty explicitly.

## Question 1: Reconstruct the pipeline order

Put the following stages in the order in which data passes through them:

```text
OPUS selection
frozen tokenizer
packing
raw source verification and document cleaning
curriculum/lane scheduler
shards and manifests
data firewall
```

For every transition, briefly state what changes. For example, explain what kind of object
exists before and after tokenization, packing, and OPUS.

### Your response
1. raw source verification and document cleaning
This is where you clean records (say remove HTML footers/headers if they aren't relevant)
2. frozen tokenizer
This is to ensure that tokenization is standard. The same byte stream should lead to the same token ID sequence.
3. shards and manifests
This is to ensure that training is resumable. Resumability is not just about model weights - it is about knowing what data the model has seen and in what order and what data it will see.
4. data firewall
This is to ensure that proxy, evaluation and test data don't appear in training
5. curriculum/lane scheduler
This is basically at what stage we have what kind of data. For instance, initially we'll have general data as opposed to code data
6. packing
This is to make our data GPU friendly.
7. OPUS selection
This tells us what will give us the biggest gains in loss minimization


## Question 2: Explain the integrity hierarchy

Assume one tokenized record in the code shard is moved before another record without changing
either record's token IDs.

1. Which hash or validation check should detect this ordering change?
2. What does that shard's manifest contribute beyond the shard file itself?
3. What does `root.json` contribute beyond the individual manifests?
4. Why is `tokenizer.json` necessary even if all shard bytes are unchanged?

Finish by explaining what trusted value a checkpoint uses to bind itself to the complete
dataset snapshot.

### Your response



## Question 3: From curriculum lanes to packed candidates

At foundation step 0, the curriculum compiles this six-slot candidate request:

```text
3 general records
2 instruction records
1 code record
```

Explain:

1. What a lane is and what state each lane must retain.
2. What the curriculum contributes that the shards do not.
3. What records or objects the packer receives from the scheduler.
4. What one packed candidate contains when it leaves the packer.

Include the important shapes for one packed candidate when sequence length is 64:

```text
input_ids       = ?
labels          = ?
loss_mask       = ?
position_ids    = ?
segment_ids     = ?
attention_mask  = ?
```

Also explain why six requested lane slots produce six packed OPUS candidates rather than the
final training batch.

### Your response



## Question 4: Reason through one packed sequence

Two document records are placed into one physical sequence of length 8:

```text
position:    0    1    2    3    4    5    6    7
input:      BOS   A   EOS  BOS   B   EOS  PAD  PAD
segment:     0    0    0    1    1    1   -1   -1
```

Answer the following:

1. What should the position IDs be?
2. Which positions should bear next-token loss, and what labels should they predict?
3. Can the token `B` at position 4 attend to position 1? Why or why not?
4. Describe the nonzero regions of the 8 x 8 attention mask.
5. What information must the provenance/source span store for segment 1 so that its source
   can still be identified after batching, training, and replay?

### Your response



## Question 5: OPUS input, processing, and output

Immediately before OPUS at one optimizer step, the system has:

```text
6 fixed-length packed training candidates
2 fixed-length protected proxy candidates
the current model parameters
the current AdamW optimizer state
the saved OPUS RNG state
```

Explain the OPUS stage from input to output:

1. What gradient information is calculated for the proxy and for each training candidate?
2. Why does OPUS use the AdamW preconditioner instead of comparing only raw gradients?
3. What do proxy alignment and the redundancy penalty measure?
4. What are CountSketch and Boltzmann sampling doing at a high level?
5. What exactly leaves OPUS and enters the next training stage?

Your answer should mention the shapes/counts at the boundary: how many packed candidates enter,
how many are selected, what happens to unselected candidates, and the conceptual global-batch
shape produced from the selected sequences.

### Your response
