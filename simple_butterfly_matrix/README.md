# simple_butterfly_matrix

Minimal transferable matrix/butterfly architecture.

## Core Shape

```text
wav / signal
  -> MatrixEvidence
  -> block-specific matrix init
  -> L0 extract
  -> L1 compare
  -> L2 suppress
  -> L3 aggregate
  -> MatrixClassificationHead over all saved slots
```

The backbone is separate from the head:

```text
ButterflyMatrixBackbone -> slots
MatrixClassificationHead(slots) -> logits
```

To move to another task, keep the backbone and replace the task head. For another
modality, replace only the evidence adapter and keep the same slot interface.

## Blocks, Layers, Steps

- `block`: parallel state cell. Blocks specialize because they start from
  different evidence queries and class heads can read different block slots.
- `layer`: fixed phase in the sequence.
  - `L0 extract`
  - `L1 compare`
  - `L2 suppress`
  - `L3 aggregate`
- `step`: repeated matrix update inside a layer. Each step reads evidence by a
  matrix attention and writes back to the same block state.

There is no internal router. The sequence itself creates specialization.

## Primitive Operations

Every step computes all primitive operations:

```text
channel_butterfly
block_butterfly
low_rank
ctx_matrix
product_gate
phase_matrix
```

They are not selected by top-1/top-k routing. All primitives receive gradient.

## Operation Matrix

Operations are connected by a learned matrix:

```text
mixed_primitive[p] = sum_q op_transition[p, q] * primitive[q]
update = sum_p sigmoid(primitive_gain[p]) * mixed_primitive[p]
```

So the operation choice is still matrix-based. It is a differentiable matrix over
primitive outputs, not a router that can kill branches early.

## Starting Basis

Each layer phase starts with a different `op_transition` prior:

```text
extract:   channel_butterfly + ctx_matrix
compare:   channel_butterfly + low_rank + product_gate
suppress:  phase_matrix common-mode suppression + block_butterfly
aggregate: block_butterfly + channel_butterfly + low_rank
```

This is the important bootstrap. Without a useful starting grammar, the model can
move matrices but still fail to form a useful class signal.

## Run Smoke

```bash
bash simple_butterfly_matrix/commands/run_smoke.sh
```

Run SpeechCommands if the local dataset exists:

```bash
python simple_butterfly_matrix/simple_butterfly_matrix.py \
  --data-root ../architecture_builder/data/speechcommands \
  --epochs 15 \
  --batch-size 128 \
  --eval-batch-size 256 \
  --out-dir simple_butterfly_matrix/runs/speechcommands_v1
```
