# simple_butterfly_matrix_v3

ClassMatrixTransport version.

## Difference From v2

v2 improved the backbone by adding soft variants and matrix transport, but the
task head still collapsed mostly to late `L3.aggregate` slots.

v3 keeps the v2 backbone and replaces the head:

```text
slots
  -> phase read matrix
  -> ClassMatrix[C,D]
  -> ClassPairMatrix[P,D]
  -> class update matrix
  -> logits
```

Classes are now trainable matrix states, not only final read queries.

## No Router Rule

There is still no hard router and no top-k.

The head uses:

```text
class_phase_logits[C,phase]
phase_slot_matrix[phase,slot]
class_pair_logits[C,pair]
```

All of these are soft differentiable matrices.

## New Metrics

Each run writes:

```text
phase_balance
pair_update_norm
class_phase_mass
class_top_reads
```

The key success sign is that classes stop reading only `aggregate` and start
using `extract/compare/suppress` differently.

## Run Smoke

```bash
bash simple_butterfly_matrix_v3/commands/run_smoke.sh
```

## Run SpeechCommands

```bash
bash simple_butterfly_matrix_v3/commands/run_speechcommands.sh
```
