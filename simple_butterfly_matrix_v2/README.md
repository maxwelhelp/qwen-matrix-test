# simple_butterfly_matrix_v2

Soft matrix-transport version.

## Difference From v1

v1 has one state bank:

```text
blocks -> phases -> all-slot head
```

v2 adds parallel variants and soft transport:

```text
variants x blocks
  -> all primitive outputs
  -> primitive_flow matrix
  -> variant_transport matrix
  -> all-slot head
```

No top-k and no hard branch choice are used. Every primitive and every variant
remains differentiable.

## Matrices

The main trainable matrices are:

```text
primitive_flow[P, P]       primitive-to-primitive flow
variant_transport[V, V]    variant-to-variant flow
variant_primitive_bias[V,P] soft primitive pressure per variant
phase_ctx[L,S,V,D,D]       evidence context matrices
butterfly matrices          channel and block mixing
low-rank matrices           compressed transform
class read matrices         task head
```

This is closer to a matrix ecology than a router: paths compete softly by flow,
not by discrete selection.

## Run Smoke

```bash
bash simple_butterfly_matrix_v2/commands/run_smoke.sh
```

SpeechCommands:

```bash
python simple_butterfly_matrix_v2/soft_matrix_transport.py \
  --data-root ../architecture_builder/data/speechcommands \
  --epochs 15 \
  --batch-size 128 \
  --eval-batch-size 256 \
  --amp bf16 \
  --device cuda \
  --dim 96 \
  --evidence-cells 48 \
  --variants 3 \
  --out-dir simple_butterfly_matrix_v2/runs/speechcommands_v2
```
