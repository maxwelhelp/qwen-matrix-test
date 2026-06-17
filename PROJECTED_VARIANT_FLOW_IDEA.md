# Projected Variant Flow Idea

## Идея

В местах, где модель обычно ломается или схлопывается в один поздний shortcut,
держать не один путь, а несколько мягких вариантов.

Каждый вариант считается в дешевой проекции:

```text
H[D] -> Z[r] -> projected primitive/transition/write -> score/energy
```

Потом варианты не выбираются hard-router-ом. Они перетекают друг в друга через
soft matrix:

```text
variant_state[V,r]
variant_energy[V]
variant_flow[V,V] = softmax(matrix(energy, state, phase))
next_variant = variant_flow @ variant_state
```

То есть “лучший вариант” не забирается через `argmax`. Он получает больше массы,
а остальные не исчезают. Это сохраняет органичность и дифференцируемость.

## Где применять

1. Между примитивами:
   `primitive_a -> transition_op -> primitive_b`.
2. Между шагами:
   `step_t -> step_t+1`.
3. Между фазами:
   `extract/compare/suppress/aggregate`.
4. В class head:
   когда классы начинают читать только late aggregate.
5. В местах confusion:
   например `go <-> no`, `left <-> off`, `right <-> left`.

## Дешевые проекции вместо full

Не обязательно считать full matrix для всех вариантов.

Базовый режим:

```text
all variants: projected r-space
few anchors: full D-space
loss: projected consistency + anchor full reconstruction
```

Так дешевле:

```text
Z = H @ Down[D,r]
projected_update = projected_ops(Z)
H_update = projected_update @ Up[r,D]
```

Full-счет нужен не всегда. Он нужен:

1. Для anchor-вариантов, чтобы проекции не начали врать.
2. Для финального `W` или `prefix_mats`.
3. Для проверки real checkpoint decode.

## Почему это не роутер

Router делает резкий выбор:

```text
choose one path
```

Projected Variant Flow делает мягкое перетекание:

```text
all paths alive
better paths get more mass
weaker paths keep gradient
```

Это соответствует нашей философии:

```text
no hard top-k
no argmax selection
all matrices differentiable
specialization through sequence, phase, variant flow, and class matrix
```

## Что добавить в v4

```text
ProjectedOperatorBasis:
  Down[D,r]
  Up[r,D]
  op_basis[K,r,r]

ProjectedVariantFlow:
  variant_state[V,r]
  variant_flow[V,V]
  primitive_transition_pair_flow[P,T,P]

BreakPointProjection:
  detects high loss / class confusion / phase collapse
  increases variant mass around that breakpoint
```

Losses:

```text
W reconstruction
step/prefix reconstruction
primitive_transition_pair KL
variant entropy floor
projection/full consistency
phase anti-collapse
class confusion repair
```

## First Experiment

Before full v4, test only projected syntax:

```text
input W[D,D]
predict projected pair flow[P,T,P]
reconstruct W through low-rank projected basis
compare against existing baseline decoder
```

If pair KL drops and reconstruction stays stable, then move this into
`simple_butterfly_matrix_v4`.

