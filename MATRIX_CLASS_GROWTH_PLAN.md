# Matrix Class Growth Plan

Goal: keep the architecture matrix-native and fully differentiable. No hard
router, no top-k, no branch deletion during training. Choices are represented as
soft matrix flows.

## Current Finding

`simple_butterfly_matrix` reached `52.8%` on SpeechCommands at epoch 12.

This proves the primitive base is alive:

```text
channel_butterfly
block_butterfly
low_rank
ctx_matrix
product_gate
phase_matrix
```

The problem is not dead primitives. The current main bottleneck is the class
side:

```text
class head reads mostly L3.aggregate slots
class read distributions are too similar
logit norm grows while validation becomes unstable
weak classes move around between epochs
```

So the next useful change is not "add many primitives". The next useful change
is to make classes themselves differentiable matrix objects.

## Differentiable Classes

Current head:

```text
slots -> class attention -> logits
```

Better matrix-native design:

```text
slots
  -> ClassMatrix[C,D]
  -> ClassPairMatrix[C,C,D or P,D]
  -> class-slot interaction
  -> class state update
  -> logits
```

Classes become trainable states, not only final queries.

### Class Slots

Keep trainable class vectors:

```text
class_state: [C, D]
```

For each batch:

```text
class_read  = soft_matrix_read(class_state, all_slots)
class_delta = class_update_matrix([class_state, class_read, global_ctx])
class_next  = norm(class_state + write * class_delta)
logits      = matrix_dot(class_next, read)
```

All operations are differentiable. The labels are only used by the loss.

### Class Read Diversity

The class head needs a matrix prior that prevents all classes from reading the
same late aggregate slots.

Use a phase read matrix:

```text
class_phase_matrix: [C, phase_count]
phase_slot_matrix: [phase_count, slot_count]
class_slot_read = class_phase_matrix @ phase_slot_matrix
```

Start with weak coverage:

```text
yes/no/go/down: compare + suppress + aggregate
left/right/off/on: extract + compare + suppress
stop/up: extract + aggregate
```

This is not a router. It is a soft read basis, and gradients can move it.

### Class Pair Matrix

Most errors are pair/group confusions:

```text
go <-> no/down/on
left <-> right/off/yes
on/off/down/no
```

Add a small class-pair matrix:

```text
pair_state: [P, D]
```

where `P` can be 8 or 12, not `C*C` at first.

It reads soft confusion structure:

```text
soft_confusion = y_onehot^T @ softmax(logits)
pair_read      = pair_matrix @ class_state
pair_repair    = matrix([slot, class_state, pair_read])
```

No argmax is needed. This stays differentiable with respect to logits/model.

## Primitive Assessment

The current primitive base is enough for a first working system, but not enough
for stable class separation above the low 50s.

Useful primitives from `architecture_builder` to bring back in matrix form:

```text
matrix_mlp
time_matrix
freq_matrix
delta_matrix
short_onset_matrix
offset_matrix
residual_refdelta
block_compare
class_pair_contrast
class_pair_memory
memory_read
memory_write
ema_memory
late_read_repair
energy_count
normalize
suppress
```

Do not add all at once. The order should follow the current bottleneck.

### Add First

Class-side primitives:

```text
class_pair_contrast
class_pair_memory
late_read_repair
residual_refdelta
block_compare
```

Reason: current failures are class confusions, not lack of raw extraction.

### Add Second

Evidence primitives:

```text
time_matrix
freq_matrix
delta_matrix
short_onset_matrix
offset_matrix
energy_count
```

Reason: these helped v13 L0 become a real extractor.

### Add Third

Register primitives:

```text
memory_read
memory_write
ema_memory
global_summary
```

Reason: useful after class repair exists, otherwise memory can become another
late shortcut.

## Factorized Primitive Basis

This is the clean way to make more primitives cheap without top-k.

Instead of each primitive working in full `D`, use shared projections:

```text
h[D] -> z[r]
primitive operates in r
z_update[r] -> update[D]
```

Controlled dimensions:

```text
r = 16, 24, 32, 48
```

Matrix form:

```text
Z = H @ Down[D,r]
U_p = Primitive_p(Z)
U = primitive_flow @ U_p
Update = U @ Up[r,D]
```

All primitives still run. No router. The saving comes from a smaller shared
working dimension.

Recommended v3 setting:

```text
D=96
r=32
primitives=12 to 16
variants=2 or 3
blocks=4
layers=4
steps=2
```

## Growth Without Routers

Growth should be matrix expansion, not discrete architecture choice.

### Width Growth

Add blocks or variants by expanding state matrices:

```text
old_state: [B,D]
new_state: [B+k,D]
transport: [B+k,B+k]
```

Initialize new blocks from old blocks:

```text
new_block = old_block @ A + small_noise
```

Then train through a soft transport matrix. No branch is selected or killed.

Trigger:

```text
class reads same few slots
block cosine high
weak classes share the same wrong predictions
```

### Depth Growth

Insert a new soft step/layer initialized near identity:

```text
h_new = norm(h + small_write * matrix_step(h))
```

Start with small write, not zero:

```text
write ~= 0.10 to 0.20
```

Trigger:

```text
late aggregate update_norm is very high
class head depends too much on final aggregate
validation plateaus while train still improves
```

### Memory Growth

Add memory cells by expanding `MemoryMatrix[M,D]`.

Trigger:

```text
same class pairs confused repeatedly
class_pair repair has high update but low accuracy gain
```

Start with:

```text
M=4 or 6
pair_memory=8
```

## Next Versions

### v2 Test

Run existing soft transport on SpeechCommands.

Question:

```text
Does variant_transport reduce class/head collapse?
```

If yes, keep variants.

If no, class matrix is definitely the next bottleneck.

### v3 Proposal

Name:

```text
class_matrix_v3
```

Changes:

```text
ClassMatrix[C,D]
ClassPairMatrix[P,D]
phase read matrix
class update matrix
class-pair repair primitive
stronger class read diversity metric
```

No new large primitive bank yet.

### v4 Proposal

Name:

```text
factorized_primitive_basis_v4
```

Changes:

```text
shared Down[D,r] / Up[r,D]
12-16 primitives in r-space
primitive_flow[P,P]
variant_transport[V,V]
memory matrix optional
```

This is where we add more primitives cheaply.

### v5 Proposal

Name:

```text
matrix_growth_v5
```

Changes:

```text
width expansion by matrix initialization
depth insertion by near-identity step
memory expansion by new cells
CSV tracker updated every run
```

No hard controller at first. Growth decisions can be manual from metrics, then
made automatic later.

## Metrics Required

Add these to the CSV tracker for all future runs:

```text
class_read_entropy
class_read_diversity
class_phase_mass_extract
class_phase_mass_compare
class_phase_mass_suppress
class_phase_mass_aggregate
class_pair_update_norm
memory_read_mass
memory_write_norm
variant_transport_entropy
primitive_flow_entropy
slot_cosine_mean
weak_class_min_acc
logit_norm
```

Success signs:

```text
classes read different phases
weak class min accuracy rises
logit norm does not explode
late aggregate is useful but not the only read source
class-pair repair helps go/no/down/on and left/right/off
```

## Current Recommendation

1. Run v2 on SpeechCommands.
2. If v2 is not clearly better than v1, build v3 class matrix.
3. Only after class matrix works, add factorized primitive basis.
4. Only after factorized primitives work, add memory growth and width/depth
   growth.
