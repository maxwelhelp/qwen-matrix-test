# Read Transform Write Grammar Extension Plan

Goal: keep the matrix-program grammar minimal enough to train, but expressive enough to cover attention-like, MLP-like, convolution-like, memory, MoE-like, recursive/refinement, and head-decision programs.

Base grammar:

```text
read -> transform -> write
```

This is not a narrow hand-written architecture. It is the minimum universal compute contract:

```text
read      = gather information
transform = compute/update candidate
write     = commit result to addressable state
```

The important design rule:

```text
Keep the grammar hard.
Keep semantic layer roles soft or emergent.
```

Do not hardcode:

```text
L0 = extract
L1 = compare
L2 = suppress
L3 = aggregate
```

Do keep:

```text
every step must read, transform, compose/update, and write through a stable matrix interface
```

---

## 1. Transform as a nested program

### 1.1 Transform can be composite

`transform` should not be only one primitive. It can be a bounded differentiable subprogram:

```text
transform(x) = gate(low_rank(x) + diff(x))
transform(x) = compose(wavelet(x), product_gate(x), residual(x))
transform(x) = product_gate(low_rank(x), memory_read(x)) + residual(x)
```

This matches the older symbolic-operator direction:

```text
additive dictionary
ordered product grammar
functional validation
residual mining
macro promotion
```

### 1.2 Transform hierarchy

Use a bounded hierarchy, not unbounded recursion:

```text
Level 0: primitive operator
Level 1: weighted mixture of primitive operators
Level 2: ordered product / short chain of operators
Level 3: macro-step made from tested useful chains
Level 4: promoted operator/macro in OperatorBank/MacroBank
```

### 1.3 Useful examples

```text
diff + smooth + gate
  -> onset / wavelet-like filter

low_rank + product_gate + residual
  -> conditional low-rank refinement / MoE-like update

shift + diagonal + compose
  -> convolution-like local operator

memory_read + product_gate + low_rank
  -> attention-like selective update

slot_A - slot_B + margin_gate
  -> contrast / class-pair decision operator
```

### 1.4 Recursion constraints

Do not allow arbitrary recursion. Require:

```text
max_transform_depth
max_active_primitives
max_chain_len
complexity cost
similarity/duplicate penalty
functional/heldout validation
residual usefulness
macro promotion only after test
```

Bad:

```text
any primitive with any primitive at any depth, accepted because train loss moved once
```

Good:

```text
small candidate subprogram -> tested on heldout/function -> promoted if useful and non-duplicate
```

---

## 2. Read/write symmetry and asymmetry

Read and write should share an address-space interface, but they are not semantically symmetric.

### 2.1 Shared interface

Both can address:

```text
state cells
memory cells
global cells
evidence cells
output slots
role/macro summaries
```

This gives a clean grammar:

```text
read(source_address)
write(target_address)
```

### 2.2 Read is gather/search

Read answers:

```text
Where should information come from?
```

Read may be broad and exploratory:

```text
state + memory + global
local window + evidence
similar slots
class/head relevant slots
```

Read can tolerate higher entropy early because it gathers candidates.

### 2.3 Write is commit/update

Write answers:

```text
What should be changed and where?
```

Write must be more conservative because it affects future computation.

Write needs:

```text
write gate
residual path
norm/stability
budget
anti-overwrite penalty
memory/global write cost
```

Design rule:

```text
read = freer search/gather
write = gated commit/update
```

Do not mirror regularization exactly.

Recommended:

```text
read entropy can be higher
write budget stronger
memory write more expensive than memory read
global write more expensive than global read
state write protected by residual/norm
```

---

## 3. Memory integration

Memory should be a first-class addressable space inside the grammar, not a separate magic module.

### 3.1 Addressable memory

The core address set should include:

```text
state cells
memory cells
global cells
evidence cells
output slots
```

Then memory naturally participates:

```text
read(source=memory)
transform(...)
write(target=memory)
```

### 3.2 Memory operations

Minimum memory grammar:

```text
memory_read
memory_write
memory_keep
memory_forget / overwrite gate
```

Soft update:

```text
M_new = keep_gate * M_old + write_gate * update
```

### 3.3 Memory controller

Memory needs a lightweight controller, but it should remain inside the grammar.

Track:

```text
memory_usage
memory_age
memory_importance
memory_overwrite_risk
memory_recall_score
```

Use these as weak biases/costs:

```text
read_memory_bias
write_memory_bias
overwrite_penalty
staleness penalty
importance protection
```

### 3.4 What not to do

Do not make memory only external to steps.

Do not let memory write be unconstrained.

Do not use memory as a hidden bypass that ignores the matrix-program grammar.

---

## 4. Compose as an explicit subgrammar

A single transform output is often not enough. Add `compose` between transform and write:

```text
read -> transform(s) -> compose -> write
```

Compose operations:

```text
sum
weighted sum
product/gate
residual add
difference / contrast
concat + projection
softmax mix
normalization
```

This lets the grammar express:

```text
low_rank(x) + diff(x)
gate(a, b)
slot_A - slot_B
residual(transform, x)
local_feature * global_context
```

`compose` should also be bounded:

```text
max_inputs
complexity cost
entropy/load balance
similarity penalty
```

---

## 5. Soft branches / mixture of paths

Some tasks require different operations for different examples or slots.

Add soft branches, not hard if-statements:

```text
path_1 = transform_1(read)
path_2 = transform_2(read)
path_3 = transform_3(read)

out = sum_i router_i * path_i
```

This is MoE-like but remains differentiable.

Constraints:

```text
max_branches
path load balancing
path dropout
entropy band
complexity cost
no hard routing in forward MVP
```

Useful for:

```text
local onset vs global shape
memory compare vs direct state transform
rare class path vs common class path
background rejector vs normal classifier
```

---

## 6. Loop / recursion / iterative refinement

Do not implement unbounded recursion.

Use:

```text
fixed max steps + soft step_alive
```

Each step:

```text
update = read -> transform -> compose -> write
cells = cells + step_alive * update
```

Possible stop signals:

```text
step_alive learned gate
delta_norm small
confidence high
critic predicts no more gain
complexity budget exhausted
```

But all must be soft in forward.

No Python `while` loop in the differentiable core for MVP.

---

## 7. Compare / contrast as macro, not separate hard block

Comparison is important for:

```text
attention-like selection
memory retrieval
copy/matching
class-pair confusion
head decision programs
```

But it can be expressed through read + compose:

```text
read A
read B
compose difference/product/similarity
write contrast slot
```

Useful macros:

```text
slot_contrast
class_pair_contrast
memory_query_match
local_global_compare
prototype_match
```

Do not add a totally separate compare module unless this macro path fails.

---

## 8. Growth and promotion

Growth should not happen as uncontrolled branching inside every forward pass.

Separate online compute from outer-loop growth.

### 8.1 Online forward

```text
uses current OperatorBank / MacroBank / HeadMacroBank
all choices soft and differentiable
```

### 8.2 Outer growth loop

```text
observe residual/error/confusion
propose candidate operator/macro/head-macro
Tier-1 screen
Tier-2 verify if needed
check duplicate/similarity
check complexity
promote if useful
```

### 8.3 Promotion principle

```text
repeated useful subprogram -> first-class macro/operator
```

Examples:

```text
diff + smooth + gate -> OnsetLikeMacro
low_rank + product_gate + residual -> ConditionalLowRankMacro
read_slot_A - read_slot_B + margin_gate -> ClassPairContrastMacro
```

---

## 9. What else grammar may need

Minimum extended grammar:

```text
READ:
  choose sources and gather information

TRANSFORM:
  apply bounded operator subprogram

COMPOSE:
  combine transformed signals

WRITE:
  gated commit to state/memory/global/output

CONTROL:
  soft branch, step_alive, complexity budget

MEMORY:
  read/write/keep/forget addressable memory

GROWTH:
  tested outer-loop promotion of useful subprograms
```

This can be summarized as:

```text
read -> transform -> compose -> write
          ↑             ↓
       branch/control  memory
          ↓             ↓
       growth outside online forward
```

---

## 10. What not to add yet

Avoid early:

```text
hard if/else branches
unbounded recursion
random sampling in main forward
unverified operator birth
dense macro growth everywhere
memory writes without overwrite budget
head macros that bypass core and memorize input
```

Prefer:

```text
soft mixtures
fixed max depth
step_alive
Gumbel/top-k only for diagnostic/export later
heldout-tested growth
sparse promoted macros
```

---

## 11. How this connects to existing plans

### Latent roles

`LATENT_ROLE_SPONTANEOUS_SPECIALIZATION_PLAN.md` says roles should emerge, not be named phases.

This plan says what every role is allowed to do:

```text
read / transform / compose / write / control / memory
```

### Input/head structure priors

`INPUT_HEAD_STRUCTURE_PRIOR_PLAN.md` gives weak hints about what data has and what task needs.

This plan gives the grammar those hints can bias.

### Head macro growth

`HEAD_MACRO_GROWTH_PLAN.md` grows decision/read macros at the head.

Those macros are special cases of:

```text
read -> compare/compose -> write logit delta
```

### Actor-critic mind

`PROGRAM_EDITOR_ACTOR_CRITIC_POLICY_PLAN.md` should use this grammar as the action space:

```text
edit read
edit transform subprogram
edit compose
edit write target
edit memory gate
edit branch weights
edit step_alive
promote macro after tested gain
```

---

## 12. First implementation target

Do not implement everything at once.

MVP order:

1. Make `compose` explicit in logs and aux outputs.
2. Make memory a first-class address source/target with usage/overwrite stats.
3. Add bounded transform depth / product-step macro support.
4. Add soft branch path count 2 or 3 with load-balance.
5. Improve `step_alive` so depth can be chosen.
6. Add candidate growth only in outer loop.
7. Add diagnostics:
   - transform depth usage
   - compose type usage
   - branch entropy/load
   - memory read/write/keep/forget
   - write overwrite risk
   - promoted macro duplicate similarity

---

## 13. Success criteria

Good:

```text
accuracy improves or same with lower complexity
transform subprograms become interpretable
memory read/write nonzero but not chaotic
write overwrite risk controlled
branch usage not collapsed or uniform forever
step_alive selects useful depth
promoted macros pass heldout/noise/duplicate checks
```

Bad:

```text
all paths become identical
memory becomes dump storage
write destroys state
transform depth always max
macros duplicate existing operators
train improves but heldout falls
```

---

## 14. Short answer

The grammar should stay minimal but not primitive-only.

`read -> transform -> write` is the base.

But `transform` may be a bounded subprogram, `compose` should be explicit, memory must be addressable with keep/forget, branches must be soft, loops must be fixed-depth with step_alive, and growth must be an outer tested promotion loop.

This gives enough expressivity without turning the architecture into uncontrolled program explosion.
