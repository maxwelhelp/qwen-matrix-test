# Read Transform Write Grammar Extension Plan

Goal: keep the outer matrix-program grammar minimal while making the internal `transform` expressive enough to cover attention-like, MLP-like, convolution-like, memory, MoE-like, recursive/refinement, and head-decision programs.

Outer grammar stays exactly:

```text
read -> transform -> write
```

This is the hard compute contract:

```text
read      = gather information from addressable sources
transform = compute/update candidate using a bounded internal program
write     = gated commit into addressable targets
```

Important rule:

```text
Keep the outer grammar minimal.
Put compose / compare / branch / loop inside transform.
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
every step must read, run a bounded transform-program, and write through a stable matrix interface
```

---

## 1. Final grammar shape

### 1.1 Outer contract

The only mandatory outer sequence is:

```text
READ -> TRANSFORM -> WRITE
```

Everything else is internal to `transform` or a property of `read/write`.

### 1.2 Transform internal grammar

```text
transform = primitive
          | compose(primitive or subtransform...)
          | compare(read_a, read_b)
          | branch(path_1, path_2, ..., router)
          | loop/refine(subtransform, step_alive)
          | macro(promoted tested subprogram)
```

So the grammar does not become:

```text
read -> transform -> compose -> branch -> memory -> control -> write
```

Instead it remains:

```text
read -> transform{primitive|compose|compare|branch|loop|macro} -> write
```

This preserves minimality while allowing complex programs.

---

## 2. Transform as a bounded recursive program

### 2.1 Transform can be composite

Examples:

```text
transform(x) = gate(low_rank(x) + diff(x))
transform(x) = compose(wavelet(x), product_gate(x), residual(x))
transform(x) = product_gate(low_rank(x), memory_read(x)) + residual(x)
transform(x) = compare(slot_A, slot_B) -> contrast_update
```

This matches the symbolic-operator direction:

```text
additive dictionary
ordered product grammar
functional validation
residual mining
macro promotion
```

### 2.2 Transform hierarchy

Use a bounded hierarchy, not unbounded recursion:

```text
Level 0: primitive operator
Level 1: weighted mixture / compose of primitives
Level 2: ordered product / short chain
Level 3: tested macro-step
Level 4: promoted operator/macro in OperatorBank/MacroBank/HeadMacroBank
```

### 2.3 Useful emergent examples

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
  -> contrast / class-pair decision transform
```

### 2.4 Recursion constraints

Require:

```text
max_transform_depth
max_active_primitives
max_chain_len
max_branch_paths
complexity cost
similarity/duplicate penalty
functional/heldout validation
residual usefulness
promotion only after test
```

Bad:

```text
any primitive with any primitive at any depth, accepted because train loss moved once
```

Good:

```text
small candidate transform-subprogram -> screened -> verified -> promoted if useful and non-duplicate
```

---

## 3. Compare is a fundamental transform primitive

Earlier plan treated compare mostly as macro. That is too weak.

Compare is needed everywhere:

```text
attention similarity
memory retrieval
copy/matching
branch/router decisions
class-pair contrast
head decision programs
```

Therefore compare should be a first-class transform primitive family, not only a promoted macro.

### 3.1 Compare forms

```text
difference:        a - b
product:           a * b
bilinear score:    a^T W b
cosine/dot score:  normalize(a) dot normalize(b)
gated contrast:    gate(a, b) * (a - b)
query-key match:   q(read_a) dot k(read_b)
```

### 3.2 Compare still uses read and write

It does not break the outer grammar:

```text
read -> transform.compare(...) -> write
```

Compare may internally do two reads or use two read slots from the outer read packet.

### 3.3 Compare constraints

```text
compare cost
score norm / z-loss
entropy band for match distribution
no dense all-pairs by default unless budget allows
local/top-k approximations where needed
```

---

## 4. Compose lives inside transform

Compose is not a fourth outer step. It is an internal transform operator.

```text
transform = compose(t1, t2, ..., mode)
```

Compose modes:

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

This lets transform express:

```text
low_rank(x) + diff(x)
gate(a, b)
slot_A - slot_B
residual(transform, x)
local_feature * global_context
```

Compose constraints:

```text
max_inputs
complexity cost
entropy/load balance
similarity penalty
```

---

## 5. Soft branch lives inside transform

Branching should not be an outer grammar step.

Use:

```text
transform = branch({path_i}, router)
path_i = subtransform_i(read_packet)
out = sum_i router_i * path_i
```

This is MoE-like but differentiable.

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

## 6. Loop/refinement lives inside transform

Do not use unbounded recursion or Python `while` in the differentiable core.

Use:

```text
transform = loop(subtransform, max_inner_steps, step_alive)
```

or use the existing outer step stack:

```text
for step in max_steps:
  update = read -> transform -> write
  cells = cells + step_alive * update
```

Possible soft stop signals:

```text
step_alive learned gate
delta_norm small
confidence high
critic predicts no more gain
complexity budget exhausted
```

All remain soft in forward.

---

## 7. Read/write symmetry and asymmetry

Read and write share the address-space interface, but they are not semantically symmetric.

### 7.1 Shared address space

Both can reference:

```text
state cells
memory cells
global cells
evidence cells
output slots
role/macro summaries
```

Interface:

```text
read(source_address)
write(target_address)
```

### 7.2 Read is gather/search

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

Read can tolerate higher entropy early.

### 7.3 Write is commit/update

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

Rule:

```text
read = freer search/gather
write = gated commit/update
```

Recommended:

```text
read entropy can be higher
write budget stronger
memory write more expensive than memory read
global write more expensive than global read
state write protected by residual/norm
```

---

## 8. Memory integration

Memory should be a first-class addressable space inside the grammar, not an external magic buffer.

### 8.1 Addressable memory

Address set:

```text
state cells
memory cells
global cells
evidence cells
output slots
```

Then memory participates naturally:

```text
read(source=memory)
transform(...)
write(target=memory)
```

### 8.2 Memory operations

Minimum memory operations:

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

### 8.3 Memory controller

Track:

```text
memory_usage
memory_age
memory_importance
memory_overwrite_risk
memory_recall_score
```

Use as weak biases/costs:

```text
read_memory_bias
write_memory_bias
overwrite_penalty
staleness penalty
importance protection
```

Do not let memory become an unconstrained bypass.

---

## 9. Growth and promotion

Growth should not happen as uncontrolled branching inside every forward pass.

Online forward:

```text
uses current OperatorBank / MacroBank / HeadMacroBank
all choices soft and differentiable
```

Outer growth loop:

```text
observe residual/error/confusion
propose candidate transform/operator/macro/head-macro
Tier-1 screen
Tier-2 verify if needed
check duplicate/similarity
check complexity
promote if useful
```

### 9.1 What counts as useful?

A candidate is useful only if it improves real behavior, not only train loss.

Use a utility score:

```text
utility = heldout_gain
        + functional_gain
        + target_subproblem_gain
        - complexity_cost
        - duplicate_penalty
        - instability_penalty
        - compute_cost
```

Where:

```text
heldout_gain          = baseline_val_loss - candidate_val_loss
functional_gain       = behavior/output error improvement on heldout probes
                         or reconstruction/functional error improvement for matrix/operator decode
target_subproblem_gain= class-pair margin gain, memory recall gain, residual reduction, etc.
complexity_cost       = extra depth, branches, active primitives, memory writes, macro params
duplicate_penalty     = similarity to existing operators/macros
instability_penalty   = logit norm spike, overwrite risk, variance across microbatches
compute_cost          = extra wall-clock / FLOPs / memory
```

### 9.2 Noise-gated acceptance

Estimate noise dynamically:

```text
noise_std = std(loss on repeated heldout microbatches)
threshold = max(k * noise_std, relative_min_gain * loss_scale)
```

Accept only if:

```text
utility_gain > threshold
and global heldout does not worsen
and candidate is not duplicate
and complexity budget is respected
```

Reject if:

```text
utility_gain < -threshold
or global heldout worsens
or overwrite/instability exceeds limit
```

Otherwise mark `uncertain`, not accepted.

### 9.3 Tier-1 screen

Forward-only:

```text
apply temporary bias/candidate
measure heldout microbatch losses and subproblem metrics
revert
```

### 9.4 Tier-2 verify

For top candidates only:

```text
save full training state
apply candidate
run N small adaptation steps or short validation window
measure global + subproblem gain
restore or keep
```

Rollback must restore:

```text
model weights
optimizer state
AMP scaler
scheduler state
RNG state if needed
running stats if any
```

### 9.5 Promotion principle

```text
repeated useful subprogram -> first-class macro/operator
```

Examples:

```text
diff + smooth + gate -> OnsetLikeTransform
low_rank + product_gate + residual -> ConditionalLowRankTransform
compare(slot_A, slot_B) + margin_gate -> ClassPairContrastTransform
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

## 11. Connections to existing plans

### Latent roles

`LATENT_ROLE_SPONTANEOUS_SPECIALIZATION_PLAN.md` says roles should emerge, not be named phases.

This plan defines what every role can do through the same outer contract:

```text
read -> transform{primitive|compose|compare|branch|loop|macro} -> write
```

### Input/head structure priors

`INPUT_HEAD_STRUCTURE_PRIOR_PLAN.md` gives weak hints about what data has and what task needs.

Those hints should bias read/transform/write choices, not create fixed layer roles.

### Head macro growth

`HEAD_MACRO_GROWTH_PLAN.md` grows decision/read macros at the head.

Those macros are head-specific transforms:

```text
read -> transform.compare/compose -> write logit delta
```

### Actor-critic mind

`PROGRAM_EDITOR_ACTOR_CRITIC_POLICY_PLAN.md` should use this grammar as the action space:

```text
edit read sources
edit transform subprogram
edit compare operator
edit branch weights
edit write targets
edit memory gates
edit step_alive
promote macro after tested gain
```

---

## 12. First implementation target

Do not implement everything at once.

MVP order:

1. Keep outer code/API named `read -> transform -> write`.
2. Make `compose` an internal transform type in logs and aux outputs.
3. Promote `compare` to a first-class transform primitive family.
4. Make memory a first-class read/write address with usage/overwrite stats.
5. Add bounded transform depth / product-step macro support.
6. Add optional soft branch inside transform with path count 2 or 3.
7. Improve `step_alive` so depth can be chosen.
8. Add growth only in outer loop with utility/noise/duplicate gates.
9. Add diagnostics:
   - transform type usage
   - transform depth usage
   - compose mode usage
   - compare type usage
   - branch entropy/load
   - memory read/write/keep/forget
   - write overwrite risk
   - candidate utility components
   - promoted macro duplicate similarity

---

## 13. Success criteria

Good:

```text
accuracy improves or same with lower complexity
transform subprograms become interpretable
compare is used for attention/memory/head confusion cases
memory read/write nonzero but not chaotic
write overwrite risk controlled
branch usage not collapsed or uniform forever
step_alive selects useful depth
promoted macros pass heldout/noise/duplicate checks
```

Bad:

```text
all transform paths become identical
memory becomes dump storage
write destroys state
transform depth always max
compare becomes dense all-pairs bottleneck
macros duplicate existing operators
train improves but heldout falls
candidate accepted without measured utility
```

---

## 14. Short answer

Keep the external grammar minimal:

```text
read -> transform -> write
```

Make transform the recursive container:

```text
transform = primitive | compose | compare | branch | loop | macro
```

Read and write share addresses but are asymmetric: read searches, write commits.

Memory is a first-class address space with read/write/keep/forget and overwrite cost.

Growth is an outer tested promotion loop, and candidate usefulness is measured by heldout/functional/subproblem gain minus complexity, duplicate, instability, and compute costs.

This keeps minimality without losing expressive power.
