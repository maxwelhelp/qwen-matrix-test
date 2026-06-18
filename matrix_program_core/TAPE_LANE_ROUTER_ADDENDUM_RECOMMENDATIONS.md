# Tape Lane Router Addendum: Separator Growth, Structured I/O, and Primitive Meaning

This addendum answers implementation questions that were not explicit enough in `TAPE_LANE_ROUTER_ARCHITECTURE_PLAN.md`.

Core questions:

1. What exactly is the separator matrix?
2. Does the separator grow, or does the tape grow?
3. Should structured input/output be added now?
4. What are primitives if the only external grammar is `read -> transform -> write`?

---

## 1. Recommendation summary

Do not build a completely new huge system immediately.

Build this MVP:

```text
fixed task input adapter
  -> structured evidence cells
  -> one program tape with T_max steps
  -> lane route matrix R_t[from_lane,to_lane]
  -> boundary gate b_t
  -> step_alive gate a_t
  -> transform primitives inside each step
  -> ClassMatrixLaneHead
```

The separator is not a single scalar and not a hard boundary.

The separator is a pair:

```text
separator_t = boundary_t + route_matrix_t
```

Where:

```text
boundary_t      says whether step t begins/ends a soft segment
route_matrix_t  says how information moves between lanes at step t
```

---

## 2. What exactly is the separator?

Use two related objects.

### 2.1 Boundary gate

```python
boundary_t = sigmoid(boundary_logit[t])
```

Meaning:

```text
soft segment split / new program region
```

It is scalar or optionally per-lane:

```text
scalar:  boundary[t]
per-lane: boundary[t,lane]
```

MVP: scalar boundary.

### 2.2 Lane route matrix

```python
R_t = softmax(route_logits[t], dim=-1)  # [from_lane, to_lane]
```

Meaning:

```text
how updates move between detail/state/abstract/memory lanes
```

Example:

```text
state -> state:    0.60
state -> abstract: 0.20
state -> memory:   0.10
detail -> state:   0.10
```

Together:

```text
boundary_t decides segment pressure;
R_t decides information transport.
```

Boundary can bias route, but should not destructively reset state in MVP.

---

## 3. Does the separator matrix grow?

There are two growth modes.

### 3.1 Differentiable growth inside one run

Do not dynamically change tensor shapes.

Allocate maximum capacity:

```text
T_max steps
L lanes
A cells per lane
```

Then grow by gates:

```text
step_alive[t] rises
boundary[t] rises
route_matrix[t] changes
operator usage in step t becomes nontrivial
```

This is differentiable.

A dormant step can become active without changing model shape:

```text
step t starts almost dead
training increases step_alive[t]
training gives it read/transform/write usage
route matrix gives it a place in the program
```

### 3.2 Outer-loop true growth between runs

After a run, if the analysis shows:

```text
many spare steps alive
boundary near the end
complexity budget saturated
heldout improves with more active steps
```

then the next run can increase capacity:

```text
T_max 12 -> 16
lanes 4 -> 5 optional
cells_per_lane 12 -> 16 optional
```

But this is a new run / promotion decision, not dynamic shape mutation during forward.

### 3.3 Should route matrix itself grow?

MVP: no.

Keep lanes fixed:

```text
L=4
R_t shape [4,4]
```

Later, if memory/control lane is overloaded or route entropy suggests missing capacity, add a lane in the next version:

```text
L=4 -> L=5
```

Use compatibility initialization:

```text
old R copied into top-left block;
new lane gets weak self-loop and weak connections.
```

Do not implement lane-count growth in the first version.

---

## 4. How the model chooses where to place computation

A step's place in the program is not chosen by a hard index like `L1.compare`.

It is determined by four learned signals:

```text
step_alive[t]
boundary[t]
R_t[from,to]
read/write source groups
```

Interpretation:

```text
step_alive high       => this step matters
boundary high         => this step starts/ends a segment
R_t detail->state     => local/detail information is being processed into state
R_t state->abstract   => abstraction/compression
R_t state->memory     => memory write/control
R_t memory->state     => retrieval/recall
```

So the model does not need a hard layer label.

A segment emerges when several adjacent steps share similar routing/operator behavior and are separated by high boundary gates.

---

## 5. Should structured input be added?

Yes, but not as a huge new system in MVP.

Add minimal structured input now.

### 5.1 Why

If the input is just generic evidence cells, the tape must discover all structure from scratch.

Structured input gives weak hints:

```text
local/detail information
frequency/onset/diff information
global/summary information
noise/smoothness information
```

This helps lane routers specialize.

### 5.2 MVP structured input for audio

Use the existing `MatrixEvidence` as base, but split/init lanes differently:

```text
lane0 detail/raw/local:
  evidence cells, local mel/time details, diff/onset-like features

lane1 working state:
  learned block queries over evidence

lane2 abstract/global:
  pooled/global summary cells

lane3 memory/control:
  learned memory seed + weak input summary
```

Do not overengineer. Just initialize lanes with different weak views.

Pseudo-code:

```python
evidence = MatrixEvidence(wav)  # [N,E,D]
local = evidence
state = attention_init(block_queries, evidence)
global_summary = evidence.mean(dim=1, keepdim=True).expand(N, A, D)
memory_seed = learned_memory + 0.05 * global_summary
X = stack_lanes(local_proj, state, global_summary, memory_seed)
```

### 5.3 Structured input should be read-only

Evidence/input cells should not be overwritten.

The tape can read them, but writes go to lanes/state/memory/output.

Use input-read curriculum:

```text
early steps can read evidence cheaply;
later steps pay cost until unlock.
```

---

## 6. Should structured output be added?

Yes.

Use `ClassMatrixHead` idea from v3, but replace phase groups with lane/segment groups.

### 6.1 Keep from v3

```text
class_state[C,D]
pair_state[P,D]
class_pair_logits[C,P]
class update from class_read + pair_ctx
class_write gate
class_read_div loss
pair_update_norm metric
```

### 6.2 Replace phase with lane/segment

Old:

```text
class_phase_logits[C,phase]
phase_slot_matrix[phase,slot]
phase_balance_loss
```

New MVP:

```text
class_lane_logits[C,lane]
lane_slot_matrix[lane,slot]
lane_balance_loss
```

Later:

```text
class_segment_logits[C,soft_segment]
segment_slot_matrix[segment,slot]
```

### 6.3 Why this matters

The head should not force class reads to `extract/compare/suppress/aggregate`.

It should learn:

```text
this class reads detail lane more;
that class reads memory/control more;
another class reads abstract/global more;
class-pair repair uses pair_state.
```

This keeps the strong v3 head idea but removes hard phase roles.

---

## 7. What are primitives if transform is the only compute part?

The outer grammar is:

```text
read -> transform -> write
```

But transform is not one operation.

Transform is a container with primitive families.

### 7.1 Current primitive families from v2/v3

```text
channel_butterfly
block_butterfly
low_rank
ctx_matrix
product_gate
phase_matrix
```

In the new version, do not keep `phase_matrix` as phase-specific.

Rename/redefine:

```text
phase_matrix -> generic_mix / learned_context_transform
```

### 7.2 Recommended primitive taxonomy

Use these categories.

#### Channel primitives

Operate along feature dimension `D`:

```text
channel_butterfly
low_rank
linear_ctx
product_gate
residual_gate
```

#### Cell/block primitives

Operate across cells inside a lane:

```text
block_butterfly
local_shift
smooth/diff
block_mix
```

#### Compare primitives

First-class transform family:

```text
difference a-b
product a*b
dot/cosine
bilinear q^T W k
gated_contrast
```

#### Memory/control primitives

```text
memory_read_match
memory_write_gate
keep/forget
importance/protection gate
```

#### Compose primitives inside transform

```text
sum
weighted sum
product/gate
residual add
difference/contrast
softmax mix
```

#### Macro primitives later

```text
promoted useful subprograms:
  diff+smooth+gate
  low_rank+product_gate+residual
  compare+margin_gate
```

### 7.3 Why read/write are not primitives

Read and write are dataflow operations.

Transform primitives are compute operations.

Read decides:

```text
what information to process
```

Transform decides:

```text
how to process it
```

Write decides:

```text
where and how strongly to commit it
```

So even if most computation is in transform, read/write are still essential.

Bad read means the right transform sees wrong data.
Bad write means the right transform result is stored in the wrong place or destroys memory.

---

## 8. Recommendations for the first new version

Implement only these in v4 tape-lane MVP:

```text
1. Structured lane initialization from evidence.
2. Fixed lanes L=4.
3. Fixed T_max=12 steps.
4. step_alive[t].
5. boundary[t].
6. route_matrix[t,from,to].
7. lane-aware read groups.
8. existing primitive families without phase if-statements.
9. ClassMatrixLaneHead.
10. full logging/report.
```

Do not implement yet:

```text
true dynamic lane growth;
active actor/critic;
macro promotion;
hard boundaries;
top-k routers;
unbounded transform recursion.
```

---

## 9. Key tests

### 9.1 Does tape avoid hidden phase roles?

Search for:

```bash
grep -R "PHASES\|phase_names\|_primitive_prior(phase)\|if self.phase\|extract\|compare\|suppress\|aggregate" -n simple_butterfly_matrix_v4_tape_lane matrix_program_core
```

Allowed only in comments/reports comparing old versions.

### 9.2 Does lane routing actually learn?

Check:

```text
route_matrix not all identity;
route entropy not max forever;
some state->abstract or state->memory appears;
boundaries not all zero/one;
step_alive not all equal.
```

### 9.3 Does structured input cause shortcut?

Check:

```text
late_input_read_mass not too high early;
late steps read state/memory/abstract;
input-read unlock works later.
```

### 9.4 Does head use lanes?

Check:

```text
class_lane_mass differs by class;
head does not read only final lane/last step;
pair_update_norm nonzero;
class_read_div decreases.
```

---

## 10. Short answer for the agent

The separator is not just a boundary and not just a route matrix.

Use:

```text
separator_t = boundary_t + R_t[from_lane,to_lane]
```

Growth in one run is soft:

```text
step_alive grows;
boundary grows;
route matrix changes;
dormant steps become active.
```

True shape growth happens only between runs.

Add structured input and structured output, but keep them weak:

```text
input adapter gives lane-specific evidence;
head reads lane/segment groups instead of phase groups.
```

Primitives are the internal vocabulary of transform:

```text
channel / block / low_rank / ctx / gate / compare / memory / compose
```

Read/write are not transform primitives; they are the dataflow contract around transform.
