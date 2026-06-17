# Latent Role Spontaneous Specialization Plan

Goal: replace fixed human-named layer roles like `extract/compare/suppress/aggregate` with learned anonymous latent roles that can specialize spontaneously under task pressure, diversity constraints, editor feedback, and later actor-critic memory.

This plan is for the MatrixProgramAssembler family.

---

## 1. Main problem

Current assembler is not fully role-free.

Even if `phase_prior_strength=0` is added at runner level, the original core has role assumptions:

```text
PHASES = extract / compare / suppress / aggregate
self.phase = PHASES[min(layer, len(PHASES)-1)]
phase_mix_logits starts biased by layer index
primitive_slot_prior depends on self.phase
primitive_transition_prior depends on self.phase
```

This means layers can learn, but they start inside our human-designed role frame.

The new goal is not:

```text
L0 = extract
L1 = compare
L2 = suppress
L3 = aggregate
```

The new goal is:

```text
anonymous role slots emerge from data and training pressure
```

After training we may interpret them and name them, but the model should not receive those names as architecture facts.

---

## 2. What should stay universal

Universal core contract:

```text
cells / memory / global cells
read_flow
primitive_flow
operator-size/depth flow
primitive_transition_flow
slot_transition_flow
composition_flow
write_flow
step_alive
macro selection
editor edits
future feedback / critic / actor signals
```

Universal operation language:

```text
read -> operator family -> transition -> compose -> write
```

Universal safety/learning pressures:

```text
slot diversity
class-read diversity
role diversity
memory/global usage control
complexity budget
anti-collapse entropy band
soft-vs-discrete diagnostics
heldout/bi-level architecture training
```

---

## 3. What should be task-specific

### 3.1 Input adapter

The first layer/input adapter is task-specific.

It converts task data into universal evidence cells:

```text
audio waveform/mel -> evidence cells
text tokens -> evidence cells
image patches -> evidence cells
code/AST/matrix -> evidence cells
```

Required output shape:

```text
[N, evidence_cells, D]
```

The adapter can be different per modality, but the format after it must be universal.

### 3.2 Output head

The final head is task-specific.

Examples:

```text
classification: class queries read program slots/cells -> logits
generation: token queries read program cells -> next token/regression
replacement layer: program cells -> residual hidden update
matrix decode: program cells -> recipe/flow predictions
```

Universal requirement:

```text
head reads the same program cell/slot interface
```

Do not force one universal first or last layer. Force a universal middle language.

---

## 4. Replace named phases with latent roles

### 4.1 Old design

```text
PHASES = extract / compare / suppress / aggregate
layer index -> phase name
phase name -> primitive prior
phase name -> transition prior
```

### 4.2 New design

Use anonymous latent roles:

```text
role_count = R, e.g. 6 or 8
role_embed[R, D]
role_logits[L, S, R]
role_mix[L, S] = softmax(role_logits[L, S])
role_context[L, S] = Σ_r role_mix[L,S,r] * role_embed[r]
```

No role has a human name at initialization.

Role embeddings generate biases:

```text
role_to_read(role_context) -> read bias
role_to_primitive(role_context) -> primitive bias
role_to_operator_variant(role_context) -> operator-size/depth bias
role_to_primitive_transition(role_context) -> transition bias
role_to_slot_transition(role_context) -> slot transition bias
role_to_composition(role_context) -> composition bias
role_to_write(role_context) -> write bias
role_to_macro(role_context) -> macro bias
role_to_step_alive(role_context) -> step alive bias
```

These biases are additive and soft:

```text
read_logits += role_read_bias
primitive_logits += role_primitive_bias
transition_logits += role_transition_bias
write_logits += role_write_bias
```

---

## 5. What must be mandatory inside every layer/step

Spontaneous specialization does not mean no structure. Some contracts must remain mandatory so training does not collapse.

Every step must have:

1. **read path**
   - must read from state/memory/global/evidence address cells through soft read.

2. **operator path**
   - must process read context through OperatorBank.

3. **transition path**
   - must allow primitive and slot interactions.

4. **composition path**
   - must combine primitive slots into update.

5. **write path**
   - must write to address cells.

6. **residual/state continuity**
   - cells should preserve useful state and not be overwritten chaotically.

7. **step alive gate**
   - each step can become weak/strong, but max available steps are present.

8. **complexity cost**
   - deeper/larger operators and extra alive steps have a price.

These are the grammar of the matrix program. The roles inside the grammar should emerge.

---

## 6. What should not be mandatory

Do not mandate:

```text
layer 0 must extract
layer 1 must compare
layer 2 must suppress
layer 3 must aggregate
phase names
specific primitive chains per layer
fixed macro per layer
fixed use of memory/global per layer
```

Those can be discovered or promoted later.

---

## 7. How spontaneous roles emerge

Spontaneous roles need conditions. They will not reliably emerge from pure uniform initialization.

Use these pressures:

### 7.1 Weak random role identity

Each layer/step gets small random identity:

```text
layer_step_key ~ N(0, small_std)
role_logits[L,S] initialized near uniform + tiny noise
role_embed random small
```

This breaks symmetry without naming roles.

### 7.2 Role diversity loss

Different layers/steps should not choose identical role mixtures.

```text
role_sim_loss = similarity(role_mix[L,S], role_mix[L',S'])
```

Use softly. Too strong diversity can force useless differences.

### 7.3 Program-output diversity

Different layers/steps should not produce identical slot/cell summaries.

Already proposed:

```text
layer_sim_loss
step_sim_loss
slot_div_loss
```

### 7.4 Role usage balance

Avoid all layers using one role.

```text
role_usage = mean_{L,S}(role_mix[L,S])
role_balance_loss = distance(role_usage, not-collapsed distribution)
```

Do not force perfectly uniform forever. Use entropy band:

```text
not too collapsed, not too uniform
```

### 7.5 Role entropy annealing

Early:

```text
role entropy high enough to explore
```

Later:

```text
allow sharper role assignments
```

Temperature schedule:

```text
tau_role starts 1.5-2.0
anneal to 0.7-1.0
```

### 7.6 Complexity and usefulness

A role is useful only if it improves heldout/task loss. Use:

```text
complexity penalty for alive steps / large operators
heldout bi-level architecture updates
counterfactual probes later
```

---

## 8. OperatorBank under latent roles

Operator selection should also be role-conditioned but not role-fixed.

Operator families:

```text
channel_butterfly variants
block_butterfly variants
low_rank sizes
ctx/local smooth/diff/diag variants
product/gate variants
phase variants
future Fourier/DCT/Wavelet/Toeplitz variants
```

Role controls additive bias over operator variants:

```text
operator_variant_logits += role_to_operator_variant(role_context)
```

But all operators remain available everywhere.

No role should hard-disable an operator.

---

## 9. MacroBank under latent roles

Macros should not be tied to named phases.

Instead:

```text
macro_selector_logits[L,S,B,M] += role_to_macro(role_context[L,S])
```

A macro can become popular in any layer/step if useful.

Macro content embeddings should be content-based:

```text
read prototype + primitive prototype + transition prototype + composition prototype + write prototype
```

not integer id only.

---

## 10. Editor / actor-critic compatibility

Latent roles do not break the future mind. They make it freer.

Editor can edit:

```text
role_logits
role_to_flow biases
read/primitive/transition/write directly
operator variants
macro selection
step_alive
```

Critic can evaluate edits like:

```text
increase role3 at L1.S2
shift role5 to use more memory reads
increase low_rank_r16 inside role2
suppress macro7 when role4 dominates
activate extra step with role1
```

Actor should operate on anonymous roles and content embeddings, not human role names.

---

## 11. Required logging

Without logging, we cannot know whether roles emerged.

Log every analysis epoch:

```text
role_mix[L,S,R]
role_usage[R]
role_entropy[L,S]
role_similarity matrix
layer_sim
step_sim
slot_div
class_read_div
primitive mix by role
operator variant mix by role
read/write cell usage by role
primitive transition top pairs by role
macro usage by role
step_alive[L,S]
```

Then produce human interpretation after training:

```text
role0 behaves like local smoothing/extraction
role1 behaves like memory compare
role2 behaves like gated suppressor
role3 behaves like global aggregate
```

Names are assigned after observation, not before training.

---

## 12. Implementation plan

### Stage A: true phase removal

Patch core permanently, not only runtime script:

1. Remove `self.phase = PHASES[min(layer,...)]` from behavior.
2. Keep `PHASES` only for old report compatibility or delete later.
3. Set `phase_mix_logits` neutral or replace with latent role logits.
4. Remove phase-dependent primitive priors.
5. Remove phase-dependent primitive transition priors.
6. Ensure `phase_prior_strength=0` truly means no phase behavior.

### Stage B: latent roles MVP

Add config:

```text
role_count=6
role_embed_dim=D
role_temperature
role_init_std
```

Add parameters:

```text
role_embed[R,D]
role_logits[L,S,R]
role_to_read
role_to_primitive
role_to_operator_variant
role_to_transition
role_to_write
role_to_macro
role_to_step_alive
```

For MVP, implement only:

```text
role_to_primitive
role_to_transition
role_to_read/write
role_to_step_alive
```

Add other projections later.

### Stage C: diversity and anti-collapse

Add losses:

```text
role_usage_balance_loss
role_entropy_band_loss
role_similarity_loss
layer_program_similarity_loss
step_program_similarity_loss
slot_diversity_loss
class_read_diversity_loss
```

Start small.

### Stage D: phase-free real task test

Run:

```text
4 layers x 3 steps
role_count=6
no named phases
OperatorBankV2 on
full mode
20 epochs
```

Compare against:

```text
old phase-prior OperatorBankV2 5x3 best 60.45
phase-free no latent roles
latent-role v1
```

### Stage E: synthetic planted-role tests

Create small synthetic tasks where useful roles should emerge:

```text
local extract needed
memory compare needed
gate suppress needed
global aggregate needed
multi-step composition needed
```

Do not label roles. Check if anonymous role usage separates.

### Stage F: connect editor feedback later

Only after roles emerge:

```text
feedback_bias into role logits
critic evaluates role edits
actor applies sparse LCB-safe role edits
```

---

## 13. What could go wrong

### 13.1 All roles collapse to one

Fix:

```text
increase role_usage_balance
increase role_similarity penalty
increase role entropy early
add role dropout
```

### 13.2 Roles stay uniform forever

Fix:

```text
anneal role temperature
add role specialization reward via heldout bi-level
increase role_to_flow projection capacity
reduce too-strong balance loss
```

### 13.3 Random specialization hurts task loss

Fix:

```text
reduce diversity losses
use complexity/heldout gating
allow some layers to share roles
```

### 13.4 Actor/critic overfits role names

Fix:

```text
use content embeddings and behavior summaries
not integer role id only
```

---

## 14. Minimal first experiment

Do not build full actor-critic yet.

First build:

```text
latent_role_assembler_v1
role_count=6
4 layers x 3 steps
OperatorBankV2
no named phase priors
role diversity + layer/step similarity
```

Success criteria:

```text
accuracy close to phase-prior baseline, ideally >=55-60%
role_usage not collapsed
role_mix differs across layers/steps
class_read_div decreases
slot_div decreases
read/write entropy sharpens
memory/global used nonzero
```

If this works, the assembler no longer depends on our hand-written roles.

---

## 15. Short answer

Some structure must remain mandatory: read, operators, transitions, compose, write, memory/global, residual state, step alive, complexity budget.

Human-named layer roles must not be mandatory.

Spontaneous specialization needs carefully designed conditions:

```text
anonymous latent roles
small random identity
role diversity
role usage balance
entropy schedule
task gradient
heldout/bi-level updates
later editor/critic feedback
```

This gives the system freedom to discover useful roles on any task without being trapped in our extract/compare/suppress/aggregate frame.
