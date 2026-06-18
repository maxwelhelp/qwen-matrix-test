# Input + Head Structure Prior Plan

Goal: create weak, data-driven and task-driven priors for both the input adapter and the output/head, while keeping the middle MatrixProgramCore free to specialize through latent roles, operator choices, memory, macros, editor feedback, and later actor-critic control.

Core idea:

```text
Input tells the core: what structure exists in the data.
Head tells the core: what structure the answer/task needs.
Middle core decides: how to assemble the matrix program.
```

This replaces rigid layer roles like `L0=extract, L1=compare, L2=suppress, L3=aggregate` with weak bidirectional hints:

```text
data structure hints -> operator/role/read/write priors
head/task hints -> output/read/class/query priors
```

The priors must be soft, learnable, weak, and overridable by task loss.

---

## 1. Why this is needed

The middle grammar `read -> transform -> write` is universal, but it is not enough to fully guide specialization.

Without any hints, roles can stay uniform:

```text
role_usage almost uniform
role_entropy close to max
role_similarity high
```

With hard named phase priors, specialization is biased by our design:

```text
L0 extract
L1 compare
L2 suppress
L3 aggregate
```

We need a middle path:

```text
weak structure priors from input and task/head
not hardcoded layer roles
```

---

## 2. Three-part architecture

```text
Task-specific Input Adapter + Structure Probe Bank
        ↓
universal evidence cells + structure tokens + input priors
        ↓
Free MatrixProgramCore
  read / operators / transitions / roles / macros / memory / editor
        ↓
universal program cells/slots + output structure tokens
        ↓
Task-specific Head + Task Probe Bank
```

Important separation:

1. input adapter is task/modality-specific;
2. output head is task-specific;
3. middle core is universal;
4. hints use a universal structure language.

---

## 3. Universal structure language

Different inputs and tasks should produce a shared set of structure tokens.

Suggested universal token families:

```text
LOCAL
GLOBAL
LOW_RANK
SPARSE
DENSE_MIX
BLOCK
DIAGONAL
BAND
PERIODIC
SMOOTH
DIFF
BOUNDARY
ONSET
COPY
COMPARE
FILTER_GATE
MEMORY
AGGREGATE
WRITE_HEAVY
CLASS_SEPARATION
SEQUENCE_ORDER
LONG_RANGE
SHORT_RANGE
NOISE_SUPPRESSION
```

These tokens are not commands. They are weak hints.

Example:

```text
audio onset -> BOUNDARY + DIFF + LOCAL + FILTER_GATE
matrix block structure -> BLOCK + LOW_RANK + SPARSE
classification head -> CLASS_SEPARATION + AGGREGATE + READ_SLOTS
sequence generation -> SEQUENCE_ORDER + MEMORY + LONG_RANGE
```

---

## 4. Input-side structure priors

### 4.1 StructureProbeBank

For each modality, build a structure probe bank.

Audio probes:

```text
local_energy
onset/diff_energy
smoothness
frequency_band_energy
wavelet-like burst score
periodicity
noise score
low_rank/compressibility score
global_context_need
memory_need
```

Matrix/code probes:

```text
diagonal_score
band_score
block_score
low_rank_score
sparsity_score
symmetry_score
toeplitz_score
periodicity_score
row/column locality
operator-sketch histogram
```

Text/token probes:

```text
local token continuity
content similarity
repeat/copy score
boundary/special-token score
long-range dependency score
causal/context need
```

Image probes:

```text
local edge/texture
patch smoothness
spatial locality
multi-scale structure
channel correlation
object/global score
```

### 4.2 Input structure embedding

```text
input_structure_tokens = probes(x)
input_structure_embedding = StructureTokenEncoder(input_structure_tokens)
```

### 4.3 Data-to-operator similarity

Each operator family has a learned or fixed content embedding:

```text
operator_embed[low_rank]
operator_embed[ctx_matrix]
operator_embed[product_gate]
operator_embed[channel_butterfly]
operator_embed[block_butterfly]
operator_embed[diff/smooth/wavelet]
operator_embed[memory_read]
operator_embed[global_read]
```

Compute weak hints:

```text
operator_hint_logits = input_structure_embedding @ operator_embed.T
```

Then project them to flow biases:

```text
primitive_logits += alpha_input * primitive_hint
operator_variant_logits += alpha_input * operator_variant_hint
read_logits += alpha_input * read_hint
write_logits += alpha_input * write_hint
role_logits += alpha_input * role_hint
macro_logits += alpha_input * macro_hint
```

Recommended start:

```text
alpha_input = 0.05 to 0.15
clamp bias to [-0.25, 0.25]
```

---

## 5. Head/task-side structure priors

The head should also provide weak hints. The head knows what kind of answer is needed.

### 5.1 TaskProbeBank

Task/head probes are not about raw input. They describe the output contract.

Classification:

```text
num_classes
class imbalance
class confusion vector
class query diversity need
slot separation need
read global vs read local ratio
```

Generation:

```text
causal dependency need
next-token/local need
long-range memory need
copy/retrieval need
sequence order need
```

Regression:

```text
smooth output need
global summary need
low-rank compression need
noise robustness need
```

Layer replacement / attention replacement:

```text
residual preservation need
hidden-state shape
token mixing need
value/write need
local/global attention need
patch sensitivity
```

Matrix-program decoding:

```text
recipe length need
operator-category prediction need
flow reconstruction need
step order need
macro use need
```

### 5.2 Head structure embedding

```text
head_structure_tokens = task/head probes
head_structure_embedding = HeadStructureEncoder(head_structure_tokens)
```

### 5.3 Head-to-core hints

The head can softly bias the program core:

```text
role_logits += alpha_head * role_hint_from_task
read_logits += alpha_head * read_hint_from_task
write_logits += alpha_head * write_hint_from_task
macro_logits += alpha_head * macro_hint_from_task
step_alive_logits += alpha_head * depth_hint_from_task
```

Examples:

Classification:

```text
+ class slot separation
+ final read/write stability
+ moderate global summary
+ diverse class queries
```

Generation:

```text
+ memory/sequence order
+ causal/local read
+ long-range retrieval
```

Replacement head:

```text
+ residual preservation
+ hidden-state continuity
+ local/global token mixing
```

### 5.4 Head internal priors

The head itself can have weak priors:

Classification head:

```text
class queries should not all read the same slot
class-slot attention entropy band
class-query diversity
optional weak class-slot prior, annealed down
```

Generation head:

```text
causal/local read prior
copy/memory read prior
output-token query diversity
```

Matrix recipe head:

```text
operator/category read diversity
step-order read hints
macro-read hints
```

---

## 6. Bidirectional data-task matching

Best version uses both input and head hints:

```text
input_structure_embedding = what data contains
head_structure_embedding = what task needs
joint_structure = fuse(input_structure_embedding, head_structure_embedding)
```

Then:

```text
joint_hint = JointStructureProjector(joint_structure)
```

This answers:

```text
this data has local/onset structure
and the task needs class separation
=> early/local + later class-readable slot separation
```

Another example:

```text
this data has low-rank/block structure
and the task needs matrix recipe reconstruction
=> bias low_rank/block operators and step-order slots
```

---

## 7. Keep middle core free

Do not use input/head hints to recreate hard phases.

Bad:

```text
if classification: L0 extract, L1 compare, L2 aggregate
```

Good:

```text
classification task raises weak CLASS_SEPARATION/GLOBAL/READ_SLOTS hints
input structure raises LOCAL/DIFF/LOW_RANK hints
latent roles decide where to use them
```

Middle core still decides:

```text
which layer/step uses which role
which operator goes where
which memory/global addresses matter
which macro is active
which steps are alive
```

---

## 8. Schedules and safety

Hints must be weak and annealable.

Recommended:

```text
alpha_input starts 0.10, cap 0.25
alpha_head starts 0.10, cap 0.25
hint dropout 0.05-0.20
hint clamp [-0.25, 0.25]
```

Annealing options:

1. Keep weak hints constant.
2. Start with stronger hints and decay if core learns better.
3. Use critic/heldout to decide if hints help.

Do not allow hints to dominate learned logits.

---

## 9. Relationship to self-organizing prior

Input/head hints are priors before/at training.

Self-organizing prior is behavior feedback during training:

```text
layer/step/component usage EMA -> weak lagged prior
```

They should combine:

```text
flow_logits = base
            + context
            + input_structure_hint
            + head_task_hint
            + latent_role_bias
            + self_organizing_usage_prior
            + editor_bias
            + later actor_bias
```

All hints must be:

```text
lag-safe
weak
clamped
detached when based on statistics
validated by heldout/counterfactual later
```

---

## 10. What can be universal and what cannot

Universal:

```text
structure token language
operator embeddings
role embeddings
matrix-program grammar
read/write/transition/write interface
memory/global concepts
hint projection mechanism
```

Task/modality-specific:

```text
raw input probes
input adapter
output head
head/task probes
loss function
some output priors
```

This is the right split.

---

## 11. MVP implementation

### Stage A: input structure hints for audio

Add:

```text
AudioStructureProbeBank
StructureTokenEncoder
StructureToPriorProjector
```

Use audio probes:

```text
energy
local_energy
diff_energy
smoothness
frequency/mel band summaries
noise score
compressibility/low-rank proxy
```

Inject weak hints into:

```text
primitive logits
operator variant logits
read/write logits
role logits
```

### Stage B: head/task hints for classification

Add:

```text
ClassificationHeadTaskProbe
HeadStructureEncoder
HeadToPriorProjector
```

Use:

```text
num_classes
class query stats
class attention entropy
class confusion if available lagged from previous epoch
class slot diversity need
```

Inject weak hints into:

```text
class query/read logits
role logits
final-step read/write logits
macro hints later
```

### Stage C: joint structure fusion

```text
joint = fuse(input_structure_embedding, head_structure_embedding)
joint -> prior hints
```

### Stage D: A/B tests

Run:

```text
A: no structure hints
B: input hints only
C: head hints only
D: input + head hints
E: input + head + self-organizing usage prior
```

Metrics:

```text
best_acc
class_read_div
slot_div
role_usage_max
role_entropy
role_similarity
layer_sim
step_sim
memory/global usage
primitive/operator usage
```

---

## 12. Success criteria

Good sign:

```text
accuracy improves or reaches same with faster convergence
role_entropy decreases from near-uniform but does not collapse
role_similarity decreases
class_read_div decreases
slot_div decreases
memory/global remain nonzero
operator use becomes more semantically matched to input structure
```

Bad sign:

```text
hints dominate and recreate fixed phases
all tasks use same roles
role collapse to one role
class-head depends only on class_slot_prior
accuracy improves only with poor generalization
```

---

## 13. Short summary

Yes, we can do this for both input and head.

Input side:

```text
what structure exists in the data?
```

Head side:

```text
what structure does the task/output need?
```

Middle core:

```text
how should a matrix program be assembled to connect them?
```

This gives weak useful priors without trapping the model inside our hand-written layer roles.
