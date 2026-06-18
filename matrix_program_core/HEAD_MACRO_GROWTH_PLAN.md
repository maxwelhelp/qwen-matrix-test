# Head Macro Growth Plan

Goal: extend the same growth idea used for matrix/program operators to the output head.

Core growth answers:

```text
How should the model transform data?
```

Head growth answers:

```text
How should the model read the produced slots/cells and turn them into the task answer?
```

This plan is compatible with:

- `INPUT_HEAD_STRUCTURE_PRIOR_PLAN.md`
- `LATENT_ROLE_SPONTANEOUS_SPECIALIZATION_PLAN.md`
- `PROGRAM_EDITOR_ACTOR_CRITIC_POLICY_PLAN.md`
- the older `symbolic_operator_lab_v5_joint_programs` principle:

```text
primitive_A + primitive_B + condition_between = new useful operator
```

For the head, the equivalent is:

```text
read_A + read_B + task/error condition = new head macro
```

---

## 1. Why head growth is needed

The current matrix program core can learn useful slots and cells. But the final answer depends on the head reading those slots correctly.

Observed pattern from recent runs:

```text
class_read_div falls strongly
slot_div falls
layer/step outputs specialize
latent roles still weak
```

This suggests that a major part of performance is coming from the output head learning to read different slots for different classes.

Therefore the head should not be a fixed simple reader forever. It should be able to grow safe, tested read/decision macros.

---

## 2. What is a head macro?

A head macro is a reusable differentiable decision/read recipe.

Examples:

```text
class_pair_contrast
class_unique_slot_read
suppress_shared_slot
memory_global_gate
local_global_product_read
background_rejector
confidence_calibrator
prototype_matcher
rare_class_recall
```

It is not arbitrary Python code. It is a soft matrix/read program applied at the head.

A head macro can modify:

```text
class query vectors
class read logits
slot/cell read mixture
memory/global read mixture
pairwise class contrast logits
logit calibration deltas
confidence/gate values
```

---

## 3. Head growth principle

Core operator growth:

```text
primitive_A + primitive_B + transition/condition = new operator/macro-step
```

Head macro growth:

```text
query_A + read_pattern_B + task/error_condition = new head macro
```

Examples:

```text
class_query + slot_read + contrast_gate
  -> class_separator_macro

memory_read + global_read + suppress_gate
  -> background_rejector_macro

local_slot_read + global_summary_read + product_gate
  -> command_detector_macro

class_A_query - class_B_query + margin_gate
  -> confusion_resolver_macro
```

The condition comes from real head/task signals:

```text
confusion matrix
per-class loss
low margin
high class entropy
class read overlap
wrong confidence
head gradient pressure
heldout gain from candidate macro
```

---

## 4. What head macros may read

The head can read from the same universal program interface:

```text
program slots
state cells
memory cells
global cells
latent role summaries
macro usage summaries
operator usage summaries
input structure tokens
head/task structure tokens
```

Do not let the head bypass the core by directly memorizing raw input unless explicitly testing a baseline.

---

## 5. Candidate macro families

### 5.1 Class pair contrast

Triggered by:

```text
confusion(A,B) high
margin(A,B) low
class_read(A) and class_read(B) too similar
```

Recipe:

```text
read slots used by A
read slots used by B
compute soft contrast
add bounded logit delta for A/B
```

### 5.2 Class unique slot read

Triggered by:

```text
class C has high loss
class C read distribution too broad or too similar to others
```

Recipe:

```text
find slots with positive gradient for C and low use by other classes
increase C read logits to those slots
```

### 5.3 Suppress shared slot

Triggered by:

```text
many classes read the same slot
slot contributes to confusion
```

Recipe:

```text
reduce shared slot read for confused classes
or route it through a common/background channel
```

### 5.4 Memory/global gate

Triggered by:

```text
local slots insufficient
memory/global usage predicts correct class
```

Recipe:

```text
read memory/global
product gate with class query
bounded logit delta
```

### 5.5 Local/global product read

Triggered by:

```text
class needs both local evidence and global context
```

Recipe:

```text
read local slot group
read global summary
product/gate them
write class-specific evidence score
```

### 5.6 Background rejector

Triggered by:

```text
false positives on noise/background-like examples
low confidence but wrong high logit
```

Recipe:

```text
read background/common slots
suppress logits if common slot dominates class-unique slot
```

### 5.7 Confidence calibrator

Triggered by:

```text
overconfident wrong predictions
class-specific calibration error
```

Recipe:

```text
bounded temperature/logit-scale delta per class or class group
```

### 5.8 Rare class recall

Triggered by:

```text
rare class recall low
class has few but strong unique slots
```

Recipe:

```text
prototype/memory read for rare class
boost class if prototype match high
```

---

## 6. Growth pipeline

### Step 1: observe head failures

Collect lagged statistics:

```text
confusion matrix
per-class loss
per-class accuracy/recall
class margin distribution
class read entropy
class read overlap matrix
slot contribution estimates
head gradient norms
heldout val metrics
```

### Step 2: generate sparse candidates

Generate only a small top-k set:

```text
candidate family
class or class-pair target
read source slots/cells
macro parameters
complexity cost
```

Candidate format:

```json
{
  "macro_type": "class_pair_contrast",
  "target": ["yes", "no"],
  "read_slots": [3, 7, 11],
  "condition": "high_confusion_low_margin",
  "delta_scale": 0.05,
  "complexity": 0.012
}
```

### Step 3: Tier-1 screen

Forward-only heldout micro-batch:

```text
apply temporary macro bias
measure heldout loss / class-pair margin / total val acc
revert
```

### Step 4: Tier-2 verify

Only for candidates that pass Tier-1:

```text
save full training state
apply macro
run short adaptation or heldout window
measure gain
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

### Step 5: memory record

Store tested result:

```text
accepted / rejected / uncertain
heldout gain
noise std
z-score
class-pair gain
global val impact
complexity cost
context embedding
macro content embedding
```

### Step 6: promote macro

Promotion requires:

```text
heldout gain passes dynamic noise gate
not duplicate of existing macro
complexity budget acceptable
retested positive
```

Promotion tiers:

```text
local promotion: useful inside one run across several classes/locations
global promotion: useful across runs/tasks
```

---

## 7. Avoiding combination explosion

Head candidates can explode:

```text
classes x class_pairs x slots x read_patterns x gates x macro types
```

Use strict filters.

### 7.1 Candidate budget

```text
max_candidate_pairs per epoch/window
max_slots_per_candidate
max_macros_tested_per_window
```

### 7.2 Complexity cost

Penalize:

```text
number of slots read
number of class pairs affected
extra gates
extra memory/global reads
logit delta size
```

### 7.3 Similarity / duplicate penalty

Do not add macros that are equivalent to existing ones.

Compare macro content embeddings:

```text
read pattern
class target pattern
gate/operator type
slot/cell source pattern
effect vector on logits
```

Reject if similarity above threshold unless heldout gain is significantly higher.

### 7.4 One-factor tests first

Early memory should test one or two factors:

```text
only read change
only contrast gate
read + gate after both tested separately
```

This keeps credit assignment clean.

### 7.5 No fake accepted records

Never mark a macro accepted because train gradient looked good.

Accepted requires real Tier-1 repeated or Tier-2 heldout measurement.

---

## 8. Relationship to input/head structure priors

`INPUT_HEAD_STRUCTURE_PRIOR_PLAN.md` gives weak hints:

```text
input structure: what the data contains
head/task structure: what the answer needs
```

Head macro growth uses failures to create new reusable recipes:

```text
head/task hints guide where to look
head macro growth tests what actually helps
```

Example:

```text
input hint: local/diff/onset high
head hint: class separation needed
failure: yes/no confusion high
candidate: class_pair_contrast using local onset slots
heldout improves -> promote yes/no contrast macro
```

---

## 9. Relationship to latent roles

Head macros can give feedback to latent roles later.

Example:

```text
head macro repeatedly reads slots produced by role3
critic sees positive gain
feedback bus increases role3 usefulness in similar contexts
```

But do not implement same-pass future feedback.

Use lagged feedback:

```text
forward t: head uses slots, record trace
forward t+1/window: role/core gets summarized feedback
```

---

## 10. Relationship to actor-critic

Head macro growth should later use the same safe actor-critic principles:

```text
UCB for probe/testing
LCB for live deploy
sparse actor output
hard rollback
noise gate
critic re-verify
memory records only from tested edits
```

The head macro critic predicts:

```text
critic_head(context_embedding, macro_embedding)
  -> predicted heldout gain
  -> uncertainty
  -> risk
```

Deploy score:

```text
mean_gain - beta * uncertainty - risk - complexity
```

Probe score:

```text
mean_gain + beta * uncertainty - risk - complexity
```

---

## 11. Head macro content embedding

Do not use only integer macro ids.

Use content-based embedding:

```text
macro_type
class target or class-pair target
read source pattern
slot/cell/memory/global pattern
gate/operator family
expected effect vector
complexity features
```

This lets new promoted macros be compared without retraining an ID table.

---

## 12. MVP for SpeechCommands

Start with classification head only.

### Metrics to collect

```text
confusion matrix
per-class accuracy
per-class loss
class margin
class read entropy
class read overlap
slot contribution proxy
head gradient norms
```

### Candidate types

MVP only:

```text
class_pair_contrast
class_unique_slot_read
suppress_shared_slot
memory_global_gate
```

### Safety

```text
max 1 accepted macro per epoch/window
macro bias scale starts 0.03
clamp logit delta
rollback if global val worsens > noise gate
```

### Report

Every macro record should print:

```text
macro type
target class/class-pair
slots/cells read
heldout gain
class-pair gain
global val impact
accepted/rejected/uncertain
similar existing macro
complexity cost
```

---

## 13. Success criteria

Good signs:

```text
faster convergence
higher best_acc
class confusion pairs improve
class_read_div improves without relying only on class_slot_prior
head macros are sparse and interpretable
macros are reused, not one-off noise
no global val damage
```

Bad signs:

```text
many macros but no heldout gain
macros duplicate each other
head overfits train confusion
global val drops while target pair improves
macro effects are dense and uninterpretable
critic predicts gains that do not verify
```

---

## 14. What not to do

Do not:

```text
add all possible class-pair macros at once
accept macros based only on train gradient
let macros bypass the core and memorize raw input
create dense head bias everywhere
use UCB for live deployment
rollback only model weights after Tier-2
promote duplicate macros
```

---

## 15. Short summary

Core growth builds better transformations.

Head growth builds better ways to read and decide.

For the head:

```text
read_A + read_B + task/error condition = new head macro
```

The error condition comes from confusion, margins, per-class loss, read overlap, and heldout tests.

The macro is promoted only if it passes noise-gated heldout verification and is not a duplicate.

This turns the head from a fixed reader into a safe, growing decision-program bank while keeping the whole system matrix-based, differentiable, sparse, and testable.
