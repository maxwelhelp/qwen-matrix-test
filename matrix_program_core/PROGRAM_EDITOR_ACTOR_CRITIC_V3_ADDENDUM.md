# Editor Actor-Critic V3 Addendum: Candidate Generator, Workshop Budget, Grammar Mapping

This addendum patches the main gaps in `PROGRAM_EDITOR_ACTOR_CRITIC_POLICY_PLAN.md` before implementation.

It does not replace V3. It clarifies the pieces that must exist so V3 does not become unstable or vague.

---

## 1. Main fixes added by this addendum

The V3 plan is strong on safety:

```text
feedback_bias != actor_bias
lag-by-one feedback only
ridge critic cold start
full training state rollback
UCB for probe, LCB for deploy
```

But it needs four concrete additions:

1. **Candidate edit generator**: what edits are proposed before critic scoring.
2. **Workshop coordination/budget**: prevent 9 workshops from fighting each other.
3. **Grammar mapping**: every actor edit must map to read/transform/write grammar elements.
4. **Exploration budget**: critic must see zones not already preferred by the actor.

Also stage order should change:

```text
critic/actor refinement must come before growth/promotion
```

Growth without a reliable critic promotes noise.

---

## 2. Grammar-to-workshop mapping

Actor actions must be expressed through the grammar from `READ_TRANSFORM_WRITE_GRAMMAR_PLAN.md`.

Outer grammar remains:

```text
read -> transform -> write
```

`transform` internally contains:

```text
primitive | compose | compare | branch | loop | macro
```

### 2.1 Mapping table

| Grammar element | Workshop | Editable objects | Example sparse edit |
|---|---|---|---|
| read | read workshop | read source group, read cell, read entropy | increase memory read at L2.S1.B0.K2 |
| transform.primitive | primitive workshop | primitive slot weights | increase low_rank in L1.S2.B3.K1 |
| transform.operator_variant | operator-size/depth workshop | rank/depth/radius/variant | increase low_rank_r16, decrease r64 |
| transform.compare | compare workshop | compare type, source pair, score norm | add slot contrast between K1 and K3 |
| transform.compose | slot-transition/composition workshop | sum/gate/residual/diff/product compose | increase product-gate compose for B2 |
| transform.branch | branch/path workshop | path weights, path entropy, path dropout | route rare examples to path2 |
| transform.loop | step/layer recipe workshop | step_alive, depth, inner loop gate | reduce step_alive at L3.S2 |
| transform.macro | macro workshop | macro id/content embedding, macro scale | activate local macro M7 at L2.S0 |
| write | write workshop | target group, target cell, write gate | reduce global write, increase state write |
| memory | memory workshop | memory read/write/keep/forget | protect M1 from overwrite |
| budget | global budget workshop | complexity, sparsity, compute, trust region | allow only one extra branch path |
| head decision | head macro workshop | class read, pair contrast, logit delta | add yes/no contrast read |

The actor should never emit a free-form tensor without a grammar label.

Every edit record must include:

```text
grammar_element
workshop
target_location
action_type
signed_delta
scale
complexity_delta
expected_effect
```

---

## 3. Workshop coordination

V3 lists many workshops. Without coordination, they can conflict.

Example conflict:

```text
read workshop: increase memory read
write workshop: reduce memory write
macro workshop: activate macro that expects memory write
budget workshop: penalize memory usage
```

This can produce unstable or contradictory actor edits.

### 3.1 Workshop coordinator

Add a `WorkshopCoordinator` that receives candidate edits from all workshops and selects a small compatible set.

Inputs:

```text
candidate edits
critic score / uncertainty / risk
complexity cost
conflict matrix
workshop quota
recent accepted/rejected history
```

Output:

```text
0 to K sparse edits per window
```

### 3.2 Hard limits

Start with:

```text
max_workshops_per_window = 2
max_edits_per_window = 1 initially, then 2
max_edit_scale = 0.03 initially
max_total_actor_bias_norm = small fixed cap
```

For MVP Stage 3:

```text
one sparse edit per window
one workshop per edit
no compound workshop edits
```

Only after stable memory exists:

```text
allow two-workshop compound edits if individual A and B were tested separately
```

### 3.3 Conflict rules

Do not deploy together in one window:

```text
increase and decrease same softmax group
increase read to source while simultaneously suppressing all writes needed by that source
activate macro and suppress its required primitive/operator
increase complexity while budget says current run is over budget
increase branch path and reduce step_alive for the only step using it
```

For compound edits, store sub-edits:

```text
A only
B only
A+B
```

when possible.

---

## 4. Candidate edit generator

This is the missing core of Stage 3.

The critic does not invent edits from nothing. It scores candidates. We need deterministic and trace-driven candidate generators.

### 4.1 Candidate sources

Generate candidates from five sources:

1. **Gradient/saliency source**
   ```text
   grad * weight, grad wrt logits, pressure on read/primitive/write groups
   ```

2. **Collapse/entropy source**
   ```text
   too-uniform or too-collapsed softmax groups
   high layer_sim / step_sim
   class_read overlap
   role_similarity high
   ```

3. **Residual/error source**
   ```text
   per-class errors
   confusion pairs
   high loss examples
   residual reconstruction gaps
   memory recall failures
   ```

4. **Memory retrieval source**
   ```text
   similar past accepted/rejected edits
   local task/run memory
   stale accepted edit retests
   ```

5. **Random/exploration source**
   ```text
   low-cost candidate edits in underexplored grammar zones
   revived suppressed components
   random role/operator perturbations within safe bounds
   ```

### 4.2 Candidate families

Generate sparse candidate edits for these families.

#### Read candidates

```text
increase/decrease source group read: evidence/state/memory/global/output
shift read from overused cell to underused useful cell
revive suppressed source if gradient/counterfactual suggests possible gain
```

#### Transform primitive candidates

```text
increase/decrease primitive slot weight
revive suppressed primitive
increase compare primitive for high confusion/memory retrieval errors
increase gate/filter when noise/background errors rise
```

#### Operator variant candidates

```text
change low_rank size
change butterfly depth
change local radius / smooth/diff/wavelet variant
increase cheaper variant if accuracy same but complexity lower
```

#### Compare candidates

```text
add slot contrast
add memory query match
add class-pair compare
add local-vs-global compare
```

#### Compose candidates

```text
increase residual compose
increase product/gate compose
increase difference/contrast compose
reduce dense sum if too uniform
```

#### Write candidates

```text
increase state write for useful update
reduce global/memory overwrite risk
protect memory slot
shift write target from overused to underused cell
```

#### Step/layer candidates

```text
increase/decrease step_alive
open/close late input read gate
reduce shortcut input-read if specialization weak
allow input reread if confidence low or critic predicts gain
```

#### Macro candidates

```text
activate existing local macro
reduce duplicate macro
test small macro variation
```

#### Head candidates

```text
class_pair_contrast
class_unique_slot_read
suppress_shared_slot
memory_global_gate
confidence_calibration
```

### 4.3 Candidate object schema

Every generated candidate must be explicit:

```json
{
  "id": "run_epoch_window_candidate_id",
  "source": "gradient|collapse|residual|memory|explore",
  "grammar_element": "read|transform.primitive|transform.compare|transform.compose|write|memory|head",
  "workshop": "read|primitive|compare|write|memory|head|budget",
  "location": {"L": 1, "S": 2, "B": 0, "K": 3, "A": null, "P": "low_rank"},
  "action": "increase|decrease|shift|revive|protect|activate_macro",
  "target": "low_rank_r16 or memory.M2 or class_pair.yes_no",
  "delta_scale": 0.03,
  "expected_effect": "reduce yes/no confusion by increasing slot contrast",
  "complexity_delta": 0.004,
  "risk_flags": ["same_softmax_group", "memory_overwrite"],
  "requires": ["primitive.low_rank active"],
  "conflicts_with": ["reduce low_rank same location"]
}
```

---

## 5. Candidate ranking before critic

Before critic scoring, apply cheap filters:

```text
valid grammar target
inside current complexity budget
not exact duplicate of already tested recent edit
not conflicting with mandatory constraints
not dense / not multi-workshop in MVP
scale within trust region
```

Then compute a bootstrap score for cold start:

```text
bootstrap_score =
    gradient_pressure
  + collapse_repair_score
  + residual_error_score
  + memory_similarity_score
  + exploration_bonus
  - complexity_cost
  - duplicate_penalty
  - risk_penalty
```

When critic has records:

```text
probe_score = blend(bootstrap_score, critic_UCB_score)
deploy_score = blend(bootstrap_score, critic_LCB_score)
```

---

## 6. Exploration budget

Closed-loop re-verify on actor-visited regions is necessary, but not sufficient.

Risk:

```text
actor repeatedly visits the same biased region
critic gets trained mostly on actor's mistakes
critic becomes accurate only in that narrow region
```

Add explicit exploration budget.

### 6.1 Probe candidate mixture

For Tier-1 screening candidate pool:

```text
50% exploitation: high bootstrap/critic score
25% uncertainty: high critic uncertainty
15% revival: suppressed low-probability components
10% random safe: low-cost grammar-valid edits
```

Tune later, but do not make exploration zero.

### 6.2 Deploy remains conservative

Exploration affects what is tested, not what is live-deployed.

```text
probe: UCB / exploration allowed
deploy: LCB / safety only
```

### 6.3 Underexplored grammar tracking

Track coverage by:

```text
grammar_element
workshop
operator family
read/write source group
layer/step bucket
head macro family
```

Boost testing for low-coverage zones if compute budget allows.

---

## 7. Stage order correction

Original V3 order placed local growth before critic actor refinement.

Change order:

```text
Stage 1: reliable observation
Stage 2: anti-collapse + bi-level
Stage 3: counterfactual screen + memory + cold-start critic MVP
Stage 4: feedback bus MVP
Stage 5: critic/actor refinement + exploration + closed-loop reverify
Stage 6: local growth/promotion
Stage 7: neural typed-token actor
Stage 8: global growth/promotion
```

Reason:

```text
local growth without reliable critic and tested memory risks promoting noise
```

Local growth can still store candidate macro sketches earlier, but promotion should wait until Stage 5 is stable.

---

## 8. Updated minimal first code target

First active-mind code target should include:

```text
CandidateEditGenerator
WorkshopCoordinator
CounterfactualScreen
ProgramMemoryBank
ColdStartCritic
SparseLCBEditApplier
FullTrainingStateRollback
```

Safe MVP loop:

```text
once per epoch/window:
  observe current program summaries
  generate sparse candidate edits from 5 candidate sources
  filter by grammar/budget/conflicts
  select Tier-1 screen pool with exploration budget
  screen on heldout micro-batches
  update JSONL memory with accepted/rejected/uncertain
  train/update ridge critic
  choose at most one sparse deploy edit by LCB
  apply bias with scale 0.03
  verify / rollback / store result
```

No neural feedback bus yet.

No local macro promotion yet, only candidate sketches.

---

## 9. Metrics to add

Report per window:

```text
num_candidates_generated by source
num_candidates_after_filter
num_screened
num_accepted/rejected/uncertain
candidate coverage by grammar element
candidate coverage by workshop
exploration/exploitation ratio
actor deployed edit type
workshops touched per deployed edit
conflict rejects
critic predicted vs real gain
full rollback count
compute overhead
```

---

## 10. Short summary

V3 remains correct, but implementation needs this addendum.

Do not let the critic/actor act on vague tensors.

Every edit must be:

```text
grammar-labeled
workshop-labeled
sparse
budgeted
conflict-checked
screened or verified
stored with measured result
```

Candidate generation is the first concrete intelligence layer:

```text
gradient + collapse + residual + memory + exploration
  -> sparse grammar-valid candidate edits
  -> critic scores
  -> heldout tests
  -> safe deploy or reject
```

Growth comes after this loop is reliable, not before.
