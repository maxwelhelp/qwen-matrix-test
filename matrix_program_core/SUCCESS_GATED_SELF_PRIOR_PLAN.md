# Success-Gated Self Prior Plan

This plan adds a safer alternative to naive usage priors.

Instead of reinforcing whatever a layer/block selected often, we reinforce only selections and editor/attention replacements that were measured as useful.

Core idea:

```text
Do not promote frequent components.
Promote successful components.
```

A component becomes more likely only if it belongs to a block/edit that improved train/heldout metrics or passed a counterfactual/validation gate.

---

## 1. Why naive usage prior is dangerous

Naive rule:

```text
if layer often chose component X:
    increase prior for X
```

Problem:

```text
random early choice -> reinforced -> alternatives suppressed -> rich-get-richer collapse
```

This recreates a hand-coded prior, only with random initialization as the author.

---

## 2. Safer rule: success-gated prior

New rule:

```text
if component X was used in a successful block/edit:
    increase weak lagged prior for X in similar contexts
else:
    do not reinforce it
```

Successful means at least one of:

1. block contributed to improved heldout/val window;
2. editor/attention replacement passed screen/verify;
3. counterfactual edit improved heldout above noise gate;
4. repeated local usage correlated with better class/head read and lower loss;
5. future critic predicts positive LCB-safe gain, after enough memory records.

---

## 3. What counts as a component

Track success for:

```text
latent role selection
primitive selection
OperatorBankV2 variant selection
primitive transition pair
slot transition
read target cell
write target cell
composition slot
macro selection
step alive
editor replacement action
```

Examples:

```text
role3 at L1.S2 helped
low_rank_r16 helped in role2
product_gate -> phase_matrix helped
read memory.M2 helped
macro7 replacement helped
step L2.S1 activation helped
```

---

## 4. What are successful blocks

A block/step can be marked as successful by a weak attribution signal first, then later by counterfactual verification.

MVP success sources:

1. **Window-level improvement**

```text
val_acc improved or val_loss decreased over last K epochs/windows
```

2. **Class/head usefulness**

```text
class attention increasingly reads this slot/block
class loss decreases for classes using it
```

3. **Gradient/usefulness proxy**

```text
positive pressure on selected components, but only as weak evidence
```

4. **Editor action result**

```text
editor changed component/action
after lagged window, heldout improved
```

5. **Counterfactual result later**

```text
explicit edit test improved heldout above noise threshold
```

Do not treat train-only gain as strong success.

---

## 5. How the prior is stored

Use lagged EMA buffers, detached from gradient:

```text
success_ema[group, location, item]
failure_ema[group, location, item]
uncertain_ema[group, location, item]
```

Where group can be:

```text
role
primitive
operator_variant
primitive_transition
read_cell
write_cell
macro
step_alive
```

Update rule:

```text
success_ema = decay * old + (1-decay) * successful_usage
failure_ema = decay * old + (1-decay) * failed_usage
```

Recommended:

```text
decay = 0.97 to 0.995
warmup_epochs = 3 to 5
```

No same-forward update. Use lag-by-one epoch/window.

---

## 6. How it affects logits

For each choice group:

```text
success_score = normalize(success_ema + eps)
failure_score = normalize(failure_ema + eps)
prior_bias = alpha * log(success_score + eps) - beta * log(failure_score + eps)
prior_bias = clamp(prior_bias, -max_bias, +max_bias)
logits += prior_bias.detach()
```

Start values:

```text
alpha = 0.03 to 0.10
beta = 0.02 to 0.07
max_bias = 0.10 to 0.25
```

This is not a trainable route. It is a lagged success memory bias.

---

## 7. Should attention/editor do this?

Attention/editor should not directly write permanent priors inside the same forward pass.

Correct division:

### Editor / attention

```text
proposes or applies soft replacements/edits
records what it changed
records attention/read/write/macro/role choices
```

### Success tracker

```text
observes whether the edited block/window improved
updates success/failure EMA after delay
```

### Success-gated prior module

```text
turns success/failure EMA into weak lagged bias
feeds bias back into assembly logits next epoch/window
```

### Later critic/actor

```text
uses tested memory and LCB/UCB rules for active edits
```

So attention/editor participates as the actuator and observer, but not as an unchecked self-reinforcement loop.

---

## 8. Why this helps latent roles

Latent roles need stabilization.

Without any self-prior:

```text
roles may stay uniform
or layers may switch roles randomly between batches
```

With success-gated prior:

```text
if role3 helped at L1.S2:
    role3 becomes slightly easier there later
if role3 stopped helping:
    prior decays or failure term suppresses it
```

This lets roles emerge from successful behavior, not human labels.

---

## 9. Relation to old fixed phase prior

Old fixed prior:

```text
L0 = extract
L1 = compare
L2 = suppress
L3 = aggregate
```

Success-gated prior:

```text
L1 used low_rank/product_gate and this helped
therefore L1 gets a weak bias toward that pattern
```

The result may become extract/compare-like, but it was discovered, not imposed.

---

## 10. What else to optimize with this logic

Use success-gated prior for:

1. **role choices**
   - stabilize useful anonymous roles.

2. **operator size/depth**
   - rank/depth grows only if useful.

3. **primitive chains**
   - promote useful transition pairs like `low_rank -> product_gate` only when useful.

4. **read/write addresses**
   - stabilize memory/global usage only if useful.

5. **macro use**
   - macro gets stronger only after successful contexts.

6. **step alive**
   - extra depth remains active only if useful.

7. **class-slot reads**
   - class attention stabilizes only when class loss improves.

8. **editor replacements**
   - if the editor repeatedly replaces A with B and B works, B gets local prior.

9. **future growth/promotion**
   - repeatedly successful priors can become local macro/operator promotions.

---

## 11. Safety rules

1. No update inside same forward pass.
2. Stop-gradient on EMA prior.
3. Warmup before prior starts.
4. Clamp prior bias.
5. Do not reinforce uncertain windows.
6. Penalize failure separately.
7. Keep entropy/revival mechanisms active.
8. Retest promoted patterns with counterfactual/heldout when available.
9. Never let success prior replace actor-critic safety for active edits.

---

## 12. MVP implementation

Create:

```text
matrix_program_core/success_gated_self_prior.py
```

Contains:

```text
SuccessGatedSelfPrior
  update_from_aux(aux, metrics, accepted_editor_events=None)
  make_biases(current_shapes)
  state_dict/load_state_dict
```

MVP groups:

```text
role if latent roles exist
primitive
primitive_transition
read_cell
write_cell
step_alive
```

Skip macro/operator variants until logging is reliable.

Runner flags:

```text
--use-success-gated-prior
--success-prior-alpha 0.05
--success-prior-beta 0.03
--success-prior-max-bias 0.15
--success-prior-decay 0.985
--success-prior-warmup-epochs 4
```

Report metrics:

```text
success_prior_mean_bias
success_prior_max_bias
success_prior_active_frac
success_prior_top_components
success_prior_failure_top_components
```

---

## 13. First experiment

Run on top of latent roles or phase-free OperatorBankV2:

```text
baseline: no success prior
variant: success-gated prior enabled after warmup
```

Success criteria:

```text
higher or equal best_acc
lower layer/step instability
role usage not collapsed
class_read_div decreases
slot_div decreases
no explosion of logit_norm
```

If performance improves but entropy collapses, alpha is too high.

---

## 14. Short summary

Yes: raise priors only for successful blocks and successful editor/attention replacements.

No: do not raise priors for whatever was frequently selected.

Editor/attention should propose and record edits. The success tracker should decide whether they were useful. The success-gated prior should apply weak lagged bias later. The actor-critic system later replaces the heuristic success signal with measured counterfactual/heldout gain.
