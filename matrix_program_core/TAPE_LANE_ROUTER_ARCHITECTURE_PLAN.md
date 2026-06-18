# Tape Lane Router Architecture Plan

Goal: create the next simple MatrixProgram version from the `simple_butterfly_matrix_v3` idea, but replace fixed semantic layers with one growing program tape and differentiable matrix separators / lane routers.

Reference repo/path:

```text
https://github.com/maxwelhelp/test2/tree/main/simple_butterfly_matrix_v3
```

Important: this plan is an implementation assignment for the agent. It should be read together with:

```text
READ_TRANSFORM_WRITE_GRAMMAR_PLAN.md
LATENT_ROLE_SPONTANEOUS_SPECIALIZATION_PLAN.md
INPUT_HEAD_STRUCTURE_PRIOR_PLAN.md
HEAD_MACRO_GROWTH_PLAN.md
AGENT_IMPLEMENTATION_MASTER_PLAN.md
```

---

## 1. What to take from `simple_butterfly_matrix_v3`

`simple_butterfly_matrix_v3` is useful because it is simple, matrix-based, and avoids hard routers/top-k.

The README says v3 replaced the passive task head with a matrix class-state head:

```text
slots
  -> phase read matrix
  -> ClassMatrix[C,D]
  -> ClassPairMatrix[P,D]
  -> class update matrix
  -> logits
```

This is the best part to keep conceptually:

```text
classes are trainable matrix states;
class-pair repair exists;
class reads are soft differentiable matrices;
no hard router;
no top-k.
```

The code implements this through:

```text
class_state[C,D]
class_phase_logits[C,phase]
pair_state[pair_slots,D]
class_pair_logits[C,pair]
class read attention over slots
pair repair context
class update matrix
```

Keep this idea, but remove dependency on named phase slots in the new tape version.

---

## 2. What must be removed / changed from v3

The current v3/v2 backbone still has hard phase structure.

Examples from `simple_butterfly_matrix_v2/soft_matrix_transport.py`:

```text
PHASES = extract / compare / suppress / aggregate
phase_names = [PHASES[min(i, len(PHASES)-1)] for i in range(layers)]
SoftMatrixTransportStep(..., phase_names[l], ...)
_primitive_prior(phase)
if phase == extract/compare/suppress/aggregate in _primitive_outputs
slot names include L{l}.{phase}
```

Examples from `simple_butterfly_matrix_v3/class_matrix_transport.py`:

```text
PHASE_NAMES = input/extract/compare/suppress/aggregate
phase_slot_matrix(layers, steps, variants, blocks)
class_phase_logits[C,phase]
phase_balance_loss
phase_prior_strength
```

These were useful to fix collapse, but they still encode human-defined layer roles.

New architecture must not say:

```text
L0 = extract
L1 = compare
L2 = suppress
L3 = aggregate
```

New architecture must say:

```text
we have one tape of T steps;
steps route information between lanes;
soft boundaries discover segments;
posthoc reports may name segments after training.
```

---

## 3. New high-level architecture

Use:

```text
Input Adapter -> TapeLaneRouterBackbone -> ClassMatrixLaneHead
```

More explicitly:

```text
raw input/audio
  -> MatrixEvidence / input adapter
  -> lane-initialized cells X[N, lanes, cells_per_lane, D]
  -> tape steps t=0..T-1
       read -> transform -> write
       lane route matrix R_t[lane_from,lane_to]
       boundary gate b_t
       step_alive gate a_t
  -> all intermediate tape/lane slots
  -> ClassMatrixLaneHead
  -> logits
```

The middle is one long matrix program tape, not fixed layers.

---

## 4. Lanes

MVP lanes:

```text
lane0 = detail/raw/local
lane1 = working state
lane2 = abstract/global
lane3 = memory/control
```

These names are weak initialization/report labels only.

Do not hard-bind operators forever:

```text
bad: lane0 must always use diff
bad: lane2 must always use global
```

Allowed:

```text
lane embeddings give weak operator/read/write biases;
training can override them.
```

State shape:

```python
X: [N, LANES, CELLS_PER_LANE, D]
```

Flattened slots for head/report:

```python
slots = X_all_steps.reshape(N, total_slots, D)
```

---

## 5. Matrix lane routers / separators

Main mechanism:

```text
R_t[lane_from, lane_to]
```

Every tape step has a soft route matrix.

Example behavior:

```text
state -> state:    0.60
state -> abstract: 0.20
state -> memory:   0.10
detail -> state:   0.10
```

Meaning:

```text
some information stays in working state;
some moves upward to abstract/global;
some is written to memory/control;
some detail feeds the state lane.
```

MVP parameterization:

```python
self.route_logits = nn.Parameter(torch.zeros(T, LANES, LANES))
route = torch.softmax(self.route_logits[t], dim=-1)  # from -> to
```

Context-conditioned later:

```python
route_logits_t = self.route_logits[t] + route_net(step_summary)
```

Routed update:

```python
# update: [N, from_lane, A, D]
# route:  [from_lane, to_lane]
routed = torch.einsum('ij,niad->njad', route, update)
X = norm(X + step_alive[t] * write_gate[t] * routed)
```

Test this carefully. Shape mistakes here are dangerous.

---

## 6. Learned boundaries

Boundary gate:

```python
boundary_t = sigmoid(boundary_logit[t])
```

Boundary does not hard-reset. It softly marks a segment transition.

MVP uses boundary for:

```text
logging;
weak bias to route more state->abstract/memory;
weak bias to close old segment summary;
optional segment summary update.
```

Do not use destructive reset in MVP.

Later:

```python
segment_summary = (1 - boundary_t) * segment_summary + boundary_t * new_summary
```

Report discovered segments by high boundary mass.

---

## 7. Tape step grammar

Each tape step keeps the same outer grammar:

```text
read -> transform -> write
```

### 7.1 Read

Read is lane-aware.

Source groups:

```text
evidence/input
lane0 detail
lane1 state
lane2 abstract
lane3 memory/control
```

Read logits:

```text
read_group_logits[t, lane, source_group]
read_cell_logits[t, lane, block/slot, cell]
```

Early curriculum:

```text
step0-1 can read evidence cheaply;
later steps pay input_read_depth_cost;
input reread unlocks later.
```

This avoids all steps shortcutting raw input and becoming identical.

### 7.2 Transform

Transform remains the compute container:

```text
transform = primitive | compose | compare | branch | loop | macro
```

For the first version, reuse v2/v3 primitive ideas:

```text
channel_butterfly
block_butterfly
low_rank
ctx_matrix
product_gate
phase-like learned generic transform, but not phase-named
```

Important change:

```text
remove if phase == extract/compare/suppress/aggregate from primitive outputs.
```

Replace `phase_matrix` with neutral/generic transform candidates:

```text
generic_mix
compare_transform
suppress/gate_transform as learned operator family, not layer phase
```

Add first-class compare later or in MVP if simple:

```text
diff: a-b
product: a*b
dot/cosine
bilinear q^T W k
```

### 7.3 Write

Write is conservative.

Use:

```text
write_gate[t,lane]
residual update
LayerNorm
memory overwrite cost
global/abstract write cost
```

Write targets are lanes through `R_t`, not hard layer outputs.

---

## 8. Length growth without dynamic shape changes

Do not dynamically insert tensors during training.

Use max capacity:

```text
T_max = 12 or 16
```

Each step has:

```python
step_alive[t] = sigmoid(step_alive_logit[t])
```

Update is multiplied by `step_alive[t]`.

This gives differentiable length selection.

### Dormant spare steps

Initialize some later steps lower alive:

```text
active-ish: t0..t7
spare/dormant: t8..t15
```

Growth means:

```text
a dormant step's alive gate rises;
its read/transform/write and route matrix become useful;
boundary around it increases;
reports show a new segment.
```

Outer-loop later can make next run larger if many spare steps are used:

```text
T=12 -> T=16
```

But MVP keeps fixed shape.

---

## 9. Head: keep ClassMatrix idea, replace phases with lanes/segments

Current v3 head reads phases through `class_phase_logits` and `phase_slot_matrix`.

New head should read lane/segment groups instead.

Replace:

```text
class_phase_logits[C, phase]
phase_slot_matrix[phase, slot]
```

With one or both:

```text
class_lane_logits[C, lane]
class_segment_logits[C, segment_bucket]
```

MVP simpler:

```text
class_lane_logits[C, LANES]
lane_slot_matrix[LANES, slot]
```

Later:

```text
class_boundary_segment_logits[C, soft_segment]
```

Keep:

```text
class_state[C,D]
pair_state[P,D]
class_pair_logits[C,P]
class update MLP/matrix
pair repair attention
class_write gate
class_read_diversity_loss
slot_diversity_loss
```

Remove or rename:

```text
phase_balance -> lane_balance / segment_balance
phase_prior_strength -> lane_prior_strength or group_prior_strength
class_phase_mass -> class_lane_mass / class_segment_mass
```

New head formula:

```python
lane_w = softmax(class_lane_logits, dim=-1)      # [C,L]
slot_prior = lane_w @ lane_slot_matrix           # [C,S]
score = class_query @ slot_keys + group_prior_strength * log(slot_prior)
attn = softmax(score)
class_read = attn @ slot_values
pair repair as in v3
logits as in v3
```

This keeps the strong v3 head logic but removes fixed extract/compare/suppress/aggregate phases.

---

## 10. What to implement first

Create new folder or file, do not destroy v3:

```text
simple_butterfly_matrix_v4_tape_lane/
```

or in current repo:

```text
matrix_program_core/tape_lane_router_core.py
matrix_program_core/tape_lane_transport_audio.py
agent_scripts/run_tape_lane_router_audio.sh
```

Recommended if working inside `test2` first:

```text
simple_butterfly_matrix_v4_tape_lane/tape_lane_transport.py
simple_butterfly_matrix_v4_tape_lane/commands/run_smoke.sh
simple_butterfly_matrix_v4_tape_lane/commands/run_speechcommands.sh
```

MVP order:

1. Copy v3 training shell/loop/report style.
2. Reuse `MatrixEvidence`, loaders, losses utilities from `simple_butterfly_matrix`.
3. Implement `TapeLaneRouterBackbone`.
4. Implement `ClassMatrixLaneHead` by adapting `ClassMatrixHead`.
5. Add logs for lane routes, boundaries, step_alive, lane reads/writes.
6. Smoke synthetic.
7. SpeechCommands 8 epochs.
8. SpeechCommands 20 epochs.

---

## 11. Key pseudo-code

### 11.1 Backbone skeleton

```python
class TapeLaneRouterBackbone(nn.Module):
    def __init__(self, dim, evidence_cells, lanes=4, cells_per_lane=12, steps=12,
                 variants=3, channel_stages=3, dropout=0.04):
        super().__init__()
        self.D = dim
        self.L = lanes
        self.A = cells_per_lane
        self.T = steps
        self.V = variants
        self.evidence = MatrixEvidence(...)

        self.lane_embed = nn.Parameter(torch.randn(lanes, dim) * 0.02)
        self.step_embed = nn.Parameter(torch.randn(steps, dim) * 0.02)
        self.init_query = nn.Parameter(torch.randn(lanes, cells_per_lane, dim) * 0.04)

        self.step_alive_logit = nn.Parameter(torch.linspace(0.5, -0.5, steps))
        self.boundary_logit = nn.Parameter(torch.zeros(steps))
        self.route_logits = nn.Parameter(torch.zeros(steps, lanes, lanes))

        self.read_group_logits = nn.Parameter(torch.zeros(steps, lanes, lanes + 1))
        self.write_gate_logit = nn.Parameter(torch.full((steps, lanes), -0.35))

        self.units = nn.ModuleList([
            TapeLaneTransformUnit(dim, lanes, cells_per_lane, variants, channel_stages, dropout)
            for _ in range(steps)
        ])
        self.norm = nn.LayerNorm(dim)
```

### 11.2 Init lanes

```python
def init_lanes(self, evidence):
    # evidence: [N,E,D]
    q = self.init_query.reshape(self.L * self.A, self.D)
    score = torch.einsum('ad,ned->nae', q, evidence) / math.sqrt(self.D)
    attn = torch.softmax(score.float(), dim=-1).to(evidence.dtype)
    base = torch.einsum('nae,ned->nad', attn, evidence)
    X = base.reshape(evidence.shape[0], self.L, self.A, self.D)
    X = X + self.lane_embed.view(1, self.L, 1, self.D)
    return self.norm(X)
```

### 11.3 Forward tape

```python
def forward(self, wav):
    evidence = self.evidence(wav)
    X = self.init_lanes(evidence)
    slots = [X]
    route_list, boundary_list, alive_list = [], [], []
    read_group_list, write_gate_list, update_norm_list = [], [], []

    for t, unit in enumerate(self.units):
        alive = torch.sigmoid(self.step_alive_logit[t])
        boundary = torch.sigmoid(self.boundary_logit[t])
        route = torch.softmax(self.route_logits[t].float(), dim=-1).to(X.dtype)

        read_packet, read_info = self.read_step(X, evidence, t)
        update, info = unit(X, read_packet, self.lane_embed, self.step_embed[t])
        write_gate = torch.sigmoid(self.write_gate_logit[t]).to(X.dtype).view(1, self.L, 1, 1)

        routed = torch.einsum('ij,niad->njad', route, update)
        X = self.norm(X + alive.to(X.dtype) * write_gate * routed)

        slots.append(X)
        route_list.append(route.detach())
        boundary_list.append(boundary.detach())
        alive_list.append(alive.detach())
        read_group_list.append(read_info['group_mass'].detach())
        write_gate_list.append(write_gate.detach())
        update_norm_list.append(update.detach().float().norm(dim=-1).mean(dim=-1))

    flat_slots = torch.stack(slots, dim=1).reshape(wav.shape[0], -1, self.D)
    aux = TapeLaneAux(...)
    return flat_slots, aux
```

### 11.4 Transform unit

```python
class TapeLaneTransformUnit(nn.Module):
    def __init__(self, dim, lanes, cells_per_lane, variants, channel_stages, dropout):
        super().__init__()
        self.channel = ChannelButterfly(dim, channel_stages)
        self.block = BlockButterfly(cells_per_lane)
        self.low_a = nn.Parameter(torch.randn(dim, max(8, dim // 4)) * 0.04)
        self.low_b = nn.Parameter(torch.randn(max(8, dim // 4), dim) * 0.04)
        self.ctx_w = nn.Parameter(torch.randn(dim, dim) * 0.04)
        self.gate_h = nn.Parameter(torch.randn(dim, dim) * 0.02)
        self.gate_c = nn.Parameter(torch.randn(dim, dim) * 0.02)
        self.primitive_mix = nn.Parameter(torch.eye(6) * 0.20 + 0.01 * torch.randn(6, 6))
        self.lane_primitive_bias = nn.Linear(dim, 6, bias=False)
        self.drop = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(dim)

    def forward(self, X, read_packet, lane_embed, step_embed):
        # X/read_packet: [N,L,A,D]
        ctx = read_packet
        flat = X.reshape(-1, X.shape[-2], X.shape[-1])
        flat_ctx = ctx.reshape(-1, ctx.shape[-2], ctx.shape[-1])
        ctx_m = flat_ctx @ self.ctx_w.to(X.dtype)
        channel = self.channel(flat + ctx_m)
        block = self.block(flat)
        low = (flat @ self.low_a.to(X.dtype)) @ self.low_b.to(X.dtype)
        product = flat * torch.tanh(ctx_m)
        gate = torch.sigmoid(flat @ self.gate_h.to(X.dtype) + ctx_m @ self.gate_c.to(X.dtype))
        generic = gate * channel + (1.0 - gate) * low
        cands = torch.stack([channel, block, low, ctx_m, product, generic], dim=2)
        # Mix primitives softly; lane bias can be added here.
        ...
        return update, info
```

Do not include phase-specific `if phase == ...` branches.

---

## 12. Logging requirements

Every run should write `analysis_epoch_XXX.json` with:

```text
best_acc / best_epoch
step_alive[t]
boundary[t]
route_matrix[t][from][to]
route_entropy[t]
read_group_mass[t,lane,group]
write_gate[t,lane]
update_norm[t,lane]
lane_slot_mass in head
class_lane_mass
class_top_reads
pair_update_norm
slot_div
class_read_div
lane_balance
late_input_read_mass
```

Also create `REPORT_TO_CHATGPT.txt` with a readable Russian-style summary:

```text
which steps became active;
where boundaries appeared;
which lane routes dominate;
whether late input-read shortcut happened;
which lanes the classes read;
what class-pair repair did;
best accuracy and plateau signs.
```

---

## 13. Losses / regularizers

MVP losses:

```text
CE
write_budget
update_alive
slot_div
class_read_div
lane_balance
route_entropy_band
step_alive_budget
late_input_read_cost
memory_overwrite_cost
logit_norm
```

Replace old `phase_balance` with:

```text
lane_balance_loss
segment/boundary balance optional
```

Example lane balance:

```python
lane_usage = class_lane_mass.mean(dim=(0,1)) or read/write route mass
loss = entropy_floor + overuse_penalty
```

Late input-read cost:

```python
cost = sum_t depth_weight[t] * read_mass[t, :, input_group].mean()
```

where `depth_weight[t]` is small early and larger for later steps, then decays after unlock epoch.

---

## 14. Commands for agent to create

Smoke:

```bash
bash simple_butterfly_matrix_v4_tape_lane/commands/run_smoke.sh
```

SpeechCommands:

```bash
bash simple_butterfly_matrix_v4_tape_lane/commands/run_speechcommands.sh
```

Suggested smoke command body:

```bash
python simple_butterfly_matrix_v4_tape_lane/tape_lane_transport.py \
  --synthetic \
  --epochs 1 \
  --train-limit 512 \
  --val-limit 256 \
  --batch-size 64 \
  --eval-batch-size 128 \
  --device cuda \
  --amp bf16 \
  --dim 96 \
  --evidence-cells 48 \
  --lanes 4 \
  --cells-per-lane 12 \
  --tape-steps 12 \
  --variants 3 \
  --out-dir simple_butterfly_matrix_v4_tape_lane/runs/smoke
```

Suggested real command body:

```bash
python simple_butterfly_matrix_v4_tape_lane/tape_lane_transport.py \
  --data-root ../architecture_builder/data/speechcommands \
  --epochs 20 \
  --train-limit 12000 \
  --val-limit 2000 \
  --batch-size 128 \
  --eval-batch-size 256 \
  --device cuda \
  --amp bf16 \
  --dim 96 \
  --evidence-cells 48 \
  --lanes 4 \
  --cells-per-lane 12 \
  --tape-steps 12 \
  --variants 3 \
  --pair-slots 12 \
  --lr 5e-4 \
  --out-dir simple_butterfly_matrix_v4_tape_lane/runs/speechcommands_tape_lane
```

---

## 15. What counts as success

Minimum success:

```text
code runs;
no hard phase roles;
accuracy approaches v3/simple baseline;
route matrices are not all identity;
boundaries are not all 0 or all 1;
late input read is controlled;
class head reads multiple lanes, not only final aggregate-like slots.
```

Strong success:

```text
>= v3 accuracy;
clear learned segments appear;
lane routing becomes interpretable;
classes use lane/class-pair matrices differently;
spare steps/dormant steps become useful only when needed;
slot_div and class_read_div improve;
no phase_balance needed.
```

Research success:

```text
posthoc discovered program:
  early steps: detail->state local processing
  middle steps: state->abstract low_rank/compare
  late steps: memory/control and output-ready slots
without hardcoding extract/compare/suppress/aggregate.
```

---

## 16. What not to do

Do not:

```text
copy phase_names from v2/v3 into new backbone;
keep _primitive_prior(phase);
keep if phase == extract/compare/suppress/aggregate;
make hard boundary if/else;
make hard top-k router;
let all steps read input freely from epoch 0;
dynamically change tensor shape during MVP;
remove the strong v3 class matrix/pair idea;
accept growth/promotion before basic tape-lane model works.
```

---

## 17. Agent summary

Implement a new simple version based on `simple_butterfly_matrix_v3`, but change the backbone from:

```text
fixed layers with phase roles
```

to:

```text
one tape of T differentiable steps;
4 computation lanes;
route matrix R_t[from_lane,to_lane] at each step;
soft boundary gates;
step_alive gates;
read -> transform -> write grammar;
ClassMatrix head reading lane/segment groups instead of phase groups.
```

Keep the best parts of v3:

```text
soft matrices;
no hard router;
no top-k;
ClassMatrix state head;
ClassPair repair;
rich analysis reports.
```

Remove the core weakness:

```text
hard phase/layer role construction.
```
