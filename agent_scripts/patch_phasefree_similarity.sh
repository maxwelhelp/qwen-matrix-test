#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

python - <<'PY'
from pathlib import Path
p = Path("matrix_program_core/assembler_core.py")
txt = p.read_text(encoding="utf-8")
orig = txt

# Add cfg.phase_prior_strength if missing. This controls layer->phase priors
# inside the assembler core, not only external loss terms.
if "phase_prior_strength: float" not in txt:
    txt = txt.replace('''    step_alive_init: float = 1.65
''', '''    step_alive_init: float = 1.65
    phase_prior_strength: float = 0.0
''')
    txt = txt.replace('''    use_deltas: bool = True
''', '''    use_deltas: bool = True
    phase_prior_strength: float = 0.0
''')

old = '''        phase_logits = torch.full((len(PHASES),), -0.65)
        phase_logits[min(layer, len(PHASES) - 1)] = 1.25
        self.phase_mix_logits = nn.Parameter(phase_logits)
'''
new = '''        # Phase prior is optional. With phase_prior_strength=0, layers do not
        # start as extract/compare/suppress/aggregate; they must specialize from
        # data + losses. With strength>0, this becomes only a scaled soft init.
        phase_strength = float(getattr(cfg, "phase_prior_strength", 0.0))
        if phase_strength <= 0.0:
            phase_logits = torch.zeros((len(PHASES),))
        else:
            phase_logits = torch.full((len(PHASES),), -0.65 * phase_strength)
            phase_logits[min(layer, len(PHASES) - 1)] = 1.25 * phase_strength
        self.phase_mix_logits = nn.Parameter(phase_logits)
'''
if old in txt:
    txt = txt.replace(old, new)

# Neutralize primitive-slot phase prior when phase_prior_strength=0.
old = '''    def _primitive_slot_prior(self) -> torch.Tensor:
        x = torch.zeros(self.B, self.K, self.P)
        idx = {name: i for i, name in enumerate(PRIMITIVES)}
        phase_sets = {
            "extract": ["ctx_matrix", "channel_butterfly", "phase_matrix"],
            "compare": ["low_rank", "product_gate", "phase_matrix"],
            "suppress": ["block_butterfly", "product_gate", "phase_matrix"],
            "aggregate": ["block_butterfly", "channel_butterfly", "phase_matrix"],
        }
        names = phase_sets.get(self.phase, list(PRIMITIVES))
        for b in range(self.B):
            for k in range(self.K):
                for j, name in enumerate(names):
                    if name in idx:
                        x[b, k, idx[name]] += 0.65 / (1 + abs(k - j))
                # Keep every primitive alive, but softly phase-biased.
                x[b, k, :] += 0.03
        return x + 0.01 * torch.randn_like(x)
'''
new = '''    def _primitive_slot_prior(self) -> torch.Tensor:
        x = torch.zeros(self.B, self.K, self.P)
        phase_strength = float(getattr(self.cfg, "phase_prior_strength", 0.0))
        if phase_strength <= 0.0:
            # No role prior: all primitive families start equal. Tiny noise only
            # breaks exact symmetry; layer specialization must emerge from data.
            return x + 0.01 * torch.randn_like(x)
        idx = {name: i for i, name in enumerate(PRIMITIVES)}
        phase_sets = {
            "extract": ["ctx_matrix", "channel_butterfly", "phase_matrix"],
            "compare": ["low_rank", "product_gate", "phase_matrix"],
            "suppress": ["block_butterfly", "product_gate", "phase_matrix"],
            "aggregate": ["block_butterfly", "channel_butterfly", "phase_matrix"],
        }
        names = phase_sets.get(self.phase, list(PRIMITIVES))
        for b in range(self.B):
            for k in range(self.K):
                for j, name in enumerate(names):
                    if name in idx:
                        x[b, k, idx[name]] += phase_strength * 0.65 / (1 + abs(k - j))
                x[b, k, :] += 0.03
        return x + 0.01 * torch.randn_like(x)
'''
if old in txt:
    txt = txt.replace(old, new)

# Neutralize primitive transition phase prior when phase_prior_strength=0.
old = '''    def _primitive_transition_prior(self) -> torch.Tensor:
        x = torch.eye(self.P) * 0.35
        idx = {name: i for i, name in enumerate(PRIMITIVES)}
        def link(a: str, b: str, v: float) -> None:
            if a in idx and b in idx:
                x[idx[a], idx[b]] = v
        if self.phase == "extract":
            link("ctx_matrix", "channel_butterfly", 0.75)
            link("channel_butterfly", "phase_matrix", 0.60)
        elif self.phase == "compare":
            link("low_rank", "product_gate", 0.75)
            link("ctx_matrix", "phase_matrix", 0.55)
        elif self.phase == "suppress":
            link("block_butterfly", "phase_matrix", 0.80)
            link("product_gate", "phase_matrix", 0.60)
        elif self.phase == "aggregate":
            link("phase_matrix", "block_butterfly", 0.75)
            link("block_butterfly", "channel_butterfly", 0.60)
        return x + 0.01 * torch.randn_like(x)
'''
new = '''    def _primitive_transition_prior(self) -> torch.Tensor:
        phase_strength = float(getattr(self.cfg, "phase_prior_strength", 0.0))
        if phase_strength <= 0.0:
            # No role prior: transitions start near uniform logits with tiny noise.
            return torch.zeros(self.P, self.P) + 0.01 * torch.randn(self.P, self.P)
        x = torch.eye(self.P) * (0.35 * phase_strength)
        idx = {name: i for i, name in enumerate(PRIMITIVES)}
        def link(a: str, b: str, v: float) -> None:
            if a in idx and b in idx:
                x[idx[a], idx[b]] = v * phase_strength
        if self.phase == "extract":
            link("ctx_matrix", "channel_butterfly", 0.75)
            link("channel_butterfly", "phase_matrix", 0.60)
        elif self.phase == "compare":
            link("low_rank", "product_gate", 0.75)
            link("ctx_matrix", "phase_matrix", 0.55)
        elif self.phase == "suppress":
            link("block_butterfly", "phase_matrix", 0.80)
            link("product_gate", "phase_matrix", 0.60)
        elif self.phase == "aggregate":
            link("phase_matrix", "block_butterfly", 0.75)
            link("block_butterfly", "channel_butterfly", 0.60)
        return x + 0.01 * torch.randn_like(x)
'''
if old in txt:
    txt = txt.replace(old, new)

# Config dictionary.
if '"phase_prior_strength": float(getattr(self.cfg, "phase_prior_strength", 0.0)),' not in txt:
    txt = txt.replace('''            "step_alive_init": float(getattr(self.cfg, "step_alive_init", 1.65)),
''', '''            "step_alive_init": float(getattr(self.cfg, "step_alive_init", 1.65)),
            "phase_prior_strength": float(getattr(self.cfg, "phase_prior_strength", 0.0)),
''')

if txt != orig:
    p.write_text(txt, encoding="utf-8")
    print("[patch] phase-free core support applied")
else:
    print("[patch] phase-free core support already present")
PY

python - <<'PY'
from pathlib import Path
p = Path("matrix_program_core/transfer_audio_assembler.py")
txt = p.read_text(encoding="utf-8")
orig = txt

# Pass phase_prior_strength into AssemblerConfig.
if 'phase_prior_strength=float(getattr(args, "phase_prior_strength", 0.0))' not in txt:
    txt = txt.replace('''            step_alive_init=float(getattr(args, "step_alive_init", 1.65)),
        )
''', '''            step_alive_init=float(getattr(args, "step_alive_init", 1.65)),
            phase_prior_strength=float(getattr(args, "phase_prior_strength", 0.0)),
        )
''')
    txt = txt.replace('''            use_deltas=True,
        )
''', '''            use_deltas=True,
            phase_prior_strength=float(getattr(args, "phase_prior_strength", 0.0)),
        )
''')

# Add similarity losses.
marker = '''def aux_losses(logits: torch.Tensor, aux, haux, args, flow_targets, skill_weights) -> Dict[str, torch.Tensor]:
'''
helpers = r'''def layer_program_similarity_loss(slots: torch.Tensor, layers: int, steps: int, blocks: int, margin: float) -> torch.Tensor:
    # slots [N,T*B,D]. Build layer summaries [N,L,D] and penalize too-similar layers.
    s = slots.float()
    N, SB, D = s.shape
    L, S, B = int(layers), int(steps), int(blocks)
    need = L * S * B
    if L <= 1 or SB < need:
        return torch.zeros((), device=slots.device)
    x = s[:, :need].view(N, L, S, B, D).mean(dim=(2, 3))
    x = F.normalize(x, dim=-1)
    sim = torch.einsum("nld,nmd->nlm", x, x)
    eye = torch.eye(L, device=slots.device, dtype=sim.dtype).view(1, L, L)
    off = sim - eye
    return F.relu(off - float(margin)).mean()


def step_program_similarity_loss(slots: torch.Tensor, layers: int, steps: int, blocks: int, margin: float) -> torch.Tensor:
    # Penalize adjacent/near step summaries becoming clones.
    s = slots.float()
    N, SB, D = s.shape
    L, S, B = int(layers), int(steps), int(blocks)
    T = L * S
    need = T * B
    if T <= 1 or SB < need:
        return torch.zeros((), device=slots.device)
    x = s[:, :need].view(N, T, B, D).mean(dim=2)
    x = F.normalize(x, dim=-1)
    sim = torch.einsum("ntd,nud->ntu", x, x)
    eye = torch.eye(T, device=slots.device, dtype=sim.dtype).view(1, T, T)
    off = sim - eye
    # Adjacent steps are allowed to be somewhat related, but not identical.
    return F.relu(off - float(margin)).mean()


'''
if helpers not in txt:
    if marker not in txt:
        raise SystemExit("PATCH_FAIL: aux_losses marker not found")
    txt = txt.replace(marker, helpers + marker)

# Add losses in aux_losses.
if 'out["layer_sim"]' not in txt:
    txt = txt.replace('''    out["slot_div"] = slot_diversity_loss(aux.slots)
''', '''    out["slot_div"] = slot_diversity_loss(aux.slots)
    out["layer_sim"] = layer_program_similarity_loss(aux.slots, args.layers, args.steps, args.blocks, args.layer_sim_margin)
    out["step_sim"] = step_program_similarity_loss(aux.slots, args.layers, args.steps, args.blocks, args.step_sim_margin)
''')

# Add loss terms.
if 'args.lambda_layer_sim * losses["layer_sim"]' not in txt:
    txt = txt.replace('''            loss = loss + args.lambda_slot_div * losses["slot_div"]
''', '''            loss = loss + args.lambda_slot_div * losses["slot_div"]
            loss = loss + args.lambda_layer_sim * losses["layer_sim"]
            loss = loss + args.lambda_step_sim * losses["step_sim"]
''')

# Add CSV fields.
if '"layer_sim", "step_sim"' not in txt:
    txt = txt.replace('''"phase_balance", "slot_div",''', '''"phase_balance", "slot_div", "layer_sim", "step_sim",''')

# Add parser args if missing.
if '--lambda-layer-sim' not in txt:
    txt = txt.replace('''    p.add_argument("--lambda-slot-div", type=float, default=0.002)
''', '''    p.add_argument("--lambda-slot-div", type=float, default=0.002)
    p.add_argument("--lambda-layer-sim", type=float, default=0.000)
    p.add_argument("--lambda-step-sim", type=float, default=0.000)
    p.add_argument("--layer-sim-margin", type=float, default=0.25)
    p.add_argument("--step-sim-margin", type=float, default=0.45)
''')

if txt != orig:
    p.write_text(txt, encoding="utf-8")
    print("[patch] phase-free similarity losses applied to transfer_audio_assembler.py")
else:
    print("[patch] phase-free similarity losses already present")
PY

python -m py_compile matrix_program_core/assembler_core.py matrix_program_core/transfer_audio_assembler.py

grep -n "phase_prior_strength\|layer_sim\|step_sim\|_primitive_slot_prior\|_primitive_transition_prior" matrix_program_core/assembler_core.py matrix_program_core/transfer_audio_assembler.py | head -140
