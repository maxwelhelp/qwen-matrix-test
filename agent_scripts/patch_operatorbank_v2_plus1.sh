#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

python - <<'PY'
from pathlib import Path
p = Path("matrix_program_core/assembler_core.py")
txt = p.read_text(encoding="utf-8")
orig = txt

# Config flags.
txt = txt.replace('''    channel_stages: int = 3
    dropout: float = 0.04
    use_deltas: bool = True
''', '''    channel_stages: int = 3
    dropout: float = 0.04
    use_deltas: bool = True
    operator_v2: bool = False
    step_alive_init: float = 1.65
''')

# Add OperatorBankV2 parameters inside AssemblerStep.__init__ after norm.
marker = '''        self.drop = nn.Dropout(cfg.dropout)
        self.norm = nn.LayerNorm(self.D)
'''
insert = '''        self.drop = nn.Dropout(cfg.dropout)
        self.norm = nn.LayerNorm(self.D)

        # OperatorBankV2 keeps the external primitive count P unchanged for
        # checkpoint/dataset compatibility, but each primitive becomes a soft
        # family of fast matrix operators: low-rank sizes, butterfly depths,
        # local Toeplitz-like smooth/diff, Haar wavelet-like channel transform,
        # diagonal gates, and richer phase variants.
        self.operator_v2_enabled = bool(getattr(cfg, "operator_v2", False))
        self.step_alive_logit = nn.Parameter(torch.tensor(float(getattr(cfg, "step_alive_init", 1.65))))
        self.step_alive_delta = nn.Parameter(torch.zeros(()))
        self.opv2_channel_logits = nn.Parameter(torch.tensor([1.20, -0.15, -0.35]))
        self.opv2_block_logits = nn.Parameter(torch.tensor([1.10, -0.10, -0.35]))
        self.opv2_lowrank_logits = nn.Parameter(torch.tensor([0.20, 0.70, 0.35]))
        self.opv2_ctx_logits = nn.Parameter(torch.tensor([1.00, -0.10, -0.25, -0.35]))
        self.opv2_product_logits = nn.Parameter(torch.tensor([0.90, -0.05, -0.25]))
        self.phase_extra_logits = nn.Parameter(torch.tensor([-0.10, -0.20, -0.25, -0.35]))
        self.opv2_channel_delta = nn.Parameter(torch.zeros(3))
        self.opv2_block_delta = nn.Parameter(torch.zeros(3))
        self.opv2_lowrank_delta = nn.Parameter(torch.zeros(3))
        self.opv2_ctx_delta = nn.Parameter(torch.zeros(4))
        self.opv2_product_delta = nn.Parameter(torch.zeros(3))
        self.phase_extra_delta = nn.Parameter(torch.zeros(4))
        self.diag_gate = nn.Parameter(torch.zeros(self.D))
        self.diag_bias = nn.Parameter(torch.zeros(self.D))
'''
if insert not in txt:
    if marker not in txt:
        raise SystemExit("PATCH_FAIL: AssemblerStep norm marker not found")
    txt = txt.replace(marker, insert)

# Add helper methods before _primitive_outputs.
marker = '''    def _primitive_outputs(self, read_ctx: torch.Tensor) -> torch.Tensor:
'''
helpers = r'''    def _opv2_weights(self, logits: torch.Tensor, delta: torch.Tensor | None = None, dtype=None, device=None) -> torch.Tensor:
        z = logits
        if self.cfg.use_deltas and delta is not None:
            z = z + delta
        z = z.float()
        w = torch.softmax(z, dim=-1)
        if device is not None:
            w = w.to(device=device)
        if dtype is not None:
            w = w.to(dtype=dtype)
        return w

    def _local_smooth_flat(self, x: torch.Tensor) -> torch.Tensor:
        return 0.50 * x + 0.25 * torch.roll(x, 1, dims=-1) + 0.25 * torch.roll(x, -1, dims=-1)

    def _local_diff_flat(self, x: torch.Tensor) -> torch.Tensor:
        return x - self._local_smooth_flat(x)

    def _haar_flat(self, x: torch.Tensor) -> torch.Tensor:
        # Fast orthogonal-ish Haar step over channel pairs. Shape is preserved.
        D = x.shape[-1]
        if D < 2:
            return x
        even = x[..., 0::2]
        odd = x[..., 1::2]
        m = min(even.shape[-1], odd.shape[-1])
        avg = (even[..., :m] + odd[..., :m]) * 0.70710678
        dif = (even[..., :m] - odd[..., :m]) * 0.70710678
        y = x.clone()
        y[..., 0:2*m:2] = avg
        y[..., 1:2*m:2] = dif
        return y

    def _mix_last(self, variants: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        # variants [...,V,D], weights [V]
        return torch.einsum("v,...vd->...d", weights, variants)

'''
if helpers not in txt:
    if marker not in txt:
        raise SystemExit("PATCH_FAIL: _primitive_outputs marker not found")
    txt = txt.replace(marker, helpers + marker)

old = r'''    def _primitive_outputs(self, read_ctx: torch.Tensor) -> torch.Tensor:
        # read_ctx: [N,B,K,D] -> [N,B,K,P,D]
        N, B, K, D = read_ctx.shape
        flat = read_ctx.reshape(N * B * K, D)
        ctx_m = flat @ self.ctx_w.to(device=read_ctx.device, dtype=read_ctx.dtype)
        channel = self.channel((flat + ctx_m).view(N * B * K, 1, D)).view(N, B, K, D)
        low = ((flat @ self.low_a.to(device=read_ctx.device, dtype=read_ctx.dtype)) @ self.low_b.to(device=read_ctx.device, dtype=read_ctx.dtype)).view(N, B, K, D)
        ctx_m = ctx_m.view(N, B, K, D)
        product = read_ctx * torch.tanh(ctx_m)

        # Block primitive mixes across blocks for every slot k.
        block_in = read_ctx.permute(0, 2, 1, 3).reshape(N * K, B, D)
        block = self.block(block_in).reshape(N, K, B, D).permute(0, 2, 1, 3)

        gate = torch.sigmoid(
            read_ctx @ self.gate_h.to(device=read_ctx.device, dtype=read_ctx.dtype)
            + ctx_m @ self.gate_c.to(device=read_ctx.device, dtype=read_ctx.dtype)
            + self.gate_bias.to(device=read_ctx.device, dtype=read_ctx.dtype)
        )
        phase_candidates = torch.stack([
            channel + 0.50 * ctx_m,
            channel - low,
            -gate * block.mean(dim=1, keepdim=True),
            block + read_ctx.mean(dim=1, keepdim=True),
        ], dim=3)
        phase_logits = self.phase_mix_logits
        if self.cfg.use_deltas:
            phase_logits = phase_logits + self.phase_mix_delta
        phase_w = torch.softmax(phase_logits.float(), dim=-1).to(read_ctx.dtype)
        phase = torch.einsum("f,nbkfd->nbkd", phase_w, phase_candidates)
        return torch.stack([channel, block, low, ctx_m, product, phase], dim=3)
'''
new = r'''    def _primitive_outputs(self, read_ctx: torch.Tensor) -> torch.Tensor:
        # read_ctx: [N,B,K,D] -> [N,B,K,P,D]
        N, B, K, D = read_ctx.shape
        flat = read_ctx.reshape(N * B * K, D)
        dev, dtype = read_ctx.device, read_ctx.dtype

        ctx_lin_flat = flat @ self.ctx_w.to(device=dev, dtype=dtype)
        smooth_flat = self._local_smooth_flat(flat)
        diff_flat = self._local_diff_flat(flat)
        haar_flat = self._haar_flat(flat)
        diag_flat = flat * torch.sigmoid(self.diag_gate.to(device=dev, dtype=dtype).view(1, -1) + self.diag_bias.to(device=dev, dtype=dtype).view(1, -1))

        # Low-rank size selection from slices of the max-rank factorization.
        rank_total = int(self.low_a.shape[1])
        r1 = max(4, rank_total // 4)
        r2 = max(r1, rank_total // 2)
        lows = []
        for r in (r1, r2, rank_total):
            la = self.low_a[:, :r].to(device=dev, dtype=dtype)
            lb = self.low_b[:r, :].to(device=dev, dtype=dtype)
            lows.append((flat @ la) @ lb)
        low_w = self._opv2_weights(self.opv2_lowrank_logits, self.opv2_lowrank_delta, dtype=dtype, device=dev)
        low = self._mix_last(torch.stack(lows, dim=1), low_w).view(N, B, K, D)

        # Context primitive becomes a selectable family: dense ctx, local smooth,
        # local diff, diagonal gate. All are linear/diagonal/Toeplitz-like fast ops.
        ctx_w = self._opv2_weights(self.opv2_ctx_logits, self.opv2_ctx_delta, dtype=dtype, device=dev)
        ctx_m = self._mix_last(torch.stack([ctx_lin_flat, smooth_flat, diff_flat, diag_flat], dim=1), ctx_w).view(N, B, K, D)

        # Channel primitive variants: one butterfly pass, two passes, Haar wavelet-like pass.
        ch1_flat = self.channel((flat + ctx_lin_flat).view(N * B * K, 1, D)).view(N * B * K, D)
        ch2_flat = self.channel((ch1_flat + 0.35 * ctx_lin_flat).view(N * B * K, 1, D)).view(N * B * K, D)
        ch_w = self._opv2_weights(self.opv2_channel_logits, self.opv2_channel_delta, dtype=dtype, device=dev)
        channel = self._mix_last(torch.stack([ch1_flat, ch2_flat, haar_flat], dim=1), ch_w).view(N, B, K, D)

        # Block primitive variants: one block butterfly, two block butterfly passes,
        # and global block mean injection.
        block_in = read_ctx.permute(0, 2, 1, 3).reshape(N * K, B, D)
        block1 = self.block(block_in).reshape(N, K, B, D).permute(0, 2, 1, 3)
        block2_in = block1.permute(0, 2, 1, 3).reshape(N * K, B, D)
        block2 = self.block(block2_in).reshape(N, K, B, D).permute(0, 2, 1, 3)
        block_mean = read_ctx.mean(dim=1, keepdim=True).expand_as(read_ctx)
        bl_w = self._opv2_weights(self.opv2_block_logits, self.opv2_block_delta, dtype=dtype, device=dev)
        block = self._mix_last(torch.stack([block1, block2, block_mean], dim=3), bl_w)

        # Product/gate primitive variants.
        product0 = read_ctx * torch.tanh(ctx_m)
        product1 = read_ctx * torch.sigmoid(ctx_m)
        product2 = diff_flat.view(N, B, K, D) * torch.sigmoid(ctx_m)
        pr_w = self._opv2_weights(self.opv2_product_logits, self.opv2_product_delta, dtype=dtype, device=dev)
        product = self._mix_last(torch.stack([product0, product1, product2], dim=3), pr_w)

        gate = torch.sigmoid(
            read_ctx @ self.gate_h.to(device=dev, dtype=dtype)
            + ctx_m @ self.gate_c.to(device=dev, dtype=dtype)
            + self.gate_bias.to(device=dev, dtype=dtype)
        )
        smooth = smooth_flat.view(N, B, K, D)
        diff = diff_flat.view(N, B, K, D)
        haar = haar_flat.view(N, B, K, D)
        phase_candidates = torch.stack([
            channel + 0.50 * ctx_m,
            channel - low,
            -gate * block.mean(dim=1, keepdim=True),
            block + read_ctx.mean(dim=1, keepdim=True),
            smooth + 0.35 * ctx_m,
            diff + 0.25 * product,
            haar + 0.25 * channel,
            low + product,
        ], dim=3)
        phase_logits = torch.cat([self.phase_mix_logits, self.phase_extra_logits], dim=0)
        if self.cfg.use_deltas:
            phase_logits = phase_logits + torch.cat([self.phase_mix_delta, self.phase_extra_delta], dim=0)
        phase_w = torch.softmax(phase_logits.float(), dim=-1).to(dtype)
        phase = torch.einsum("f,nbkfd->nbkd", phase_w, phase_candidates)
        return torch.stack([channel, block, low, ctx_m, product, phase], dim=3)
'''
if old not in txt:
    raise SystemExit("PATCH_FAIL: original _primitive_outputs body not found; apply on clean main")
txt = txt.replace(old, new)

# Step alive in forward.
txt = txt.replace('''        update = torch.einsum("nbk,nbkd->nbd", comp_w, slot_val)                   # [N,B,D]
        update = self.norm(self.drop(update))

        write_gate = torch.sigmoid(self.write_gate_logit.to(device=cells.device, dtype=cells.dtype)).view(1, self.B, 1)
        delta_cells = torch.einsum("nba,nbd->nad", write_w, write_gate * update)    # [N,A,D]
        next_cells = self.norm(cells + delta_cells)
''', '''        update = torch.einsum("nbk,nbkd->nbd", comp_w, slot_val)                   # [N,B,D]
        update = self.norm(self.drop(update))
        step_alive_logit = self.step_alive_logit
        if self.cfg.use_deltas:
            step_alive_logit = step_alive_logit + self.step_alive_delta
        step_alive = torch.sigmoid(step_alive_logit.to(device=cells.device, dtype=cells.dtype))
        active_update = step_alive * update

        write_gate = torch.sigmoid(self.write_gate_logit.to(device=cells.device, dtype=cells.dtype)).view(1, self.B, 1)
        delta_cells = torch.einsum("nba,nbd->nad", write_w, write_gate * active_update)    # [N,A,D]
        next_cells = self.norm(cells + delta_cells)
''')
txt = txt.replace('''            "update_norms": update.detach().float().norm(dim=-1),
            "slot_values": slot_val.detach(),
''', '''            "update_norms": active_update.detach().float().norm(dim=-1),
            "step_alive": step_alive.detach().float(),
            "slot_values": slot_val.detach(),
''')
txt = txt.replace('''        return next_cells, update, info
''', '''        return next_cells, active_update, info
''')

# Core entropies / step_alive summaries.
txt = txt.replace('''        ent_acc: Dict[str, List[torch.Tensor]] = {
            "read": [], "primitive": [], "slot_transition": [], "primitive_transition": [], "write": []
        }
''', '''        ent_acc: Dict[str, List[torch.Tensor]] = {
            "read": [], "primitive": [], "slot_transition": [], "primitive_transition": [], "write": [], "step_alive": []
        }
''')
txt = txt.replace('''            ent_acc["write"].append(info["entropy_write"])
''', '''            ent_acc["write"].append(info["entropy_write"])
            ent_acc["step_alive"].append(info["step_alive"])
''')

# Config dict additions.
txt = txt.replace('''            "use_deltas": self.cfg.use_deltas,
            "primitives": list(PRIMITIVES),
''', '''            "use_deltas": self.cfg.use_deltas,
            "operator_v2": bool(getattr(self.cfg, "operator_v2", False)),
            "step_alive_init": float(getattr(self.cfg, "step_alive_init", 1.65)),
            "primitives": list(PRIMITIVES),
''')

# Train mode: allow opv2 deltas and step_alive in editor_delta.
txt = txt.replace('''            allowed = (
                "flow_editor.scale_delta",
                "flow_editor.variant_editors",
                "phase_mix_delta",
            )
            for name, p in self.named_parameters():
                p.requires_grad = name.endswith("_delta") and any(key in name for key in allowed)
''', '''            allowed = (
                "flow_editor.scale_delta",
                "flow_editor.variant_editors",
                "phase_mix_delta",
                "phase_extra_delta",
                "opv2_",
                "step_alive",
            )
            for name, p in self.named_parameters():
                p.requires_grad = (name.endswith("_delta") and any(key in name for key in allowed)) or ("step_alive_logit" in name)
''')

if txt != orig:
    p.write_text(txt, encoding="utf-8")
    print("[patch] OperatorBankV2 + step_alive applied to assembler_core.py")
else:
    print("[patch] assembler_core already patched")
PY

python - <<'PY'
from pathlib import Path
p = Path("matrix_program_core/transfer_audio_assembler.py")
txt = p.read_text(encoding="utf-8")
orig = txt

# Pass new cfg args.
txt = txt.replace('''            dropout=args.dropout,
            use_deltas=True,
        )
''', '''            dropout=args.dropout,
            use_deltas=True,
            operator_v2=bool(getattr(args, "operator_v2", False)),
            step_alive_init=float(getattr(args, "step_alive_init", 1.65)),
        )
''')

# Add anti-collapse helper functions after slot_diversity_loss.
marker = '''def aux_losses(logits: torch.Tensor, aux, haux, args, flow_targets, skill_weights) -> Dict[str, torch.Tensor]:
'''
helpers = r'''def _load_balance_loss(p: torch.Tensor, active_floor: float = 0.0) -> torch.Tensor:
    # p [...,C], normalized distribution. Penalize extreme collapse while keeping it soft.
    q = p.float().mean(dim=tuple(range(max(0, p.ndim - 1))))
    q = q / q.sum().clamp_min(1e-8)
    C = q.numel()
    if C <= 1:
        return torch.zeros((), device=p.device)
    uniform = torch.full_like(q, 1.0 / float(C))
    mse = (q - uniform).pow(2).mean()
    ent = -(q.clamp_min(1e-8) * q.clamp_min(1e-8).log()).sum() / math.log(float(C))
    floor = F.relu(torch.tensor(float(active_floor), device=p.device) - ent).pow(2)
    return mse + floor


def primitive_load_balance_loss(aux, floor: float) -> torch.Tensor:
    return _load_balance_loss(aux.primitive_slot_flow, active_floor=floor)


def read_write_cell_balance_loss(aux, floor: float) -> torch.Tensor:
    rw = torch.cat([
        aux.read_flow.float().mean(dim=-2).reshape(-1, aux.read_flow.shape[-1]),
        aux.write_flow.float().reshape(-1, aux.write_flow.shape[-1]),
    ], dim=0)
    return _load_balance_loss(rw, active_floor=floor)


def entropy_band_loss(aux, low: float, high: float) -> torch.Tensor:
    vals = []
    for key in ("read", "primitive", "slot_transition", "primitive_transition", "write"):
        if key in aux.entropies:
            vals.append(aux.entropies[key].float())
    if not vals:
        return torch.zeros((), device=aux.cells.device)
    ent = torch.stack(vals).mean()
    return F.relu(torch.tensor(float(low), device=ent.device) - ent).pow(2) + F.relu(ent - float(high)).pow(2)


def step_alive_budget_loss(aux, target: float) -> torch.Tensor:
    if "step_alive" not in aux.entropies:
        return torch.zeros((), device=aux.cells.device)
    return (aux.entropies["step_alive"].float() - float(target)).pow(2)


'''
if helpers not in txt:
    if marker not in txt:
        raise SystemExit("PATCH_FAIL: aux_losses marker not found")
    txt = txt.replace(marker, helpers + marker)

# Add aux loss outputs.
txt = txt.replace('''    out["slot_div"] = slot_diversity_loss(aux.slots)
    out["logit_norm"] = logits.float().pow(2).mean()
''', '''    out["slot_div"] = slot_diversity_loss(aux.slots)
    out["primitive_balance"] = primitive_load_balance_loss(aux, args.primitive_balance_entropy_floor)
    out["cell_balance"] = read_write_cell_balance_loss(aux, args.cell_balance_entropy_floor)
    out["entropy_band"] = entropy_band_loss(aux, args.entropy_band_low, args.entropy_band_high)
    out["step_alive_budget"] = step_alive_budget_loss(aux, args.step_alive_target)
    out["logit_norm"] = logits.float().pow(2).mean()
''')

# Add losses into train loss.
txt = txt.replace('''            loss = loss + args.lambda_slot_div * losses["slot_div"]
            loss = loss + args.lambda_logit_norm * losses["logit_norm"]
''', '''            loss = loss + args.lambda_slot_div * losses["slot_div"]
            loss = loss + args.lambda_primitive_balance * losses["primitive_balance"]
            loss = loss + args.lambda_cell_balance * losses["cell_balance"]
            loss = loss + args.lambda_entropy_band * losses["entropy_band"]
            loss = loss + args.lambda_step_alive_budget * losses["step_alive_budget"]
            loss = loss + args.lambda_logit_norm * losses["logit_norm"]
''')

# Add fields.
txt = txt.replace('''        "write_budget", "update_alive", "class_read_div", "class_slot_prior", "class_attn_entropy", "phase_balance", "slot_div", "logit_norm",
''', '''        "write_budget", "update_alive", "class_read_div", "class_slot_prior", "class_attn_entropy", "phase_balance", "slot_div",
        "primitive_balance", "cell_balance", "entropy_band", "step_alive_budget", "logit_norm",
''')

# Add parser args after channel-stages.
txt = txt.replace('''    p.add_argument("--channel-stages", type=int, default=3)
    p.add_argument("--pair-slots", type=int, default=12)
''', '''    p.add_argument("--channel-stages", type=int, default=3)
    p.add_argument("--operator-v2", action="store_true")
    p.add_argument("--step-alive-init", type=float, default=1.65)
    p.add_argument("--pair-slots", type=int, default=12)
''')

# Add parser loss args after lambda-slot-div.
txt = txt.replace('''    p.add_argument("--lambda-slot-div", type=float, default=0.002)
    p.add_argument("--lambda-logit-norm", type=float, default=0.0007)
''', '''    p.add_argument("--lambda-slot-div", type=float, default=0.002)
    p.add_argument("--lambda-primitive-balance", type=float, default=0.000)
    p.add_argument("--lambda-cell-balance", type=float, default=0.000)
    p.add_argument("--lambda-entropy-band", type=float, default=0.000)
    p.add_argument("--lambda-step-alive-budget", type=float, default=0.000)
    p.add_argument("--primitive-balance-entropy-floor", type=float, default=0.72)
    p.add_argument("--cell-balance-entropy-floor", type=float, default=0.62)
    p.add_argument("--entropy-band-low", type=float, default=1.05)
    p.add_argument("--entropy-band-high", type=float, default=2.35)
    p.add_argument("--step-alive-target", type=float, default=0.78)
    p.add_argument("--lambda-logit-norm", type=float, default=0.0007)
''')

if txt != orig:
    p.write_text(txt, encoding="utf-8")
    print("[patch] OperatorBankV2 flags/losses applied to transfer_audio_assembler.py")
else:
    print("[patch] transfer_audio_assembler already patched")
PY

python -m py_compile matrix_program_core/assembler_core.py matrix_program_core/transfer_audio_assembler.py

grep -n "operator_v2\|OperatorBankV2\|step_alive\|primitive_balance\|entropy_band" matrix_program_core/assembler_core.py matrix_program_core/transfer_audio_assembler.py | head -160
