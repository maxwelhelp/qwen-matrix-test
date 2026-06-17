#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

python - <<'PY'
from pathlib import Path

p = Path("matrix_program_core/assembler_core.py")
txt = p.read_text(encoding="utf-8")
orig = txt

# Add context-conditioned dense soft-flow. This is not a router: no top-k, no argmax,
# just additive logits produced by a small projection from current cells.
if "self.ctx_flow = nn.Linear" not in txt:
    needle = '''        self.write_gate_logit = nn.Parameter(torch.full((self.B,), -0.15))

        # LoRA-like task deltas. In delta mode only these are trainable.
'''
    insert = '''        self.write_gate_logit = nn.Parameter(torch.full((self.B,), -0.15))

        # Context-conditioned dense soft-flow. This lets the same assembler core
        # choose different valid matrix programs for different evidence/cell states
        # without discrete routing. The base projection is learned in pretrain;
        # ctx_flow_delta is LoRA-like and trainable in delta mode.
        self.flow_context_size = (
            self.B * self.K * self.A
            + self.B * self.K * self.P
            + self.B * self.K * self.K
            + self.P * self.P
            + self.B * self.K
            + self.B * self.A
        )
        self.ctx_flow = nn.Linear(self.D, self.flow_context_size, bias=False)
        self.ctx_flow_delta = nn.Linear(self.D, self.flow_context_size, bias=False)
        nn.init.zeros_(self.ctx_flow.weight)
        nn.init.zeros_(self.ctx_flow_delta.weight)

        # LoRA-like task deltas. In delta mode only these are trainable.
'''
    if needle not in txt:
        raise SystemExit("PATCH_FAIL: ctx_flow insertion point not found")
    txt = txt.replace(needle, insert)

if "def _context_flow_bias" not in txt:
    needle = '''    def _eff(self, base: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
        return base + delta if self.cfg.use_deltas else base

    def _primitive_outputs(self, read_ctx: torch.Tensor) -> torch.Tensor:
'''
    insert = '''    def _eff(self, base: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
        return base + delta if self.cfg.use_deltas else base

    def _context_flow_bias(self, cells: torch.Tensor):
        # cells: [N,A,D] -> per-example additive logits for all flow matrices.
        ctx = cells.mean(dim=1).float()
        flat = self.ctx_flow(ctx)
        if self.cfg.use_deltas:
            flat = flat + self.ctx_flow_delta(ctx)
        flat = flat.to(device=cells.device, dtype=cells.dtype)
        sizes = [
            self.B * self.K * self.A,
            self.B * self.K * self.P,
            self.B * self.K * self.K,
            self.P * self.P,
            self.B * self.K,
            self.B * self.A,
        ]
        r, ps, st, pt, comp, wr = torch.split(flat, sizes, dim=-1)
        return (
            r.view(cells.shape[0], self.B, self.K, self.A),
            ps.view(cells.shape[0], self.B, self.K, self.P),
            st.view(cells.shape[0], self.B, self.K, self.K),
            pt.view(cells.shape[0], self.P, self.P),
            comp.view(cells.shape[0], self.B, self.K),
            wr.view(cells.shape[0], self.B, self.A),
        )

    def _primitive_outputs(self, read_ctx: torch.Tensor) -> torch.Tensor:
'''
    if needle not in txt:
        raise SystemExit("PATCH_FAIL: _eff block not found")
    txt = txt.replace(needle, insert)

old = '''        read_w = torch.softmax(read_logits.float(), dim=-1).to(cells.dtype)        # [B,K,A]
        prim_w = torch.softmax(prim_logits.float(), dim=-1).to(cells.dtype)        # [B,K,P]
        slot_trans = torch.softmax(slot_trans_logits.float(), dim=-1).to(cells.dtype) # [B,K,K]
        prim_trans = torch.softmax(prim_trans_logits.float(), dim=-1).to(cells.dtype) # [P,P]
        comp_w = torch.softmax(comp_logits.float(), dim=-1).to(cells.dtype)        # [B,K]
        write_w = torch.softmax(write_logits.float(), dim=-1).to(cells.dtype)      # [B,A]

        read_ctx = torch.einsum("bka,nad->nbkd", read_w, cells)                   # [N,B,K,D]
        prim_out = self._primitive_outputs(read_ctx)                               # [N,B,K,P,D]

        # Factorized transitions: primitive type transition and slot transition.
        prim_mixed = torch.einsum("pq,nbkqd->nbkpd", prim_trans, prim_out)        # [N,B,K,P,D]
        slot_mixed = torch.einsum("bkj,nbjpd->nbkpd", slot_trans, prim_mixed)     # [N,B,K,P,D]

        slot_val = torch.einsum("bkp,nbkpd->nbkd", prim_w, slot_mixed)             # [N,B,K,D]
        update = torch.einsum("bk,nbkd->nbd", comp_w, slot_val)                   # [N,B,D]
        update = self.norm(self.drop(update))

        write_gate = torch.sigmoid(self.write_gate_logit.to(device=cells.device, dtype=cells.dtype)).view(1, self.B, 1)
        delta_cells = torch.einsum("ba,nbd->nad", write_w, write_gate * update)    # [N,A,D]
'''
new = '''        read_b, prim_b, slot_b, ptrans_b, comp_b, write_b = self._context_flow_bias(cells)
        read_w = torch.softmax(read_logits.float().unsqueeze(0) + read_b.float(), dim=-1).to(cells.dtype)        # [N,B,K,A]
        prim_w = torch.softmax(prim_logits.float().unsqueeze(0) + prim_b.float(), dim=-1).to(cells.dtype)        # [N,B,K,P]
        slot_trans = torch.softmax(slot_trans_logits.float().unsqueeze(0) + slot_b.float(), dim=-1).to(cells.dtype) # [N,B,K,K]
        prim_trans = torch.softmax(prim_trans_logits.float().unsqueeze(0) + ptrans_b.float(), dim=-1).to(cells.dtype) # [N,P,P]
        comp_w = torch.softmax(comp_logits.float().unsqueeze(0) + comp_b.float(), dim=-1).to(cells.dtype)        # [N,B,K]
        write_w = torch.softmax(write_logits.float().unsqueeze(0) + write_b.float(), dim=-1).to(cells.dtype)      # [N,B,A]

        read_ctx = torch.einsum("nbka,nad->nbkd", read_w, cells)                   # [N,B,K,D]
        prim_out = self._primitive_outputs(read_ctx)                               # [N,B,K,P,D]

        # Factorized transitions: primitive type transition and slot transition.
        prim_mixed = torch.einsum("npq,nbkqd->nbkpd", prim_trans, prim_out)        # [N,B,K,P,D]
        slot_mixed = torch.einsum("nbkj,nbjpd->nbkpd", slot_trans, prim_mixed)     # [N,B,K,P,D]

        slot_val = torch.einsum("nbkp,nbkpd->nbkd", prim_w, slot_mixed)             # [N,B,K,D]
        update = torch.einsum("nbk,nbkd->nbd", comp_w, slot_val)                   # [N,B,D]
        update = self.norm(self.drop(update))

        write_gate = torch.sigmoid(self.write_gate_logit.to(device=cells.device, dtype=cells.dtype)).view(1, self.B, 1)
        delta_cells = torch.einsum("nba,nbd->nad", write_w, write_gate * update)    # [N,A,D]
'''
if old in txt:
    txt = txt.replace(old, new)
elif 'read_b, prim_b, slot_b, ptrans_b, comp_b, write_b = self._context_flow_bias(cells)' not in txt:
    raise SystemExit("PATCH_FAIL: forward flow block not found")

# Keep gradients for flow losses; harmless if patch_assembler_flow_grad was not run.
for a, b in {
    '"read_flow": read_w.detach().float(),': '"read_flow": read_w.float(),',
    '"primitive_slot_flow": prim_w.detach().float(),': '"primitive_slot_flow": prim_w.float(),',
    '"slot_transition_flow": slot_trans.detach().float(),': '"slot_transition_flow": slot_trans.float(),',
    '"primitive_transition_flow": prim_trans.detach().float(),': '"primitive_transition_flow": prim_trans.float(),',
    '"slot_composition_flow": comp_w.detach().float(),': '"slot_composition_flow": comp_w.float(),',
    '"write_flow": write_w.detach().float(),': '"write_flow": write_w.float(),',
}.items():
    txt = txt.replace(a, b)

old = '''def flow_kl(pred: torch.Tensor, target: torch.Tensor, dim: int = -1) -> torch.Tensor:
    pred = pred.float()
    target = target.to(device=pred.device, dtype=pred.dtype)
    target = target / target.sum(dim=dim, keepdim=True).clamp_min(1e-8)
    pred = pred / pred.sum(dim=dim, keepdim=True).clamp_min(1e-8)
    return F.kl_div(pred.clamp_min(1e-8).log(), target, reduction="batchmean")
'''
new = '''def flow_kl(pred: torch.Tensor, target: torch.Tensor, dim: int = -1) -> torch.Tensor:
    pred = pred.float()
    target = target.to(device=pred.device, dtype=pred.dtype)
    # Targets are usually [T,...], while context-conditioned predictions are [T,N,...].
    # Add broadcast dimensions after time until ranks match.
    while target.ndim < pred.ndim:
        target = target.unsqueeze(1)
    target = target / target.sum(dim=dim, keepdim=True).clamp_min(1e-8)
    pred = pred / pred.sum(dim=dim, keepdim=True).clamp_min(1e-8)
    return F.kl_div(pred.clamp_min(1e-8).log(), target.expand_as(pred), reduction="batchmean")
'''
if old in txt:
    txt = txt.replace(old, new)
elif 'target.expand_as(pred)' not in txt:
    raise SystemExit("PATCH_FAIL: flow_kl block not found")

if txt != orig:
    p.write_text(txt, encoding="utf-8")
    print("[patch] context-conditioned soft-flow enabled")
else:
    print("[patch] already applied")
PY

grep -n "ctx_flow\|_context_flow_bias\|target.expand_as\|nbka,nad" matrix_program_core/assembler_core.py
