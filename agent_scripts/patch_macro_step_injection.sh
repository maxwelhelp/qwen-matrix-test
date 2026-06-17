#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

python - <<'PY'
from pathlib import Path
p = Path("matrix_program_core/assembler_core.py")
txt = p.read_text(encoding="utf-8")
orig = txt

# Add macro attach/bias methods after _context_flow_bias.
marker = '''    def _primitive_outputs(self, read_ctx: torch.Tensor) -> torch.Tensor:\n'''
if "def attach_macro_step_bank" not in txt:
    insert = r'''    def attach_macro_step_bank(self, bank: Dict[str, torch.Tensor], max_gain: float = 0.25, train_selector: bool = True) -> None:
        """Attach frozen soft macro-step prototypes to this step.

        Macros are not routers. They are additive log-probability biases over the
        existing dense flow logits. Selection is soft per example/block.
        """
        M = int(bank["macro_read"].shape[0])
        if M <= 0:
            return
        self.macro_count = M
        self.macro_max_gain = float(max_gain)
        self.macro_selector = nn.Linear(self.D, M, bias=True)
        nn.init.zeros_(self.macro_selector.weight)
        nn.init.zeros_(self.macro_selector.bias)
        self.macro_gain_logit = nn.Parameter(torch.tensor(-2.0))
        eps = 1e-6
        def logbuf(name: str, tensor: torch.Tensor):
            value = tensor.float().clamp_min(eps).log()
            self.register_buffer(name, value, persistent=False)
        logbuf("macro_read_logits", bank["macro_read"])
        logbuf("macro_primitive_logits", bank["macro_primitive"])
        logbuf("macro_slot_transition_logits", bank["macro_slot_transition"])
        logbuf("macro_primitive_transition_logits", bank["macro_primitive_transition"])
        logbuf("macro_composition_logits", bank["macro_composition"])
        logbuf("macro_write_logits", bank["macro_write"])
        for p in self.macro_selector.parameters():
            p.requires_grad = bool(train_selector)
        self.macro_gain_logit.requires_grad = bool(train_selector)

    def _macro_flow_bias(self, cells: torch.Tensor):
        if not hasattr(self, "macro_selector") or int(getattr(self, "macro_count", 0)) <= 0:
            z_read = cells.new_zeros(cells.shape[0], self.B, self.K, self.A)
            z_prim = cells.new_zeros(cells.shape[0], self.B, self.K, self.P)
            z_slot = cells.new_zeros(cells.shape[0], self.B, self.K, self.K)
            z_ptrans = cells.new_zeros(cells.shape[0], self.P, self.P)
            z_comp = cells.new_zeros(cells.shape[0], self.B, self.K)
            z_write = cells.new_zeros(cells.shape[0], self.B, self.A)
            info = {"macro_entropy": cells.new_tensor(0.0), "macro_top1_usage": cells.new_tensor(0.0), "macro_gain": cells.new_tensor(0.0)}
            return z_read, z_prim, z_slot, z_ptrans, z_comp, z_write, info
        state = cells[:, : self.B]                                           # [N,B,D]
        global_ctx = cells.mean(dim=1, keepdim=True).expand(-1, self.B, -1)
        block_ctx = state + 0.35 * global_ctx + self.layer_step_key.to(device=cells.device, dtype=cells.dtype).view(1, 1, -1)
        logits = self.macro_selector(block_ctx.float())                      # [N,B,M]
        weight = torch.softmax(logits, dim=-1).to(cells.dtype)
        gain = (float(getattr(self, "macro_max_gain", 0.25)) * torch.sigmoid(self.macro_gain_logit.float())).to(device=cells.device, dtype=cells.dtype)
        mr = self.macro_read_logits.to(device=cells.device, dtype=cells.dtype)
        mp = self.macro_primitive_logits.to(device=cells.device, dtype=cells.dtype)
        ms = self.macro_slot_transition_logits.to(device=cells.device, dtype=cells.dtype)
        mt = self.macro_primitive_transition_logits.to(device=cells.device, dtype=cells.dtype)
        mc = self.macro_composition_logits.to(device=cells.device, dtype=cells.dtype)
        mw = self.macro_write_logits.to(device=cells.device, dtype=cells.dtype)
        read = torch.einsum("nbm,mka->nbka", weight, mr)
        prim = torch.einsum("nbm,mkp->nbkp", weight, mp)
        slot = torch.einsum("nbm,mkj->nbkj", weight, ms)
        comp = torch.einsum("nbm,mk->nbk", weight, mc)
        write = torch.einsum("nbm,ma->nba", weight, mw)
        step_weight = weight.mean(dim=1)
        ptrans = torch.einsum("nm,mpq->npq", step_weight, mt)
        ent = _entropy(weight, dim=-1).mean() / math.log(max(2, int(self.macro_count)))
        top1 = weight.max(dim=-1).values.mean()
        info = {"macro_entropy": ent.detach().float(), "macro_top1_usage": top1.detach().float(), "macro_gain": gain.detach().float()}
        return gain * read, gain * prim, gain * slot, gain * ptrans, gain * comp, gain * write, info

'''
    if marker not in txt:
        raise SystemExit("PATCH_FAIL: _primitive_outputs marker not found")
    txt = txt.replace(marker, insert + marker)

# Insert macro bias in forward.
old = '''        read_b, prim_b, slot_b, ptrans_b, comp_b, write_b = self._context_flow_bias(cells)
        read_e, prim_e, slot_e, ptrans_e, comp_e, write_e = self.flow_editor(cells, use_deltas=self.cfg.use_deltas)
        read_w = torch.softmax(read_logits.float().unsqueeze(0) + read_b.float() + read_e.float(), dim=-1).to(cells.dtype)        # [N,B,K,A]
        prim_w = torch.softmax(prim_logits.float().unsqueeze(0) + prim_b.float() + prim_e.float(), dim=-1).to(cells.dtype)        # [N,B,K,P]
        slot_trans = torch.softmax(slot_trans_logits.float().unsqueeze(0) + slot_b.float() + slot_e.float(), dim=-1).to(cells.dtype) # [N,B,K,K]
        prim_trans = torch.softmax(prim_trans_logits.float().unsqueeze(0) + ptrans_b.float() + ptrans_e.float(), dim=-1).to(cells.dtype) # [N,P,P]
        comp_w = torch.softmax(comp_logits.float().unsqueeze(0) + comp_b.float() + comp_e.float(), dim=-1).to(cells.dtype)        # [N,B,K]
        write_w = torch.softmax(write_logits.float().unsqueeze(0) + write_b.float() + write_e.float(), dim=-1).to(cells.dtype)      # [N,B,A]
'''
new = '''        read_b, prim_b, slot_b, ptrans_b, comp_b, write_b = self._context_flow_bias(cells)
        read_e, prim_e, slot_e, ptrans_e, comp_e, write_e = self.flow_editor(cells, use_deltas=self.cfg.use_deltas)
        read_m, prim_m, slot_m, ptrans_m, comp_m, write_m, macro_info = self._macro_flow_bias(cells)
        read_w = torch.softmax(read_logits.float().unsqueeze(0) + read_b.float() + read_e.float() + read_m.float(), dim=-1).to(cells.dtype)        # [N,B,K,A]
        prim_w = torch.softmax(prim_logits.float().unsqueeze(0) + prim_b.float() + prim_e.float() + prim_m.float(), dim=-1).to(cells.dtype)        # [N,B,K,P]
        slot_trans = torch.softmax(slot_trans_logits.float().unsqueeze(0) + slot_b.float() + slot_e.float() + slot_m.float(), dim=-1).to(cells.dtype) # [N,B,K,K]
        prim_trans = torch.softmax(prim_trans_logits.float().unsqueeze(0) + ptrans_b.float() + ptrans_e.float() + ptrans_m.float(), dim=-1).to(cells.dtype) # [N,P,P]
        comp_w = torch.softmax(comp_logits.float().unsqueeze(0) + comp_b.float() + comp_e.float() + comp_m.float(), dim=-1).to(cells.dtype)        # [N,B,K]
        write_w = torch.softmax(write_logits.float().unsqueeze(0) + write_b.float() + write_e.float() + write_m.float(), dim=-1).to(cells.dtype)      # [N,B,A]
'''
if old in txt:
    txt = txt.replace(old, new)

# Add macro info into info dict.
old = '''            "entropy_write": _entropy(write_w, dim=-1).mean().detach(),
        }
'''
new = '''            "entropy_write": _entropy(write_w, dim=-1).mean().detach(),
            "macro_entropy": macro_info["macro_entropy"],
            "macro_top1_usage": macro_info["macro_top1_usage"],
            "macro_gain": macro_info["macro_gain"],
        }
'''
if old in txt:
    txt = txt.replace(old, new)

# Add ent_acc macro keys and accumulation.
txt = txt.replace('''        ent_acc: Dict[str, List[torch.Tensor]] = {
            "read": [], "primitive": [], "slot_transition": [], "primitive_transition": [], "write": []
        }
''', '''        ent_acc: Dict[str, List[torch.Tensor]] = {
            "read": [], "primitive": [], "slot_transition": [], "primitive_transition": [], "write": [],
            "macro_entropy": [], "macro_top1_usage": [], "macro_gain": []
        }
''')
for key in ("macro_entropy", "macro_top1_usage", "macro_gain"):
    line = f'            ent_acc["{key}"].append(info["{key}"])\n'
    if line.strip() not in txt:
        txt = txt.replace('''            ent_acc["write"].append(info["entropy_write"])
''', '''            ent_acc["write"].append(info["entropy_write"])
            ent_acc["macro_entropy"].append(info["macro_entropy"])
            ent_acc["macro_top1_usage"].append(info["macro_top1_usage"])
            ent_acc["macro_gain"].append(info["macro_gain"])
''')
        break

# Add core load_macro_step_bank method before init_cells.
marker2 = '''    def init_cells(self, evidence: torch.Tensor) -> torch.Tensor:\n'''
if "def load_macro_step_bank" not in txt:
    insert2 = r'''    def load_macro_step_bank(self, path: str, max_gain: float = 0.25, train_selector: bool = True, device: torch.device | None = None) -> Dict[str, object]:
        bank = torch.load(path, map_location="cpu")
        required = ["macro_read", "macro_primitive", "macro_slot_transition", "macro_primitive_transition", "macro_composition", "macro_write"]
        missing = [k for k in required if k not in bank]
        if missing:
            raise KeyError(f"macro step bank is missing keys: {missing}")
        M = int(bank["macro_read"].shape[0])
        if int(bank["macro_read"].shape[1]) != self.K or int(bank["macro_read"].shape[2]) != self.A:
            raise ValueError(f"macro read shape mismatch: got {tuple(bank['macro_read'].shape)}, expected [M,{self.K},{self.A}]")
        if int(bank["macro_primitive"].shape[-1]) != self.P:
            raise ValueError(f"macro primitive P mismatch: got {tuple(bank['macro_primitive'].shape)}, expected P={self.P}")
        for step in self.steps:
            step.attach_macro_step_bank(bank, max_gain=max_gain, train_selector=train_selector)
            if device is not None:
                step.to(device)
        return {"path": str(path), "macros": M, "max_gain": float(max_gain), "train_selector": bool(train_selector), "metrics": bank.get("macro_metrics", {})}

'''
    if marker2 not in txt:
        raise SystemExit("PATCH_FAIL: init_cells marker not found")
    txt = txt.replace(marker2, insert2 + marker2)

# Adjust freeze_for_mode editor_delta to train macro selector/gain.
old = '''            for name, p in self.named_parameters():
                p.requires_grad = name.endswith("_delta") and any(key in name for key in allowed)
            return
'''
new = '''            for name, p in self.named_parameters():
                p.requires_grad = (name.endswith("_delta") and any(key in name for key in allowed)) or ("macro_selector" in name) or ("macro_gain_logit" in name)
            return
'''
if old in txt:
    txt = txt.replace(old, new)

if txt != orig:
    p.write_text(txt, encoding="utf-8")
    print("[patch] assembler_core macro step injection applied")
else:
    print("[patch] assembler_core macro step injection already applied")
PY

python - <<'PY'
from pathlib import Path
p = Path("matrix_program_core/transfer_audio_assembler.py")
txt = p.read_text(encoding="utf-8")
orig = txt

# Load macro bank after checkpoints and before configure_train_mode.
old = '''    if args.init_checkpoint:
        ckpt = torch.load(args.init_checkpoint, map_location=device)
        state = ckpt.get("model", ckpt)
        missing, unexpected = model.load_state_dict(state, strict=False)
        print(f"loaded full task checkpoint: {args.init_checkpoint} missing={len(missing)} unexpected={len(unexpected)}", flush=True)

    configure_train_mode(model, args.train_mode, args.train_input_adapter, args.train_head, args.train_task_context)
'''
new = '''    if args.init_checkpoint:
        ckpt = torch.load(args.init_checkpoint, map_location=device)
        state = ckpt.get("model", ckpt)
        missing, unexpected = model.load_state_dict(state, strict=False)
        print(f"loaded full task checkpoint: {args.init_checkpoint} missing={len(missing)} unexpected={len(unexpected)}", flush=True)
    if args.macro_step_bank:
        macro_info = model.assembler_core.load_macro_step_bank(
            args.macro_step_bank,
            max_gain=args.macro_gain,
            train_selector=not args.freeze_macro_selector,
            device=torch.device(device),
        )
        print(f"loaded macro step bank: {macro_info}", flush=True)

    configure_train_mode(model, args.train_mode, args.train_input_adapter, args.train_head, args.train_task_context)
'''
if old in txt:
    txt = txt.replace(old, new)

# Add macro CSV fields.
txt = txt.replace('''        "pair_update_norm", "class_write",
''', '''        "pair_update_norm", "class_write", "macro_entropy", "macro_top1_usage", "macro_gain",
''')

# Add parser args.
old = '''    p.add_argument("--skill-target-pack", default="")
    p.add_argument("--use-prior-skill-targets", action="store_true")
    p.add_argument("--train-mode", choices=["freeze_core", "editor_delta", "delta", "full"], default="delta")
'''
new = '''    p.add_argument("--skill-target-pack", default="")
    p.add_argument("--use-prior-skill-targets", action="store_true")
    p.add_argument("--macro-step-bank", default="")
    p.add_argument("--macro-gain", type=float, default=0.18)
    p.add_argument("--freeze-macro-selector", action="store_true")
    p.add_argument("--train-mode", choices=["freeze_core", "editor_delta", "delta", "full"], default="delta")
'''
if old in txt:
    txt = txt.replace(old, new)

if txt != orig:
    p.write_text(txt, encoding="utf-8")
    print("[patch] transfer_audio macro args/load applied")
else:
    print("[patch] transfer_audio macro patch already applied")
PY

python -m py_compile matrix_program_core/assembler_core.py matrix_program_core/transfer_audio_assembler.py matrix_program_core/build_macro_step_bank.py

grep -n "macro_step_bank\|macro_selector\|macro_entropy\|macro-gain\|macro-step-bank" matrix_program_core/assembler_core.py matrix_program_core/transfer_audio_assembler.py | head -120
