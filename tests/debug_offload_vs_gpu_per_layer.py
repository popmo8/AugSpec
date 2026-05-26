"""GPU-full vs offload Mixtral, per-layer per-stage comparison.

The two paths should compute identical numerics (same weights, same dtype,
same hardware) — offload just adds H2D/cache management. Any divergence is
a moe_infinity bug.

For each of 32 MoE layers we capture:
  1. hidden_states going INTO block_sparse_moe
  2. router_logits = block.gate(hidden_states)
  3. selected_experts (top-2 indices)
  4. routing_weights (top-2 normalized)
  5. final MoE output AFTER expert dispatch

…and at the end:
  6. lm_head logits' top-1 token

We monkey-patch BOTH:
  - the offload SyncMixtralSparseMoeBlock.forward
  - the HF MixtralSparseMoeBlock.forward (grabbed from the live HF instance,
    so we get the original stock class, NOT the one moe_infinity globally
    re-bound after init)

Both forward functions write their captures into per-layer dicts.
"""

from __future__ import annotations

import os
import time

import torch
import torch.nn.functional as F

os.environ.setdefault("HF_HOME", "/work/morrisliu07/.cache/huggingface")

from moe_infinity import MoE
from transformers import AutoModelForCausalLM, AutoTokenizer


CKPT = "mistralai/Mixtral-8x7B-v0.1"
OFFLOAD_DIR = "/work/morrisliu07/aug_spec/cache/offload/mixtral"


def banner(s):
    print("\n" + "=" * 70)
    print(f"  {s}")
    print("=" * 70, flush=True)


try:
    # ---------------------------------------------------------------------
    # 1. Load HF Mixtral on GPU (the "reference" path) FIRST, before moe_infinity
    #    can globally rebind MixtralSparseMoeBlock.
    # ---------------------------------------------------------------------
    banner("Step 1/3: load HF Mixtral on cuda:0")
    print(f"  device_count: {torch.cuda.device_count()}", flush=True)
    print(f"  free VRAM before: {torch.cuda.mem_get_info(0)[0] / 1e9:.1f} GB",
          flush=True)
    t0 = time.perf_counter()
    hf_model = AutoModelForCausalLM.from_pretrained(
        CKPT,
        torch_dtype=torch.bfloat16,
        device_map="cuda:0",
    )
    hf_model.eval()
    for p in hf_model.parameters():
        p.requires_grad_(False)
    print(f"  loaded in {time.perf_counter()-t0:.1f}s")
    print(f"  free VRAM after: {torch.cuda.mem_get_info(0)[0] / 1e9:.1f} GB",
          flush=True)

    # Snapshot HF class BEFORE moe_infinity rebinds it.
    HFBlock = type(hf_model.model.layers[0].block_sparse_moe)
    print(f"  HF block class: {HFBlock.__module__}.{HFBlock.__name__}")

    # ---------------------------------------------------------------------
    # 2. Load offload (moe_infinity) — uses small extra VRAM
    # ---------------------------------------------------------------------
    banner("Step 2/3: load offload Mixtral via moe_infinity")
    t0 = time.perf_counter()
    moe = MoE(CKPT, {
        "offload_path": OFFLOAD_DIR,
        "device_memory_ratio": 0.15,
    })
    print(f"  loaded in {time.perf_counter()-t0:.1f}s")
    print(f"  free VRAM after: {torch.cuda.mem_get_info(0)[0] / 1e9:.1f} GB",
          flush=True)

    SyncBlock = type(moe.model.model.layers[0].block_sparse_moe)
    print(f"  Sync block class: {SyncBlock.__module__}.{SyncBlock.__name__}")

    tok = AutoTokenizer.from_pretrained(CKPT)

    # ---------------------------------------------------------------------
    # 3. Monkey-patch both classes' forward to capture intermediates.
    # ---------------------------------------------------------------------
    banner("Step 3/3: set up captures and run forward on both")

    captures_hf: dict[int, dict] = {}
    captures_off: dict[int, dict] = {}

    NUM_EXPERTS = 8
    TOP_K = 2

    def _patched_hf_forward(self, hidden_states):
        bs, sl, hd = hidden_states.shape
        hs = hidden_states.view(-1, hd)
        router_logits = self.gate(hs)

        rw = F.softmax(router_logits, dim=1, dtype=torch.float)
        rw, sel = torch.topk(rw, self.top_k, dim=-1)
        rw /= rw.sum(dim=-1, keepdim=True)
        rw_bf = rw.to(hs.dtype)

        final = torch.zeros((bs * sl, hd), dtype=hs.dtype, device=hs.device)
        emask = F.one_hot(sel, num_classes=self.num_experts).permute(2, 1, 0)
        for e in range(self.num_experts):
            idx, top_x = torch.where(emask[e])
            if top_x.numel() == 0:
                continue
            cur = hs[None, top_x].reshape(-1, hd)
            cur_h = self.experts[e](cur) * rw_bf[top_x, idx, None]
            final.index_add_(0, top_x, cur_h.to(hs.dtype))

        # Identify which layer this is.
        for li, layer in enumerate(hf_model.model.layers):
            if layer.block_sparse_moe is self:
                captures_hf[li] = {
                    "hs_in": hs.detach().float().cpu(),
                    "gate_logits": router_logits.detach().float().cpu(),
                    "selected_experts": sel.detach().cpu(),
                    "routing_weights": rw.detach().float().cpu(),
                    "mlp_out": final.detach().float().cpu(),
                }
                break

        return final.view(bs, sl, hd), router_logits

    def _patched_sync_forward(self, hidden_states):
        bs, sl, hd = hidden_states.shape
        hs = hidden_states.view(-1, hd)
        router_logits = self.gate(hs)

        rw = F.softmax(router_logits, dim=1, dtype=torch.float)
        rw, sel = torch.topk(rw, self.top_k, dim=-1)
        rw /= rw.sum(dim=-1, keepdim=True)
        rw_bf = rw.to(hs.dtype)

        rm_3d = F.one_hot(sel, num_classes=self.num_experts)
        rwm = (rw_bf[:, :, None] * rm_3d).permute(0, 2, 1)
        rm = rm_3d.permute(0, 2, 1)
        rm = torch.logical_or(rm[:, :, 0], rm[:, :, 1])
        rwm = torch.sum(rwm, dim=-1)

        # Save BEFORE dispatch.
        save = {
            "hs_in": hs.detach().float().cpu(),
            "gate_logits": router_logits.detach().float().cpu(),
            "selected_experts": sel.detach().cpu(),
            "routing_weights": rw.detach().float().cpu(),
        }

        self.expert_executor.dispatch_local(
            self.layer_id, hs, rm, rwm)
        final = self.expert_executor.wait_dispatch_local()
        save["mlp_out"] = final.detach().float().cpu()
        captures_off[self.layer_id] = save

        return final.view(bs, sl, hd).to(hs.dtype), router_logits

    HFBlock.forward = _patched_hf_forward
    SyncBlock.forward = _patched_sync_forward

    # Run both
    prompt = "The capital of France is"
    enc = tok(prompt, return_tensors="pt")
    print(f"\n  prompt: {prompt!r}, input_ids: {enc.input_ids.tolist()}")

    toks_gpu = enc.input_ids.to("cuda:0")
    moe._configure_hook(toks_gpu)

    print(f"\n  running HF model.forward on cuda ...", flush=True)
    t0 = time.perf_counter()
    with torch.no_grad():
        out_hf = hf_model(toks_gpu)
    torch.cuda.synchronize()
    print(f"    {time.perf_counter()-t0:.2f}s")

    print(f"\n  running offload model.forward on cuda ...", flush=True)
    t0 = time.perf_counter()
    with torch.no_grad():
        out_off = moe.model(toks_gpu)
    torch.cuda.synchronize()
    print(f"    {time.perf_counter()-t0:.2f}s")

    # ---------------------------------------------------------------------
    # Compare per layer per stage
    # ---------------------------------------------------------------------
    banner("Per-layer comparison")
    print(f"  Captured layers HF: {sorted(captures_hf.keys())[:6]}... ({len(captures_hf)} total)")
    print(f"  Captured layers OF: {sorted(captures_off.keys())[:6]}... ({len(captures_off)} total)")

    print(f"\n  {'L':>3}  {'hs_in_max':>10}  {'gate_max':>10}  "
          f"{'sel_eq':>7}  {'rw_max':>10}  {'mlp_out_max':>12}")
    print(f"  {'-'*3}  {'-'*10}  {'-'*10}  {'-'*7}  {'-'*10}  {'-'*12}")

    first_diverge = None
    for L in sorted(captures_hf):
        if L not in captures_off:
            continue
        hf = captures_hf[L]
        of = captures_off[L]

        d_hs = (hf["hs_in"] - of["hs_in"]).abs().max().item()
        d_gl = (hf["gate_logits"] - of["gate_logits"]).abs().max().item()
        sel_eq = (hf["selected_experts"] == of["selected_experts"]).all().item()
        d_rw = (hf["routing_weights"] - of["routing_weights"]).abs().max().item()
        d_mlp = (hf["mlp_out"] - of["mlp_out"]).abs().max().item()

        flag = "" if d_mlp < 0.5 else " <<< MLP divergence"
        print(f"  {L:>3}  {d_hs:>10.4e}  {d_gl:>10.4e}  "
              f"{str(sel_eq):>7}  {d_rw:>10.4e}  {d_mlp:>12.4e}{flag}")

        if first_diverge is None and d_mlp > 0.5:
            first_diverge = L

    # ---------------------------------------------------------------------
    # Final token agreement
    # ---------------------------------------------------------------------
    banner("Final last-token logits")
    last_off = out_off.logits[0, -1].float().cpu()
    last_hf = out_hf.logits[0, -1].float().cpu()
    print(f"  offload top-5: {last_off.topk(5).indices.tolist()}")
    print(f"  HF      top-5: {last_hf.topk(5).indices.tolist()}")
    print(f"  offload argmax: {last_off.argmax().item()}"
          f" ({tok.decode([last_off.argmax().item()])!r})")
    print(f"  HF      argmax: {last_hf.argmax().item()}"
          f" ({tok.decode([last_hf.argmax().item()])!r})")
    print(f"  argmax match: {last_off.argmax().item() == last_hf.argmax().item()}")

    banner("Verdict")
    if first_diverge is None:
        print("  >>> All 32 MoE layers MATCH (offload ≡ GPU). Bug is elsewhere.")
    else:
        print(f"  >>> First layer where MLP output diverges: L={first_diverge}")
        print(f"      Layer {first_diverge}'s hs_in matched but mlp_out doesn't")
        print(f"      → bug isolated to that layer's dispatch_local pipeline.")

except Exception:
    import traceback
    traceback.print_exc()
finally:
    import sys as _sys
    _sys.stdout.flush()
    _sys.stderr.flush()
    os._exit(0)
