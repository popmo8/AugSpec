"""Narrow the Mixtral bug: Python routing vs C++ dispatch.

We know layer 0's SyncBlock output differs from HF by ~0.92 max for nearly
identical input. The error is INSIDE SyncBlock.forward. Candidates:

  Python routing — F.one_hot + permute + logical_or + sum that constructs
                   (router_mask, routing_weights_mask) before dispatch_local
  C++ dispatch   — dispatch_local → ExpertDispatcher → fused_moe_ffn_into
                   → output × routing_weight → index_add

This test monkey-patches BOTH blocks' forward to capture intermediate state:
  1. gate_logits at layer 0 — should match (gate.weight identical)
  2. routing decisions (selected_experts × top-k weights) — should match
  3. (router_mask, routing_weights_mask) the offload Python builds
     vs an equivalent reconstruction from HF's selected_experts/routing_weights
  4. Final layer-0 MoE output

If (1)-(3) all agree → Python routing is fine → bug is in C++ dispatch.
If any of (1)-(3) disagree → that's the buggy step.

Run:
  python -u tests/debug_routing_vs_dispatch_mixtral.py
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
LAYER = 0


def banner(s):
    print("\n" + "=" * 70)
    print(f"  {s}")
    print("=" * 70, flush=True)


# =============================================================================
# Load both models
# =============================================================================

banner("Load")
t0 = time.perf_counter()
cpu_source = AutoModelForCausalLM.from_pretrained(
    CKPT, torch_dtype=torch.bfloat16, device_map="cpu")
cpu_source.eval()
print(f"  cpu_source loaded in {time.perf_counter()-t0:.1f}s")

t0 = time.perf_counter()
moe = MoE(CKPT, {"offload_path": OFFLOAD_DIR, "device_memory_ratio": 0.15})
print(f"  moe loaded in {time.perf_counter()-t0:.1f}s")

tok = AutoTokenizer.from_pretrained(CKPT)


# =============================================================================
# Monkey-patch the layer-0 forward of BOTH blocks to capture intermediates
# =============================================================================
#
# NOTE: moe_infinity at MoE(...) init replaces
#   transformers.models.mixtral.modeling_mixtral.MixtralSparseMoeBlock
#   = SyncMixtralSparseMoeBlock
# globally. So the "MixtralSparseMoeBlock" name in the module now refers to
# Sync*. To patch the ORIGINAL HF class (which cpu_source's instances actually
# belong to), grab the class object directly from a known instance.

captures_off: dict = {}
captures_hf: dict = {}

# Real class objects of the live instances.
HFMixtralBlock = type(cpu_source.model.layers[0].block_sparse_moe)
SyncMixtralBlock = type(moe.model.model.layers[0].block_sparse_moe)
print(f"  HF block class:      {HFMixtralBlock.__module__}.{HFMixtralBlock.__name__}")
print(f"  Sync block class:    {SyncMixtralBlock.__module__}.{SyncMixtralBlock.__name__}")


def patched_off_forward(self, hidden_states):
    """Re-implements SyncMixtralSparseMoeBlock.forward but captures every
    intermediate so we can inspect what's actually being fed to dispatch_local."""
    bs, sl, hd = hidden_states.shape
    hs = hidden_states.view(-1, hd)
    router_logits = self.gate(hs)

    routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
    routing_weights, selected_experts = torch.topk(
        routing_weights, self.top_k, dim=-1)
    routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
    routing_weights = routing_weights.to(hs.dtype)

    router_mask_3d = F.one_hot(selected_experts, num_classes=self.num_experts)
    routing_weights_mask = (
        routing_weights[:, :, None] * router_mask_3d).permute(0, 2, 1)
    router_mask = router_mask_3d.permute(0, 2, 1)
    router_mask = torch.logical_or(router_mask[:, :, 0], router_mask[:, :, 1])
    routing_weights_mask = torch.sum(routing_weights_mask, dim=-1)

    if self.layer_id == LAYER:
        captures_off["gate_logits"] = router_logits.detach().float().cpu()
        captures_off["selected_experts"] = selected_experts.detach().cpu()
        captures_off["routing_weights_topk"] = routing_weights.detach().float().cpu()
        captures_off["router_mask"] = router_mask.detach().cpu()
        captures_off["routing_weights_mask"] = routing_weights_mask.detach().float().cpu()
        captures_off["hidden_states_in"] = hs.detach().float().cpu()

    self.expert_executor.dispatch_local(
        self.layer_id, hs, router_mask, routing_weights_mask)
    final = self.expert_executor.wait_dispatch_local()

    if self.layer_id == LAYER:
        captures_off["final_pre_view"] = final.detach().float().cpu()

    final = final.view(bs, sl, hd).to(hs.dtype)
    return final, router_logits


def patched_hf_forward(self, hidden_states):
    """HF stock MixtralSparseMoeBlock.forward — captures gate_logits +
    routing decisions for comparison."""
    bs, sl, hd = hidden_states.shape
    hs = hidden_states.view(-1, hd)
    router_logits = self.gate(hs)

    routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
    routing_weights, selected_experts = torch.topk(
        routing_weights, self.top_k, dim=-1)
    routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
    routing_weights = routing_weights.to(hs.dtype)

    # Stock HF expert loop
    final = torch.zeros((bs * sl, hd), dtype=hs.dtype, device=hs.device)
    expert_mask = F.one_hot(
        selected_experts, num_classes=self.num_experts).permute(2, 1, 0)
    for e in range(self.num_experts):
        idx, top_x = torch.where(expert_mask[e])
        if top_x.numel() == 0:
            continue
        current = hs[None, top_x].reshape(-1, hd)
        current_h = self.experts[e](current) * routing_weights[top_x, idx, None]
        final.index_add_(0, top_x, current_h.to(hs.dtype))

    if id(self) == id(cpu_source.model.layers[LAYER].block_sparse_moe):
        captures_hf["gate_logits"] = router_logits.detach().float().cpu()
        captures_hf["selected_experts"] = selected_experts.detach().cpu()
        captures_hf["routing_weights_topk"] = routing_weights.detach().float().cpu()
        captures_hf["hidden_states_in"] = hs.detach().float().cpu()
        captures_hf["final_pre_view"] = final.detach().float().cpu()

    return final.view(bs, sl, hd), router_logits


SyncMixtralBlock.forward = patched_off_forward
HFMixtralBlock.forward = patched_hf_forward


# =============================================================================
# Forward both
# =============================================================================

banner("Forward both models with same prompt")
prompt = "The capital of France is"
enc = tok(prompt, return_tensors="pt")
print(f"  prompt: {prompt!r}, ids: {enc.input_ids.tolist()}")

toks_gpu = enc.input_ids.to("cuda:0")
moe._configure_hook(toks_gpu)

print("\n  Running offload moe.model(toks) ...", flush=True)
t0 = time.perf_counter()
with torch.no_grad():
    out_off = moe.model(toks_gpu)
torch.cuda.synchronize()
print(f"    {time.perf_counter()-t0:.2f}s")

print("\n  Running cpu_source(toks) ...", flush=True)
t0 = time.perf_counter()
with torch.no_grad():
    out_hf = cpu_source(enc.input_ids)
print(f"    {time.perf_counter()-t0:.2f}s")


# =============================================================================
# Compare layer 0 intermediates  (wrapped in try/finally so we always
# os._exit even on KeyError or unexpected captures — moe_infinity's bg
# threads hang the SLURM job until time-limit otherwise.)
# =============================================================================

try:
    banner(f"Layer {LAYER} intermediates")

    print(f"  captures_off keys: {sorted(captures_off.keys())}")
    print(f"  captures_hf  keys: {sorted(captures_hf.keys())}")
    if not captures_off or not captures_hf:
        raise RuntimeError(
            "Layer-0 captures empty — the monkey-patched forwards never fired "
            "for the expected instance. Check the patched class identity.")

    # (1) Layer input hidden_states — should match (essentially bf16 noise)
    diff_in = (captures_off["hidden_states_in"] - captures_hf["hidden_states_in"]).abs()
    print(f"  (1) hidden_states_in:  max={diff_in.max().item():.4e}  mean={diff_in.mean().item():.4e}")

    # (2) gate_logits — same gate.weight, same input → should match
    diff_g = (captures_off["gate_logits"] - captures_hf["gate_logits"]).abs()
    print(f"  (2) gate_logits:       max={diff_g.max().item():.4e}  mean={diff_g.mean().item():.4e}")

    # (3) selected_experts — bool/int decision, should be IDENTICAL (no fp noise on argmax)
    se_off = captures_off["selected_experts"]
    se_hf = captures_hf["selected_experts"]
    agree = (se_off == se_hf).all().item()
    print(f"  (3) selected_experts identical: {agree}")
    if not agree:
        print(f"      offload: {se_off.tolist()}")
        print(f"      hf:      {se_hf.tolist()}")

    # (4) routing_weights (top-k normalized) — should match
    diff_rw = (captures_off["routing_weights_topk"] - captures_hf["routing_weights_topk"]).abs()
    print(f"  (4) routing_weights_topk: max={diff_rw.max().item():.4e}  mean={diff_rw.mean().item():.4e}")

    # (5) Reconstruct HF's equivalent (router_mask, routing_weights_mask) and check
    # that offload's captured masks match a hand-reconstructed version.
    N = se_hf.shape[0]
    E = 8
    top_k = 2
    hf_router_mask_reconstructed = torch.zeros((N, E), dtype=torch.bool)
    hf_rw_mask_reconstructed = torch.zeros((N, E), dtype=torch.float32)
    for n in range(N):
        for k in range(top_k):
            e = int(se_hf[n, k])
            hf_router_mask_reconstructed[n, e] = True
            hf_rw_mask_reconstructed[n, e] = captures_hf["routing_weights_topk"][n, k].item()

    mask_agree = (captures_off["router_mask"] == hf_router_mask_reconstructed).all().item()
    rw_diff = (captures_off["routing_weights_mask"] - hf_rw_mask_reconstructed).abs()
    print(f"  (5a) router_mask vs HF-reconstructed: identical={mask_agree}")
    print(f"  (5b) routing_weights_mask vs HF-reconstructed: "
          f"max={rw_diff.max().item():.4e}  mean={rw_diff.mean().item():.4e}")

    # (6) The KEY: dispatch_local output (offload) vs HF's expert loop output
    diff_final = (captures_off["final_pre_view"] - captures_hf["final_pre_view"]).abs()
    print(f"  (6) final MoE output:  max={diff_final.max().item():.4e}  mean={diff_final.mean().item():.4e}")

    banner("Verdict")
    python_routing_ok = (
        diff_in.max().item() < 1e-1
        and diff_g.max().item() < 1.0
        and agree
        and diff_rw.max().item() < 1e-1
        and mask_agree
        and rw_diff.max().item() < 1e-3
    )
    dispatch_diverges = diff_final.max().item() > 0.5

    if python_routing_ok and dispatch_diverges:
        print("  >>> Python routing produces IDENTICAL (mask, weights) to HF reconstruction.")
        print("      But dispatch_local OUTPUT differs by", f"{diff_final.max().item():.3f}.")
        print("      Bug is in C++ dispatch — ExpertDispatcher / fused_moe_ffn_into / OutputFunc.")
    elif not python_routing_ok:
        print("  >>> Python routing OUTPUT differs from HF expectation.")
        print("      Bug is in Python routing code.")
    elif not dispatch_diverges:
        print("  >>> Dispatch output matches HF — no bug at layer 0??")
    else:
        print("  >>> Mixed signals — review numbers above.")

except Exception:
    import traceback
    traceback.print_exc()
finally:
    import sys as _sys
    _sys.stdout.flush()
    _sys.stderr.flush()
    os._exit(0)
