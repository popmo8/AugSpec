"""Single-block side-by-side: offload SyncMixtralSparseMoeBlock vs HF stock
MixtralSparseMoeBlock, on the same hidden_states input with identical weights.

If the outputs differ significantly → bug is somewhere inside SyncBlock.forward's
path (Python routing → dispatch_local → C++ ExpertDispatcher → fused GEMM).

This isolates the moe_infinity dispatch from spec-decoding / our aug_spec code:
both runs only invoke ONE MoE block forward, no controller patching, no
generate loop.

Three increasingly granular checks:
  1. Sanity: gate.weight bit-identical between offload and cpu_source.
  2. Sanity: pick one expert (e=0), compare its w1/w2/w3 weights — placeholder
     on offload (shape-(1,) zeros) vs real on cpu_source.
  3. Forward comparison: run the same hidden_states through both blocks; compare
     final outputs.

Run:
  python -u tests/debug_block_compare_mixtral.py
"""

from __future__ import annotations

import os
import time

import torch

os.environ.setdefault("HF_HOME", "/work/morrisliu07/.cache/huggingface")

from moe_infinity import MoE
from transformers import AutoModelForCausalLM


CKPT = "mistralai/Mixtral-8x7B-v0.1"
OFFLOAD_DIR = "/work/morrisliu07/aug_spec/cache/offload/mixtral"
LAYER_IDX = 0
NUM_TOKENS = 4


def banner(s):
    print("\n" + "=" * 70)
    print(f"  {s}")
    print("=" * 70, flush=True)


def stats(name, t):
    f = t.detach().float().cpu()
    print(f"  {name}: shape={tuple(t.shape)} dtype={t.dtype} device={t.device} "
          f"mean={f.mean().item():+.4e} max={f.max().item():+.4e} "
          f"min={f.min().item():+.4e} nan={int(torch.isnan(f).sum().item())} "
          f"|w|={f.abs().sum().item():.2e}")


# =============================================================================
# Load both models — cpu_source FIRST so moe_infinity hooks don't touch it
# =============================================================================

banner("Loading cpu_source (HF Mixtral on CPU)")
t0 = time.perf_counter()
cpu_source = AutoModelForCausalLM.from_pretrained(
    CKPT, torch_dtype=torch.bfloat16, device_map="cpu")
cpu_source.eval()
print(f"  loaded in {time.perf_counter()-t0:.1f}s")

banner("Loading moe wrapper (offloaded Mixtral)")
t0 = time.perf_counter()
moe = MoE(CKPT, {
    "offload_path": OFFLOAD_DIR,
    "device_memory_ratio": 0.15,
})
print(f"  loaded in {time.perf_counter()-t0:.1f}s")

moe_block = moe.model.model.layers[LAYER_IDX].block_sparse_moe
cpu_block = cpu_source.model.layers[LAYER_IDX].block_sparse_moe
print(f"  offload block type: {type(moe_block).__name__}")
print(f"  cpu     block type: {type(cpu_block).__name__}")
print(f"  num_experts: {len(moe_block.experts)}")
print(f"  top_k:       {moe_block.top_k}")


# =============================================================================
# Check 1 — gate weights bit-identical
# =============================================================================

banner("Check 1: gate.weight identity")

moe_gate = moe_block.gate.weight
hf_gate = cpu_block.gate.weight
stats("offload gate.weight", moe_gate)
stats("cpu_source gate.weight", hf_gate)

# Pull both to CPU for comparison.
if moe_gate.shape == hf_gate.shape:
    diff = (moe_gate.float().cpu() - hf_gate.float().cpu()).abs()
    print(f"\n  Gate diff: max={diff.max().item():.4e} mean={diff.mean().item():.4e}")
    if diff.max().item() < 1e-3:
        print("  >>> Gate weights MATCH (sanity).")
    else:
        print(f"  >>> GATE WEIGHTS DIFFER! This is suspicious.")
else:
    print(f"  >>> Gate weights have DIFFERENT shapes — placeholder?")


# =============================================================================
# Check 2 — expert.w1 placeholder vs real
# =============================================================================

banner("Check 2: expert[0].w1 weight identity")

moe_e0_w1 = moe_block.experts[0].w1.weight
hf_e0_w1 = cpu_block.experts[0].w1.weight
stats("offload experts[0].w1", moe_e0_w1)
stats("cpu_source experts[0].w1", hf_e0_w1)
print()
if list(moe_e0_w1.shape) == [1]:
    print("  >>> offload experts[0].w1 IS placeholder shape-(1,) — as expected.")
else:
    print("  >>> offload experts[0].w1 has unexpected shape; review.")


# =============================================================================
# Check 3 — Single-block forward comparison (THE KEY TEST)
# =============================================================================

banner("Check 3: SyncBlock vs HF block forward on same input")

torch.manual_seed(0)
hs = torch.randn(
    1, NUM_TOKENS, moe_block.hidden_dim,
    dtype=torch.bfloat16, device="cuda:0")
stats("hidden_states (random)", hs)

# Offload path — set up moe wrapper's per-sequence state first.
print("\n  Running offload SyncMixtralSparseMoeBlock.forward ...", flush=True)
moe._configure_hook(torch.zeros((1, NUM_TOKENS), dtype=torch.long, device="cuda:0"))
t0 = time.perf_counter()
with torch.no_grad():
    out_off, _logits_off = moe_block(hs)
torch.cuda.synchronize()
print(f"    {time.perf_counter()-t0:.2f}s")
stats("offload output", out_off)

# HF stock path — same input on CPU (cpu_block weights are CPU).
print("\n  Running HF MixtralSparseMoeBlock.forward on CPU ...", flush=True)
hs_cpu = hs.cpu()
t0 = time.perf_counter()
with torch.no_grad():
    out_hf, _logits_hf = cpu_block(hs_cpu)
print(f"    {time.perf_counter()-t0:.2f}s")
stats("HF output", out_hf)

# Bring offload to CPU for comparison.
out_off_cpu = out_off.cpu()
diff = (out_off_cpu.float() - out_hf.float()).abs()
print(f"\n  Output diff: max={diff.max().item():.4e}  mean={diff.mean().item():.4e}")

# Also compute relative.
ref_mag = out_hf.float().abs().clamp_min(1e-3)
rel = (diff / ref_mag).mean().item()
print(f"  Relative diff: {rel:.4e}")

# Logits comparison too.
print(f"\n  Router logits diff:")
diff_l = (_logits_off.float().cpu() - _logits_hf.float()).abs()
print(f"    max={diff_l.max().item():.4e}  mean={diff_l.mean().item():.4e}")

# Verdict.
print()
if diff.max().item() < 0.5:
    print("  >>> SINGLE-BLOCK MATCH (mod bf16 noise).")
    print("      → SyncBlock.forward path is OK.")
    print("      → Bug must be cumulative (32 layers of small errors compound),")
    print("        OR triggered by spec-decoding's specific call pattern.")
else:
    print("  >>> SINGLE-BLOCK DIVERGES (max diff >= 0.5).")
    print("      → Bug is INSIDE the offload SyncMixtralSparseMoeBlock.forward path.")
    print("      → Either Python routing (one_hot + permute + logical_or + sum)")
    print("        OR dispatch_local (Python wrapper + C++ ExpertDispatcher + fused GEMM)")
    print("        OR a stream-sync race in expert fetch/exec is producing wrong output.")


# Force exit to dodge the ArcherTaskPool hang.
import sys as _sys
_sys.stdout.flush()
_sys.stderr.flush()
os._exit(0)
