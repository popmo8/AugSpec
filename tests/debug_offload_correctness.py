"""Localize the offload-backend AccRate=0 bug.

Splits the offload forward into 3 testable hypotheses and prints which
one (if any) fails.

  Test A — build_weighted_avg_offload correctness:
    Build merged expert via the new offload path (reads from cpu_source).
    Build the same merged expert manually from cpu_source via a direct
    Python loop. Compare numerically. If they differ → the offload merge
    is buggy.

  Test B — _route_offload vs _route_hf output equivalence:
    Run _route_offload on the offloaded SyncMixtralSparseMoeBlock with
    a random hidden-state input. Run _route_hf on cpu_source's
    MixtralSparseMoeBlock with the SAME input (on CPU). Compare. If
    they differ much → the offload target verify is buggy.

  Test C — _run_dense_expert sanity:
    Run draft-phase forward through the offload-built merged expert.
    Print stats of the output. NaN/inf → numerical issue.

Run from repo root inside the venv:
    python -u tests/debug_offload_correctness.py
"""

from __future__ import annotations

import os
import time

import torch

os.environ.setdefault("HF_HOME", "/work/morrisliu07/.cache/huggingface")

from moe_infinity import MoE
from transformers import AutoModelForCausalLM

from aug_spec.adapters.mixtral import MixtralAdapter, _is_offload_block


CKPT = "mistralai/Mixtral-8x7B-v0.1"
OFFLOAD_DIR = "/work/morrisliu07/aug_spec/cache/offload/mixtral"
LAYER_IDX = 0
NUM_TOKENS = 4  # keep small so CPU-side _route_hf is fast
HIDDEN = 4096
NUM_EXPERTS = 8


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
# Load both models (CPU source FIRST, then MoE wrapper — same order as load_offload)
# =============================================================================

banner("Loading cpu_source (HF Mixtral on CPU)")
t0 = time.perf_counter()
cpu_source = AutoModelForCausalLM.from_pretrained(
    CKPT, torch_dtype=torch.bfloat16, device_map="cpu")
cpu_source.eval()
for p in cpu_source.parameters():
    p.requires_grad_(False)
print(f"  loaded in {time.perf_counter()-t0:.1f}s")

banner("Loading moe wrapper (offloaded)")
t0 = time.perf_counter()
moe = MoE(CKPT, {
    "offload_path": OFFLOAD_DIR,
    "device_memory_ratio": 0.15,
})
print(f"  loaded in {time.perf_counter()-t0:.1f}s")

# Resolve corresponding MoE blocks.
off_block = moe.model.model.layers[LAYER_IDX].block_sparse_moe
cpu_block = cpu_source.model.layers[LAYER_IDX].block_sparse_moe
print(f"  off_block type: {type(off_block).__name__} (offload={_is_offload_block(off_block)})")
print(f"  cpu_block type: {type(cpu_block).__name__}")

adapter = MixtralAdapter()


# =============================================================================
# Test A — build_weighted_avg correctness
# =============================================================================

banner("Test A — build_weighted_avg_offload vs manual reference")

weights = [1.0 / NUM_EXPERTS] * NUM_EXPERTS
merged_offload = adapter.build_weighted_avg(
    off_block, weights, cpu_block=cpu_block)

print("\n  offload-built merged expert (read CPU, accumulate on GPU):")
for k in ("w1", "w2", "w3"):
    stats(f"merged_offload[{k}]", merged_offload[k])

# Manual reference: same source (cpu_source), same math, but explicit and on CPU.
print("\n  Manual reference (CPU-only):")
ref_w1 = cpu_block.experts[0].w1.weight
ref_w2 = cpu_block.experts[0].w2.weight
ref_w3 = cpu_block.experts[0].w3.weight
ref = {
    "w1": torch.zeros(ref_w1.shape, dtype=torch.float32),
    "w2": torch.zeros(ref_w2.shape, dtype=torch.float32),
    "w3": torch.zeros(ref_w3.shape, dtype=torch.float32),
}
for e in range(NUM_EXPERTS):
    ref["w1"].add_(cpu_block.experts[e].w1.weight.float(), alpha=weights[e])
    ref["w2"].add_(cpu_block.experts[e].w2.weight.float(), alpha=weights[e])
    ref["w3"].add_(cpu_block.experts[e].w3.weight.float(), alpha=weights[e])
ref = {k: v.to(torch.bfloat16) for k, v in ref.items()}
for k in ("w1", "w2", "w3"):
    stats(f"ref[{k}]", ref[k])

# Compare.
print("\n  Diff (offload merged vs manual ref):")
for k in ("w1", "w2", "w3"):
    diff = (merged_offload[k].cpu().float() - ref[k].float()).abs()
    print(f"    {k}: max={diff.max().item():.6e}  mean={diff.mean().item():.6e}")

# Verdict.
max_diff_a = max(
    (merged_offload[k].cpu().float() - ref[k].float()).abs().max().item()
    for k in ("w1", "w2", "w3"))
verdict_a = "PASS" if max_diff_a < 1e-2 else "FAIL"
print(f"\n  >>> Test A: {verdict_a} (max_diff={max_diff_a:.4e}; "
      "<1e-2 = identical up to bf16 noise)")


# =============================================================================
# Test B — _route_offload vs _route_hf produce same output
# =============================================================================

banner("Test B — target verify: _route_offload vs _route_hf")

torch.manual_seed(0)
hs_flat_gpu = torch.randn(NUM_TOKENS, HIDDEN, dtype=torch.bfloat16, device="cuda:0")

# Run offload routing — needs _configure_hook first so archer tracer is set.
print(f"\n  Running _route_offload on cuda:0 ...")
moe._configure_hook(torch.zeros((1, NUM_TOKENS), dtype=torch.long, device="cuda:0"))
gate_logits_gpu = off_block.gate(hs_flat_gpu)
stats("gate_logits_gpu (off_block.gate)", gate_logits_gpu)

t0 = time.perf_counter()
with torch.no_grad():
    out_offload = adapter._route_offload(
        off_block, hs_flat_gpu, gate_logits_gpu,
        batch_size=1, sequence_length=NUM_TOKENS, hidden_dim=HIDDEN,
    )
print(f"    _route_offload took {time.perf_counter()-t0:.2f}s")
stats("out_offload", out_offload)

# Run HF routing on the cpu_block — input must be on CPU.
print(f"\n  Running _route_hf on cpu (cpu_block) ...")
hs_flat_cpu = hs_flat_gpu.cpu()
gate_logits_cpu = cpu_block.gate(hs_flat_cpu)
stats("gate_logits_cpu (cpu_block.gate)", gate_logits_cpu)
print(f"    Gate-logits cross-device diff: max="
      f"{(gate_logits_gpu.cpu() - gate_logits_cpu).abs().max().item():.4e}")

t0 = time.perf_counter()
with torch.no_grad():
    out_hf = adapter._route_hf(
        cpu_block, hs_flat_cpu, gate_logits_cpu,
        batch_size=1, sequence_length=NUM_TOKENS, hidden_dim=HIDDEN,
    )
print(f"    _route_hf took {time.perf_counter()-t0:.2f}s")
stats("out_hf", out_hf)

# Compare (both moved to CPU).
diff_b = (out_offload.cpu().float() - out_hf.float()).abs()
print(f"\n  out_offload vs out_hf diff: max={diff_b.max().item():.4e} "
      f"mean={diff_b.mean().item():.4e}")

verdict_b = "PASS" if diff_b.max().item() < 5e-1 else "FAIL"
print(f"  >>> Test B: {verdict_b} (target verify outputs "
      "{}identical".format("≈" if verdict_b == "PASS" else "NOT "))


# =============================================================================
# Test C — _run_dense_expert with merged_offload, output sanity
# =============================================================================

banner("Test C — _run_dense_expert(merged_offload, hs_flat_gpu) sanity")

with torch.no_grad():
    out_draft = adapter._run_dense_expert(merged_offload, hs_flat_gpu)
stats("_run_dense_expert output", out_draft)
print(f"\n  >>> Test C: {'PASS' if not torch.isnan(out_draft).any() else 'FAIL'} "
      "(NaN check)")


# =============================================================================
# Summary
# =============================================================================

banner("SUMMARY")
print(f"  A (build_weighted_avg): {verdict_a}  max_diff={max_diff_a:.4e}")
print(f"  B (target verify):      {verdict_b}  max_diff={diff_b.max().item():.4e}")
print(f"  C (run dense expert):   {'PASS' if not torch.isnan(out_draft).any() else 'FAIL'}")

# moe_infinity background threads hang Python exit — force quit.
import sys as _sys
_sys.stdout.flush()
_sys.stderr.flush()
os._exit(0)
