"""Compare offload vs HF Mixtral LAYER-BY-LAYER to find where divergence starts.

Both models run the same input_ids end-to-end (so moe_infinity's pre-forward
hooks materialize dense params normally). We attach forward hooks on each
layer's block_sparse_moe to capture (input, output) tensors. After both runs,
compare per-layer: where does the offload path diverge from the HF reference?

If layer 0 already shows large output diff but small input diff → the bug
is in layer 0's MoE block dispatch (i.e., inside SyncBlock.forward).
If divergence only appears at deeper layers → bug accumulates.
If input diff is also large at layer 0 → embedding / pre-MoE issue (less likely).

Run:
  python -u tests/debug_layer_diverge_mixtral.py
"""

from __future__ import annotations

import os
import time

import torch

os.environ.setdefault("HF_HOME", "/work/morrisliu07/.cache/huggingface")

from moe_infinity import MoE
from transformers import AutoModelForCausalLM, AutoTokenizer


CKPT = "mistralai/Mixtral-8x7B-v0.1"
OFFLOAD_DIR = "/work/morrisliu07/aug_spec/cache/offload/mixtral"


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
# Register forward hooks on every block_sparse_moe
# =============================================================================

captures_off: dict[int, tuple] = {}
captures_hf: dict[int, tuple] = {}


def make_hook(store: dict, layer_idx: int):
    def hook(module, inp, out):
        # inp is a tuple — first arg is hidden_states
        # out is either Tensor or tuple (Tensor, router_logits)
        x = inp[0].detach().float().cpu()
        if isinstance(out, tuple):
            y = out[0].detach().float().cpu()
        else:
            y = out.detach().float().cpu()
        store[layer_idx] = (x, y)
    return hook


for i, layer in enumerate(moe.model.model.layers):
    layer.block_sparse_moe.register_forward_hook(make_hook(captures_off, i))
for i, layer in enumerate(cpu_source.model.layers):
    layer.block_sparse_moe.register_forward_hook(make_hook(captures_hf, i))

print(f"\n  hooks registered on {len(moe.model.model.layers)} layers each")


# =============================================================================
# Prompt + forward both
# =============================================================================

banner("Forward both models with the same prompt")

prompt = "The capital of France is"
enc = tok(prompt, return_tensors="pt")
print(f"  prompt: {prompt!r}")
print(f"  input_ids: {enc.input_ids.tolist()}  shape={tuple(enc.input_ids.shape)}")

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
# Compare layer-by-layer
# =============================================================================

banner("Per-layer block_sparse_moe (input, output) divergence")

print(f"  {'layer':>6}  {'in_max_diff':>12}  {'out_max_diff':>13}  "
      f"{'in_mean_diff':>13}  {'out_mean_diff':>13}")
print(f"  {'-'*6}  {'-'*12}  {'-'*13}  {'-'*13}  {'-'*13}")

first_block_with_large_jump = None
for i in sorted(captures_off):
    x_off, y_off = captures_off[i]
    if i not in captures_hf:
        continue
    x_hf, y_hf = captures_hf[i]

    in_diff_max = (x_off - x_hf).abs().max().item()
    in_diff_mean = (x_off - x_hf).abs().mean().item()
    out_diff_max = (y_off - y_hf).abs().max().item()
    out_diff_mean = (y_off - y_hf).abs().mean().item()

    print(f"  {i:>6}  {in_diff_max:>12.4e}  {out_diff_max:>13.4e}  "
          f"{in_diff_mean:>13.4e}  {out_diff_mean:>13.4e}")

    # Detect the FIRST layer where out_diff jumps ahead of in_diff significantly.
    if first_block_with_large_jump is None and out_diff_max > 2.0 * max(in_diff_max, 0.1):
        first_block_with_large_jump = i


banner("Interpretation")
print(f"  First layer where out_diff >> in_diff (MoE block introduces error):")
print(f"    layer {first_block_with_large_jump}")
print()

# Look at final logits too.
logits_off = out_off.logits[0, -1].float().cpu()
logits_hf = out_hf.logits[0, -1].float()
print(f"  Final last-token argmax: offload={logits_off.argmax().item()} "
      f"({tok.decode([logits_off.argmax().item()])!r})")
print(f"                            cpu    ={logits_hf.argmax().item()} "
      f"({tok.decode([logits_hf.argmax().item()])!r})")

# Force exit to dodge ArcherTaskPool hang.
import sys as _sys
_sys.stdout.flush()
_sys.stderr.flush()
os._exit(0)
