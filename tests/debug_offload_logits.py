"""Compare logits between offload and HF for the SAME prompt.

If they differ significantly, target verify in offload mode is broken
(this is the only remaining hypothesis after Test A passed).

  - moe.model(toks)        — full model forward through moe_infinity's Sync block
  - cpu_source(toks.cpu()) — full model forward on CPU, reference HF behavior

Both should produce identical logits (mod bf16 noise) since:
  * same checkpoint weights
  * moe_infinity's Sync block and the stock transformers block are supposed
    to compute the same MoE: sum_e w_e * expert_e(x)

Usage:
  python -u tests/debug_offload_logits.py                       # default Mixtral
  python -u tests/debug_offload_logits.py --model qwen3         # Qwen3-30B-A3B
"""

from __future__ import annotations

import argparse
import os
import time

import torch

os.environ.setdefault("HF_HOME", "/work/morrisliu07/.cache/huggingface")

from moe_infinity import MoE
from transformers import AutoModelForCausalLM, AutoTokenizer


_PRESETS = {
    "mixtral": {
        "ckpt": "mistralai/Mixtral-8x7B-v0.1",
        "offload_dir": "/work/morrisliu07/aug_spec/cache/offload/mixtral",
    },
    "qwen3": {
        "ckpt": "Qwen/Qwen3-30B-A3B",
        "offload_dir": "/work/morrisliu07/aug_spec/cache/offload/qwen3_demo",
    },
}

_ap = argparse.ArgumentParser()
_ap.add_argument("--model", default="mixtral", choices=list(_PRESETS),
                 help="Which model preset to test")
_args = _ap.parse_args()
_p = _PRESETS[_args.model]
CKPT = _p["ckpt"]
OFFLOAD_DIR = _p["offload_dir"]


def banner(s):
    print("\n" + "=" * 70)
    print(f"  {s}")
    print("=" * 70, flush=True)


banner("Loading models")
print("  cpu_source ...", flush=True)
t0 = time.perf_counter()
cpu_source = AutoModelForCausalLM.from_pretrained(
    CKPT, torch_dtype=torch.bfloat16, device_map="cpu")
cpu_source.eval()
print(f"    {time.perf_counter()-t0:.1f}s")

print("  moe ...", flush=True)
t0 = time.perf_counter()
moe = MoE(CKPT, {
    "offload_path": OFFLOAD_DIR,
    "device_memory_ratio": 0.15,
})
print(f"    {time.perf_counter()-t0:.1f}s")

tok = AutoTokenizer.from_pretrained(CKPT)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token


# =============================================================================
# Compare logits for a short prompt
# =============================================================================

banner("Compare logits: short prompt (4 tokens)")

prompt = "The capital of France is"
enc = tok(prompt, return_tensors="pt")
print(f"  prompt: {prompt!r}")
print(f"  input_ids: {enc.input_ids.tolist()}  shape={tuple(enc.input_ids.shape)}")

# Offload forward.
print("\n  Running offload model forward ...", flush=True)
toks_gpu = enc.input_ids.to("cuda:0")
moe._configure_hook(toks_gpu)
t0 = time.perf_counter()
with torch.no_grad():
    out_off = moe.model(toks_gpu)
print(f"    {time.perf_counter()-t0:.2f}s, logits shape={tuple(out_off.logits.shape)}")
logits_off = out_off.logits[0, -1].float().cpu()  # last position, vocab dim
print(f"    last-position logits: top-5 ids = "
      f"{logits_off.topk(5).indices.tolist()}")
print(f"    last-position logits: top-5 vals = "
      f"{[f'{v:.3f}' for v in logits_off.topk(5).values.tolist()]}")
print(f"    argmax id = {logits_off.argmax().item()} "
      f"(token={tok.decode([logits_off.argmax().item()])!r})")

# CPU HF reference forward (slow but small input).
print("\n  Running cpu_source forward (this is slow — ~1-2 min on CPU) ...",
      flush=True)
toks_cpu = enc.input_ids
t0 = time.perf_counter()
with torch.no_grad():
    out_cpu = cpu_source(toks_cpu)
print(f"    {time.perf_counter()-t0:.1f}s, logits shape={tuple(out_cpu.logits.shape)}")
logits_cpu = out_cpu.logits[0, -1].float()
print(f"    last-position logits: top-5 ids = "
      f"{logits_cpu.topk(5).indices.tolist()}")
print(f"    last-position logits: top-5 vals = "
      f"{[f'{v:.3f}' for v in logits_cpu.topk(5).values.tolist()]}")
print(f"    argmax id = {logits_cpu.argmax().item()} "
      f"(token={tok.decode([logits_cpu.argmax().item()])!r})")

# Compare.
banner("Comparison")
diff = (logits_off - logits_cpu).abs()
print(f"  Logits diff:")
print(f"    max:  {diff.max().item():.4e}")
print(f"    mean: {diff.mean().item():.4e}")
print(f"    rel:  {(diff / logits_cpu.abs().clamp_min(1e-3)).mean().item():.4e}")

argmax_off = logits_off.argmax().item()
argmax_cpu = logits_cpu.argmax().item()
print(f"\n  Greedy argmax agreement: "
      f"offload={argmax_off} ({tok.decode([argmax_off])!r}) "
      f"vs cpu={argmax_cpu} ({tok.decode([argmax_cpu])!r})")
if argmax_off == argmax_cpu:
    print(f"  >>> ARGMAX MATCH — target verify is correct on offload.")
    print(f"      Bug is elsewhere (draft side / phase patch / etc).")
else:
    print(f"  >>> ARGMAX MISMATCH — target verify on offload is BROKEN.")
    print(f"      Same prompt + same model produces different greedy tokens.")

# Bonus: check overlap in top-K.
for k in (1, 5, 10):
    off_top = set(logits_off.topk(k).indices.tolist())
    cpu_top = set(logits_cpu.topk(k).indices.tolist())
    overlap = len(off_top & cpu_top)
    print(f"  top-{k} overlap: {overlap}/{k}")

# moe_infinity background threads hang Python exit — force quit.
import sys as _sys
_sys.stdout.flush()
_sys.stderr.flush()
os._exit(0)
