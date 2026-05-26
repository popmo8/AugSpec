"""Tiny: moe_infinity offload Mixtral, 'The capital of France is' → 8 toks."""

from __future__ import annotations
import os
import time

os.environ.setdefault("HF_HOME", "/work/morrisliu07/.cache/huggingface")

import torch
print(f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}", flush=True)
print(f"torch sees {torch.cuda.device_count()} GPUs", flush=True)
for i in range(torch.cuda.device_count()):
    free, tot = torch.cuda.mem_get_info(i)
    print(f"  cuda:{i}: free {free/1e9:.0f}GB / {tot/1e9:.0f}GB", flush=True)

from moe_infinity import MoE
from transformers import AutoTokenizer

CKPT = "mistralai/Mixtral-8x7B-v0.1"
OFFLOAD_DIR = "/work/morrisliu07/aug_spec/cache/offload/mixtral"

t0 = time.perf_counter()
moe = MoE(CKPT, {
    "offload_path": OFFLOAD_DIR,
    "device_memory_ratio": 0.15,
})
print(f"\nloaded in {time.perf_counter()-t0:.0f}s", flush=True)

tok = AutoTokenizer.from_pretrained(CKPT)
prompt = "The capital of France is"
enc = tok(prompt, return_tensors="pt")
toks = enc.input_ids.to("cuda:0")
print(f"\nprompt: {prompt!r}")
print(f"input_ids: {enc.input_ids.tolist()}")

moe._configure_hook(toks)

t0 = time.perf_counter()
with torch.no_grad():
    out = moe.model.generate(
        input_ids=toks,
        max_new_tokens=8,
        do_sample=False,
        pad_token_id=tok.eos_token_id,
    )
torch.cuda.synchronize()
print(f"\ngenerate wall: {time.perf_counter()-t0:.1f}s")

new_ids = out[0, enc.input_ids.shape[1]:].tolist()
new_text = tok.decode(new_ids, skip_special_tokens=False)
full_text = tok.decode(out[0].tolist(), skip_special_tokens=False)
print(f"new ids:   {new_ids}")
print(f"new text:  {new_text!r}")
print(f"full text: {full_text!r}")

import sys
sys.stdout.flush()
os._exit(0)
