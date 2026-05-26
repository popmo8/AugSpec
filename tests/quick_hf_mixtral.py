"""Tiny: HF Mixtral on GPU, one prompt, 8 new tokens. No moe_infinity."""

from __future__ import annotations
import os
import time

os.environ.setdefault("HF_HOME", "/work/morrisliu07/.cache/huggingface")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

CKPT = "mistralai/Mixtral-8x7B-v0.1"

print(f"torch: {torch.__version__}  cuda: {torch.version.cuda}", flush=True)
print(f"GPUs: {torch.cuda.device_count()}", flush=True)
for i in range(torch.cuda.device_count()):
    free, tot = torch.cuda.mem_get_info(i)
    print(f"  cuda:{i}: free {free/1e9:.0f}GB / {tot/1e9:.0f}GB", flush=True)

t0 = time.perf_counter()
model = AutoModelForCausalLM.from_pretrained(
    CKPT,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
model.eval()
print(f"\nloaded in {time.perf_counter()-t0:.0f}s", flush=True)
print(f"device_map: {dict(model.hf_device_map) if hasattr(model, 'hf_device_map') else 'N/A'}", flush=True)

tok = AutoTokenizer.from_pretrained(CKPT)

prompt = "The capital of France is"
enc = tok(prompt, return_tensors="pt").to(model.device)
print(f"\nprompt: {prompt!r}")
print(f"input_ids: {enc.input_ids.tolist()}")

t0 = time.perf_counter()
with torch.no_grad():
    out = model.generate(
        **enc,
        max_new_tokens=8,
        do_sample=False,
        pad_token_id=tok.eos_token_id,
    )
torch.cuda.synchronize()
print(f"\ngenerate wall: {time.perf_counter()-t0:.1f}s")

new_ids = out[0, enc.input_ids.shape[1]:].tolist()
new_text = tok.decode(new_ids, skip_special_tokens=False)
full_text = tok.decode(out[0].tolist(), skip_special_tokens=False)
print(f"new ids:    {new_ids}")
print(f"new text:   {new_text!r}")
print(f"full text:  {full_text!r}")

import sys
sys.stdout.flush()
os._exit(0)
