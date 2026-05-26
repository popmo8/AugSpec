"""Just run offload Mixtral generation on a few prompts and look at the
output text. If it's coherent → moe_infinity Mixtral path works (the
AccRate=0.05 we see in spec-decoding must come from somewhere else).
If it's gibberish → moe_infinity Mixtral path is broken at the model level.

This bypasses all spec-decoding / aug_spec code — pure
`moe.model.generate(input_ids)`.
"""

from __future__ import annotations

import os
import time

import torch

os.environ.setdefault("HF_HOME", "/work/morrisliu07/.cache/huggingface")

from moe_infinity import MoE
from transformers import AutoTokenizer


CKPT = "mistralai/Mixtral-8x7B-v0.1"
OFFLOAD_DIR = "/work/morrisliu07/aug_spec/cache/offload/mixtral"

PROMPTS = [
    "The capital of France is",
    "Once upon a time, there was a",
    "Q: What is 2 + 2?\nA:",
    "translate English to German: How old are you? - Answer:",
]


def banner(s):
    print("\n" + "=" * 70)
    print(f"  {s}")
    print("=" * 70, flush=True)


try:
    banner("Load offload Mixtral")
    t0 = time.perf_counter()
    moe = MoE(CKPT, {"offload_path": OFFLOAD_DIR, "device_memory_ratio": 0.15})
    print(f"  loaded in {time.perf_counter()-t0:.1f}s")

    tok = AutoTokenizer.from_pretrained(CKPT)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    for i, prompt in enumerate(PROMPTS, 1):
        banner(f"Prompt {i}: {prompt!r}")
        enc = tok(prompt, return_tensors="pt")
        toks_gpu = enc.input_ids.to("cuda:0")
        moe._configure_hook(toks_gpu)
        print(f"  input_ids: {enc.input_ids.tolist()}", flush=True)
        print(f"  input length: {enc.input_ids.shape[1]} tokens")

        t0 = time.perf_counter()
        with torch.no_grad():
            out = moe.model.generate(
                toks_gpu,
                max_new_tokens=32,
                do_sample=False,
                pad_token_id=tok.eos_token_id,
            )
        torch.cuda.synchronize()
        wall = time.perf_counter() - t0

        new_tokens = out[0, enc.input_ids.shape[1]:].tolist()
        new_text = tok.decode(new_tokens, skip_special_tokens=False)
        print(f"  wall: {wall:.1f}s")
        print(f"  new tokens (ids): {new_tokens}")
        print(f"  new tokens (text):")
        print(f"  >>> {new_text!r}")
        print(f"  full text:")
        print(f"  >>> {tok.decode(out[0], skip_special_tokens=False)!r}")

finally:
    import sys as _sys
    _sys.stdout.flush()
    _sys.stderr.flush()
    os._exit(0)
