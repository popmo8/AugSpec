"""Sanity: does HF Mixtral on cuda:0 produce coherent English by itself?

NO moe_infinity, NO offload — just transformers' Mixtral on a single H200.
If THIS is broken, the per-layer offload-vs-HF comparison is meaningless
because the "reference" is bad.

Test: generate continuation of "The capital of France is" (and 2 others).
A working Mixtral should produce " Paris" or similar.
"""

from __future__ import annotations

import os
import time

os.environ.setdefault("HF_HOME", "/work/morrisliu07/.cache/huggingface")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


CKPT = "mistralai/Mixtral-8x7B-v0.1"


def banner(s):
    print("\n" + "=" * 70)
    print(f"  {s}")
    print("=" * 70, flush=True)


try:
    banner("Step 1/3: load HF Mixtral on cuda:0 (no moe_infinity)")
    print(f"  free VRAM before: {torch.cuda.mem_get_info(0)[0] / 1e9:.1f} GB",
          flush=True)
    t0 = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(
        CKPT,
        torch_dtype=torch.bfloat16,
        device_map="cuda:0",
    )
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    print(f"  loaded in {time.perf_counter()-t0:.1f}s")
    print(f"  free VRAM after: {torch.cuda.mem_get_info(0)[0] / 1e9:.1f} GB",
          flush=True)

    tok = AutoTokenizer.from_pretrained(CKPT)

    banner("Step 2/3: greedy generate, NO offload anywhere")
    prompts = [
        "The capital of France is",
        "Once upon a time, there was a",
        "Q: What is 2 + 2?\nA:",
        "translate English to German: How old are you? - Answer:",
    ]

    for i, prompt in enumerate(prompts, 1):
        print(f"\n  --- Prompt {i}: {prompt!r}")
        enc = tok(prompt, return_tensors="pt").to("cuda:0")
        print(f"    input_ids ({len(enc.input_ids[0])} toks):"
              f" {enc.input_ids.tolist()}")
        t0 = time.perf_counter()
        with torch.no_grad():
            out = model.generate(
                **enc,
                max_new_tokens=20,
                do_sample=False,
                temperature=1.0,
                pad_token_id=tok.eos_token_id,
            )
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        new_ids = out[0, enc.input_ids.shape[1]:].tolist()
        new_text = tok.decode(new_ids, skip_special_tokens=False)
        full_text = tok.decode(out[0].tolist(), skip_special_tokens=False)
        print(f"    wall: {elapsed:.1f}s, new ids: {new_ids}")
        print(f"    new tokens: {new_text!r}")
        print(f"    full: {full_text!r}")

    banner("Step 3/3: also check single-forward argmax")
    prompt = "The capital of France is"
    enc = tok(prompt, return_tensors="pt").to("cuda:0")
    with torch.no_grad():
        logits = model(**enc).logits
    last = logits[0, -1].float().cpu()
    top5 = last.topk(5)
    print(f"  prompt: {prompt!r}")
    print(f"  top-5 ids:    {top5.indices.tolist()}")
    print(f"  top-5 tokens: {[tok.decode([i]) for i in top5.indices.tolist()]}")
    print(f"  argmax id:    {last.argmax().item()}")
    print(f"  argmax token: {tok.decode([last.argmax().item()])!r}")

    banner("VERDICT")
    print("  If 'Paris' or ' Paris' appears in any of the above, HF Mixtral")
    print("  on cuda:0 is HEALTHY and the offload-vs-HF comparison is valid.")
    print("  If all 4 prompts produce gibberish, the bug is NOT in moe_infinity")
    print("  alone — something about our environment breaks HF Mixtral too.")

except Exception:
    import traceback
    traceback.print_exc()
finally:
    import sys as _sys
    _sys.stdout.flush()
    _sys.stderr.flush()
    os._exit(0)
