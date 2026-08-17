"""Routing-trace capture (2026-07-27): per-token top-8 expert ids for RAG vs
translation, hf backend, greedy B=1. Evidence for the paper's dynamic-routing
claim — RAG's decode routing drifts from the prefill snapshot while
translation stays on a near-fixed expert set.

Output: output/routing_trace/{category}_{question_id}.npz with
  prefill_top8 [L, P, 8] int16  — per prompt token
  decode_top8  [L, T, 8] int16  — per generated token
Usage: python scripts/routing_trace.py [--qpc 15] [--mnt 512]
"""
import argparse
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from aug_spec.runtime.specbench import (_load_spec_bench_questions,
                                        _format_chat_prompt)

MODEL = "Qwen/Qwen3-30B-A3B-Base"
CATS = ("rag", "translation")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qpc", type=int, default=15)
    ap.add_argument("--mnt", type=int, default=512)
    ap.add_argument("--out", default="output/routing_trace")
    args = ap.parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_q = _load_spec_bench_questions(Path("data/spec_bench"))
    import random
    qs = []
    for cat in CATS:
        pool = [q for q in all_q if q["category"] == cat]
        random.Random(0).shuffle(pool)          # 與 runner 同 seed 慣例
        qs += pool[:args.qpc]
    print(f"questions: {len(qs)} ({', '.join(CATS)})", flush=True)

    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=torch.bfloat16, device_map="cuda")
    model.eval()

    layers = model.model.layers
    L = len(layers)
    capture = []                                # list per gate call
    hooks = []

    def mk_hook(li):
        def hook(_m, _inp, out):
            # out = router logits [tokens, E]
            top8 = out.detach().topk(8, dim=-1).indices.to(torch.int16).cpu()
            capture.append((li, top8))
        return hook

    for li, blk in enumerate(layers):
        hooks.append(blk.mlp.gate.register_forward_hook(mk_hook(li)))

    for qi, q in enumerate(qs):
        prompt = _format_chat_prompt(tok, [], q["turns"][0])
        ids = tok(prompt, return_tensors="pt").input_ids.to("cuda")
        P = ids.shape[1]
        capture.clear()
        with torch.no_grad():
            model.generate(ids, max_new_tokens=args.mnt, do_sample=False,
                           use_cache=True,
                           pad_token_id=tok.eos_token_id)
        # 依 token 數分辨 prefill(P tokens) 與 decode(1 token) 呼叫
        pre = {li: t for li, t in capture if t.shape[0] == P}
        dec = {}
        for li, t in capture:
            if t.shape[0] == P and li in pre and pre[li] is t:
                continue
            if t.shape[0] == 1:
                dec.setdefault(li, []).append(t)
        prefill = np.stack([pre[li].numpy() for li in range(L)])      # [L,P,8]
        T = min(len(v) for v in dec.values())
        decode = np.stack([torch.cat(dec[li][:T]).numpy()
                           for li in range(L)])                        # [L,T,8]
        np.savez_compressed(
            out_dir / f"{q['category']}_{q['question_id']}.npz",
            prefill_top8=prefill.astype(np.int16),
            decode_top8=decode.astype(np.int16))
        print(f"[{qi+1}/{len(qs)}] {q['category']} q{q['question_id']} "
              f"P={P} T={T}", flush=True)

    for h in hooks:
        h.remove()
    print("done", flush=True)


if __name__ == "__main__":
    main()
