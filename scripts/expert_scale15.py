"""dist/hidden merge-perturbation scale, qpc=15 per Spec-Bench category
(~195 prompts), vectorized. Reports the four paper numbers per model:
hidden_norm, out_norm, pair_dist, dist/hidden.
Usage: python scripts/expert_scale15.py --model {qwen3,mixtral,deepseek,gptoss}
"""
import argparse, random
from pathlib import Path

import numpy as np
import torch

import sys
sys.path.insert(0, "scripts")
from expert_sim_l2 import MODELS, moe_blocks, router_logits, expert_out
from aug_spec.runtime.specbench import _load_spec_bench_questions
from transformers import AutoModelForCausalLM, AutoTokenizer

QPC = 15
MAX_LEN = 512


def calib_prompts():
    qs = _load_spec_bench_questions(Path("data/spec_bench"))
    by_cat = {}
    for q in qs:
        by_cat.setdefault(q["category"], []).append(q)
    out = []
    for cat in sorted(by_cat):
        pool = by_cat[cat]
        random.Random(0).shuffle(pool)
        out += [q["turns"][0] for q in pool[:QPC]]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(MODELS))
    args = ap.parse_args()
    mid, topk = MODELS[args.model]

    tok = AutoTokenizer.from_pretrained(mid, trust_remote_code=True)
    kw = dict(torch_dtype=torch.bfloat16, device_map="cuda", trust_remote_code=True)
    if args.model == "gptoss":
        try:
            from transformers import Mxfp4Config
            kw["quantization_config"] = Mxfp4Config(dequantize=True)
        except ImportError:
            pass
    model = AutoModelForCausalLM.from_pretrained(mid, **kw)
    model.eval()
    blocks = moe_blocks(model)
    prompts = calib_prompts()
    print(f"{args.model}: {len(blocks)} blocks, {len(prompts)} prompts", flush=True)

    cap = {}
    hooks = []
    for li, blk, _ in blocks:
        def mk(li):
            def pre(_m, inp):
                cap[li] = inp[0].detach().reshape(-1, inp[0].shape[-1])
            return pre
        hooks.append(blk.register_forward_pre_hook(mk(li)))

    h_sum = h_n = o_sum = o_n = d_sum = d_n = 0.0
    for pi, p in enumerate(prompts):
        ids = tok(p, return_tensors="pt", truncation=True,
                  max_length=MAX_LEN).input_ids.to(model.device)
        with torch.no_grad():
            model(ids, use_cache=False)
        for li, blk, ex in blocks:
            h = cap[li]
            T = h.shape[0]
            h_sum += float(h.float().norm(dim=1).sum()); h_n += T
            logits = router_logits(blk, h)
            E = logits.shape[1]
            top = logits.topk(topk, dim=-1).indices          # [T,k]
            buf = torch.zeros(T, topk, h.shape[1],
                              device=h.device, dtype=torch.float32)
            with torch.no_grad():
                for e in range(E):
                    pos = (top == e).nonzero(as_tuple=False)  # [m,2] (t,slot)
                    if pos.numel() == 0:
                        continue
                    o = expert_out(ex, h[pos[:, 0]], e).float()
                    buf[pos[:, 0], pos[:, 1]] = o
            o_sum += float(buf.norm(dim=-1).sum()); o_n += T * topk
            D = torch.cdist(buf, buf)                        # [T,k,k]
            iu = torch.triu_indices(topk, topk, offset=1)
            d = D[:, iu[0], iu[1]]                           # [T, k(k-1)/2]
            d_sum += float(d.sum()); d_n += d.numel()
        if (pi + 1) % 20 == 0:
            print(f"  prompt {pi+1}/{len(prompts)}", flush=True)
    for hk in hooks:
        hk.remove()

    hn, on, pd = h_sum / h_n, o_sum / o_n, d_sum / d_n
    print(f"RESULT {args.model}: hidden_norm={hn:.2f} out_norm={on:.2f} "
          f"pair_dist={pd:.2f} | out/hidden={on/hn:.3f} "
          f"dist/hidden={pd/hn:.3f}", flush=True)


if __name__ == "__main__":
    main()
