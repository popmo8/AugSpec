"""Merge-perturbation scale (2026-07-29): per model, on the same 15 calib
prompts as expert_sim_l2 —
  hidden_norm : mean ||h_t|| entering each MoE block (residual-stream scale)
  out_norm    : mean ||expert output|| over routed (token, expert) pairs
  pair_dist   : mean co-routed pairwise L2 (recomputed for consistency)
  ratio       : pair_dist / hidden_norm   (merge substitution error vs stream)
Usage: python scripts/expert_scale.py --model {qwen3,mixtral,deepseek,gptoss}
"""
import argparse
from pathlib import Path

import numpy as np
import torch

import sys
sys.path.insert(0, "scripts")
from expert_sim_l2 import (MODELS, calib_prompts, moe_blocks, router_logits,
                           expert_out)
from transformers import AutoModelForCausalLM, AutoTokenizer


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
    print(f"{args.model}: {len(blocks)} MoE blocks", flush=True)

    cap = {}
    hooks = []
    for li, blk, _ in blocks:
        def mk(li):
            def pre(_m, inp):
                cap[li] = inp[0].detach().reshape(-1, inp[0].shape[-1])
            return pre
        hooks.append(blk.register_forward_pre_hook(mk(li)))

    h_norms, o_norms, dists = [], [], []
    for pi, p in enumerate(calib_prompts()):
        ids = tok(p, return_tensors="pt", truncation=True,
                  max_length=512).input_ids.to(model.device)
        with torch.no_grad():
            model(ids, use_cache=False)
        for li, blk, ex in blocks:
            h = cap[li]
            h_norms.append(float(h.float().norm(dim=1).mean()))
            logits = router_logits(blk, h)
            E = logits.shape[1]
            top = logits.topk(topk, dim=-1).indices
            per_tok = {}
            for e in range(E):
                mask = (top == e).any(dim=-1)
                idx = mask.nonzero(as_tuple=True)[0]
                if idx.numel() == 0:
                    continue
                with torch.no_grad():
                    o = expert_out(ex, h[idx], e).float()
                o_norms.append(float(o.norm(dim=1).mean()))
                for k_, t in enumerate(idx.tolist()):
                    per_tok.setdefault(t, []).append(o[k_])
            for vs in per_tok.values():
                if len(vs) < 2:
                    continue
                V = torch.stack(vs)
                M = torch.cdist(V.unsqueeze(0), V.unsqueeze(0)).squeeze(0)
                iu = torch.triu_indices(len(vs), len(vs), offset=1)
                dists.append(float(M[iu[0], iu[1]].mean()))
        print(f"  prompt {pi+1}/15 done", flush=True)
    for hk in hooks:
        hk.remove()

    hn, on, pd = map(lambda x: float(np.mean(x)), (h_norms, o_norms, dists))
    print(f"RESULT {args.model}: hidden_norm={hn:.2f} out_norm={on:.2f} "
          f"pair_dist={pd:.2f} | out/hidden={on/hn:.3f} "
          f"dist/hidden={pd/hn:.3f}", flush=True)


if __name__ == "__main__":
    main()
