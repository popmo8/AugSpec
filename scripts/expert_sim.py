"""Cross-model expert-output similarity (2026-07-29, limitations §3 evidence).

For each MoE layer: evaluate EVERY routed expert on the SAME shared sample of
prompt tokens (prefill hiddens), compute pairwise cosine over the flattened
outputs, report the mean off-diagonal similarity ("shared component" of the
expert pool). Shared calibration prompts = Spec-Bench, 2 per category, seed 0.
DeepSeek's always-on shared experts are EXCLUDED — we measure the routed pool.

Usage: python scripts/expert_sim.py --model {qwen3,mixtral,deepseek,gptoss}
Output: output/expert_sim/{model}.npz  (per-layer [E,E] cosine + summary)
"""
import argparse, random
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from aug_spec.runtime.specbench import _load_spec_bench_questions

MODELS = {
    "qwen3":    "Qwen/Qwen3-30B-A3B-Base",
    "mixtral":  "mistralai/Mixtral-8x7B-v0.1",
    "deepseek": "deepseek-ai/deepseek-moe-16b-base",
    "gptoss":   "openai/gpt-oss-20b",
}
S = 64          # shared token sample per layer
QPC = 2         # prompts per category


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
    return out[:16]


def moe_blocks(model, name):
    """[(layer_idx, block_module, experts_obj)] — experts_obj 是 ModuleList
    (qwen3/mixtral/deepseek) 或 fused GptOssExperts (gptoss)。"""
    out = []
    for li, layer in enumerate(model.model.layers):
        for attr in ("mlp", "block_sparse_moe"):
            blk = getattr(layer, attr, None)
            if blk is None:
                continue
            ex = getattr(blk, "experts", None)
            if ex is None:
                continue
            if isinstance(ex, torch.nn.ModuleList) or hasattr(ex, "gate_up_proj"):
                out.append((li, blk, ex))
            break
    return out


def gptoss_expert_out(ex, x, e):
    """照抄 transformers GptOssExperts.forward 的單 expert 路徑。"""
    gate_up = x @ ex.gate_up_proj[e] + ex.gate_up_proj_bias[e]
    gate, up = gate_up[..., ::2], gate_up[..., 1::2]
    gate = gate.clamp(min=None, max=ex.limit)
    up = up.clamp(min=-ex.limit, max=ex.limit)
    glu = gate * torch.sigmoid(gate * ex.alpha)
    return ((up + 1) * glu) @ ex.down_proj[e] + ex.down_proj_bias[e]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(MODELS))
    args = ap.parse_args()
    mid = MODELS[args.model]
    out_dir = Path("output/expert_sim"); out_dir.mkdir(parents=True, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(mid, trust_remote_code=True)
    kw = dict(torch_dtype=torch.bfloat16, device_map="cuda",
              trust_remote_code=True)
    if args.model == "gptoss":
        try:
            from transformers import Mxfp4Config
            kw["quantization_config"] = Mxfp4Config(dequantize=True)
        except ImportError:
            pass
    model = AutoModelForCausalLM.from_pretrained(mid, **kw)
    model.eval()

    blocks = moe_blocks(model, args.model)
    print(f"{args.model}: {len(blocks)} MoE blocks", flush=True)
    assert blocks, "找不到 MoE block"

    captured = {li: [] for li, _, _ in blocks}
    hooks = []
    for li, blk, _ in blocks:
        def mk(li):
            def pre(_m, inp):
                h = inp[0]
                captured[li].append(h.detach().reshape(-1, h.shape[-1]).cpu())
            return pre
        hooks.append(blk.register_forward_pre_hook(mk(li)))

    for pi, p in enumerate(calib_prompts()):
        ids = tok(p, return_tensors="pt", truncation=True,
                  max_length=1024).input_ids.to(model.device)
        with torch.no_grad():
            model(ids, use_cache=False)
        print(f"  prefill {pi+1} done ({ids.shape[1]} tok)", flush=True)
    for h in hooks:
        h.remove()

    rng = np.random.default_rng(0)
    mats, offdiag = {}, []
    for li, blk, ex in blocks:
        pool = torch.cat(captured[li])                       # [T,H] cpu bf16
        idx = rng.choice(pool.shape[0], size=min(S, pool.shape[0]),
                         replace=False)
        x = pool[idx].to(model.device, torch.bfloat16)       # [S,H]
        outs = []
        with torch.no_grad():
            if hasattr(ex, "gate_up_proj"):                  # gptoss fused
                E = ex.gate_up_proj.shape[0]
                for e in range(E):
                    outs.append(gptoss_expert_out(ex, x, e).float()
                                .flatten().cpu())
            else:
                for e, m in enumerate(ex):
                    outs.append(m(x).float().flatten().cpu())
        O = torch.stack(outs)                                # [E, S*H]
        O = O / (O.norm(dim=1, keepdim=True) + 1e-8)
        C = (O @ O.T).numpy()
        mats[f"layer_{li}"] = C.astype(np.float16)
        E = C.shape[0]
        off = (C.sum() - np.trace(C)) / (E * (E - 1))
        offdiag.append(off)
        if li % 8 == 0:
            print(f"  layer {li}: E={E} mean_offdiag={off:.4f}", flush=True)

    mean, std = float(np.mean(offdiag)), float(np.std(offdiag))
    np.savez_compressed(out_dir / f"{args.model}.npz",
                        offdiag=np.array(offdiag), **mats)
    print(f"RESULT {args.model}: mean_offdiag={mean:.4f} ± {std:.4f} "
          f"(layers={len(offdiag)})", flush=True)


if __name__ == "__main__":
    main()
