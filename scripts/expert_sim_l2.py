"""Co-routed L2 expert similarity — EXACT replica of the paper's act-sim
accumulation (drafts/base.py: per-token routed-expert raw outputs, pairwise
torch.cdist L2, num/cnt per pair, sim = -mean L2), computed on 15 shared
Spec-Bench prompts, prefill-only, for four MoE models.

Reports per model:  mean co-routed L2 (paper's raw metric),
                    relative L2 = L2 / mean pair output norm (跨模型可比),
                    pair coverage (visited pairs / all pairs).
Usage: python scripts/expert_sim_l2.py --model {qwen3,mixtral,deepseek,gptoss}
"""
import argparse, random
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from aug_spec.runtime.specbench import _load_spec_bench_questions

MODELS = {
    "qwen3":    ("Qwen/Qwen3-30B-A3B-Base", 8),
    "mixtral":  ("mistralai/Mixtral-8x7B-v0.1", 2),
    "deepseek": ("deepseek-ai/deepseek-moe-16b-base", 6),
    "gptoss":   ("openai/gpt-oss-20b", 4),
}
N_PROMPTS = 15
MAX_LEN = 512


def calib_prompts():
    qs = _load_spec_bench_questions(Path("data/spec_bench"))
    by_cat = {}
    for q in qs:
        by_cat.setdefault(q["category"], []).append(q)
    for pool in by_cat.values():
        random.Random(0).shuffle(pool)
    out, i = [], 0
    while len(out) < N_PROMPTS:                 # round-robin 各類別
        for cat in sorted(by_cat):
            if i < len(by_cat[cat]) and len(out) < N_PROMPTS:
                out.append(by_cat[cat][i]["turns"][0])
        i += 1
    return out


def moe_blocks(model):
    out = []
    for li, layer in enumerate(model.model.layers):
        for attr in ("mlp", "block_sparse_moe"):
            blk = getattr(layer, attr, None)
            if blk is None:
                continue
            ex = getattr(blk, "experts", None)
            if ex is not None and (isinstance(ex, torch.nn.ModuleList)
                                   or hasattr(ex, "gate_up_proj")):
                out.append((li, blk, ex))
            break
    return out


def router_logits(blk, h):
    """h [T,H] → [T,E]。gate/router 的線性層即 logits(topk ids 與 softmax 同序)。"""
    for attr in ("gate", "router"):
        g = getattr(blk, attr, None)
        if g is None:
            continue
        if hasattr(g, "weight"):
            return h @ g.weight.t().to(h.dtype)
        return g(h)
    raise RuntimeError("no gate/router")


def gptoss_expert_out(ex, x, e):
    gate_up = x @ ex.gate_up_proj[e] + ex.gate_up_proj_bias[e]
    gate, up = gate_up[..., ::2], gate_up[..., 1::2]
    gate = gate.clamp(min=None, max=ex.limit)
    up = up.clamp(min=-ex.limit, max=ex.limit)
    glu = gate * torch.sigmoid(gate * ex.alpha)
    return ((up + 1) * glu) @ ex.down_proj[e] + ex.down_proj_bias[e]


def expert_out(ex, x, e):
    if hasattr(ex, "gate_up_proj"):
        return gptoss_expert_out(ex, x, e)
    return ex[e](x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(MODELS))
    args = ap.parse_args()
    mid, topk = MODELS[args.model]
    out_dir = Path("output/expert_sim_l2"); out_dir.mkdir(parents=True, exist_ok=True)

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
    print(f"{args.model}: {len(blocks)} MoE blocks, top-{topk}", flush=True)

    E = None
    num = {}; cnt = {}; dnum = {}                # dnum: 相對距離累加
    cap = {}
    hooks = []
    for li, blk, _ in blocks:
        def mk(li):
            def pre(_m, inp):
                cap[li] = inp[0].detach().reshape(-1, inp[0].shape[-1])
            return pre
        hooks.append(blk.register_forward_pre_hook(mk(li)))

    for pi, p in enumerate(calib_prompts()):
        ids = tok(p, return_tensors="pt", truncation=True,
                  max_length=MAX_LEN).input_ids.to(model.device)
        with torch.no_grad():
            model(ids, use_cache=False)
        for li, blk, ex in blocks:
            h = cap[li]                                        # [T,H] gpu
            logits = router_logits(blk, h)                     # [T,E]
            if E is None:
                E = logits.shape[1]
            top = logits.topk(topk, dim=-1).indices            # [T,k]
            # 每 expert 一次 batch forward(僅 routed tokens)
            outs = {}                                          # e -> {t: vec}
            for e in range(E):
                mask = (top == e).any(dim=-1)
                idx = mask.nonzero(as_tuple=True)[0]
                if idx.numel() == 0:
                    continue
                with torch.no_grad():
                    o = expert_out(ex, h[idx], e).float()
                outs[e] = (idx, o)
            if li not in num:
                num[li] = torch.zeros(E, E); cnt[li] = torch.zeros(E, E)
                dnum[li] = torch.zeros(E, E)
            # token → 其 routed experts 的輸出(照論文:同 token 內 cdist)
            per_tok = {}
            for e, (idx, o) in outs.items():
                for k_, t in enumerate(idx.tolist()):
                    per_tok.setdefault(t, []).append((e, o[k_]))
            for members in per_tok.values():
                if len(members) < 2:
                    continue
                es = [e for e, _ in members]
                V = torch.stack([v for _, v in members])       # [k,D]
                M = torch.cdist(V.unsqueeze(0), V.unsqueeze(0)).squeeze(0).cpu()
                nrm = V.norm(dim=1).cpu()
                R = M / (0.5 * (nrm[:, None] + nrm[None, :]) + 1e-8)
                idxs = torch.tensor(es)
                num[li][idxs[:, None], idxs[None, :]] += M
                dnum[li][idxs[:, None], idxs[None, :]] += R
                cnt[li][idxs[:, None], idxs[None, :]] += 1.0
        print(f"  prompt {pi+1}/{N_PROMPTS} done ({ids.shape[1]} tok)", flush=True)
    for hk in hooks:
        hk.remove()

    l2s, rels, covs, mats = [], [], [], {}
    for li in num:
        c = cnt[li]; off = ~torch.eye(c.shape[0], dtype=bool)
        nz = (c > 0) & off
        if nz.sum() == 0:
            continue
        l2s.append(float((num[li][nz] / c[nz]).mean()))
        rels.append(float((dnum[li][nz] / c[nz]).mean()))
        covs.append(float(nz.sum() / off.sum()))
        mats[f"l2_{li}"] = (num[li] / c.clamp_min(1)).numpy().astype(np.float16)
        mats[f"cnt_{li}"] = c.numpy().astype(np.float32)
    np.savez_compressed(out_dir / f"{args.model}.npz", **mats)
    print(f"RESULT {args.model}: co-routed L2 = {np.mean(l2s):.4f} ± {np.std(l2s):.4f} | "
          f"relative L2 = {np.mean(rels):.4f} ± {np.std(rels):.4f} | "
          f"pair coverage = {np.mean(covs):.3f}", flush=True)


if __name__ == "__main__":
    main()
