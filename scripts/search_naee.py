#!/usr/bin/env python
"""D1 — NAEE / Enumeration-Pruning kept-set search
(baseline_tables_plan.md WS-D; consumes B1's calibration captures,
produces the spec for the `static_mask` draft).

Per layer, finds the size-r expert subset minimising the Frobenius
reconstruction loss ||F'(x, C) − F(x)||_F over the calibration tokens
(NAEE Eq.3), where F'(x, C) reroutes each token within the kept set using
EXACTLY the deployment masked-forward semantics (-inf non-kept logits →
softmax → native top-k → renorm). Candidate sets:
  * exact enumeration when C(n, r) fits the candidate budget
    (Mixtral 8→1: 8; GPT-OSS 32→4: 35,960 — plan item G4 for routing), or
  * seeded random sampling of `--num-candidates` subsets
    (Qwen3 128→16: the HC-SMoE-paper O-prune(1e5) protocol; the table
    marks this row "sampled").

Needs the model loaded (expert weights) → GPU job; see
scripts/run_search_naee.sh.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import time
from pathlib import Path

import torch

from aug_spec.adapters import adapter_for_config, get_adapter
from aug_spec.runtime.loader import get_model_device, load_model

from collect_calibration import stack_expert_weights  # sibling module


@torch.no_grad()
def all_expert_outputs(adapter, block, x: torch.Tensor,
                       expert_chunk: int = 32) -> torch.Tensor:
    """[n, t, D] dense outputs of every expert on tokens x (bf16, x's
    device), via chunked batched bmm."""
    from aug_spec.kernels.bmm import bmm_swiglu
    gw, uw, dw = stack_expert_weights(adapter, block)
    outs = []
    for e0 in range(0, gw.shape[0], expert_chunk):
        outs.append(bmm_swiglu(x, gw[e0:e0 + expert_chunk],
                               uw[e0:e0 + expert_chunk],
                               dw[e0:e0 + expert_chunk]))
    return torch.cat(outs)


@torch.no_grad()
def naee_losses(E: torch.Tensor, logits: torch.Tensor,
                candidates: torch.Tensor, top_k: int, ref: torch.Tensor,
                norm_topk_prob: bool = True, cand_chunk: int = 128,
                token_chunk: int = 512) -> torch.Tensor:
    """Frobenius reconstruction loss per candidate kept-set.

    Args:
        E:          [n, T, D] all-expert outputs (any float dtype).
        logits:     [T, n] router logits.
        candidates: [C, r] long — kept expert ids per candidate.
        top_k:      native experts-per-token; routing uses
                    min(top_k, r) winners WITHIN the kept set (matches the
                    masked forward: zero-prob padding winners get zero
                    weight after the renorm, so min() is exact).
        ref:        [T, D] the true layer output for the same tokens.
        norm_topk_prob: renorm the top-k weights (Qwen3 True; matches
                    `getattr(block, "norm_topk_prob", True)`).

    Returns [C] fp32 losses (cpu).
    """
    n, T, Dm = E.shape
    C, r = candidates.shape
    device = E.device
    Et = E.permute(1, 0, 2).contiguous()          # [T, n, D] token-major
    ref = ref.to(device, torch.float32)
    logits = logits.to(device, torch.float32)
    k = min(top_k, r)
    out = torch.empty(C, dtype=torch.float64)
    for c0 in range(0, C, cand_chunk):
        cand = candidates[c0:c0 + cand_chunk].to(device)
        keep = torch.zeros(cand.shape[0], n, dtype=torch.bool, device=device)
        keep.scatter_(1, cand, True)
        acc = torch.zeros(cand.shape[0], dtype=torch.float64, device=device)
        for t0 in range(0, T, token_chunk):
            lg = logits[t0:t0 + token_chunk]                       # [t, n]
            masked = lg.unsqueeze(0).masked_fill(
                ~keep.unsqueeze(1), float("-inf"))                 # [c, t, n]
            probs = masked.softmax(dim=-1)
            w, idx = probs.topk(k, dim=-1)                         # [c, t, k]
            if norm_topk_prob:
                w = w / w.sum(dim=-1, keepdim=True).clamp_min(1e-12)
            e_t = Et[t0:t0 + token_chunk]                          # [t, n, D]
            g = e_t.unsqueeze(0).expand(cand.shape[0], -1, -1, -1).gather(
                2, idx.unsqueeze(-1).expand(-1, -1, -1, Dm))       # [c,t,k,D]
            recon = (g.float() * w.unsqueeze(-1)).sum(dim=2)       # [c, t, D]
            diff = recon - ref[t0:t0 + token_chunk].unsqueeze(0)
            acc += diff.pow(2).sum(dim=(1, 2)).double()
        out[c0:c0 + cand_chunk] = acc.sqrt().cpu()
    return out.float()


def make_candidates(n: int, r: int, budget: int, seed: int):
    """Exact enumeration when it fits the budget, else seeded sampling.
    Returns (candidates [C, r] long, exact: bool)."""
    if math.comb(n, r) <= budget:
        combos = list(itertools.combinations(range(n), r))
        return torch.tensor(combos, dtype=torch.long), True
    g = torch.Generator()
    g.manual_seed(seed)
    cand = torch.stack([torch.randperm(n, generator=g)[:r].sort().values
                        for _ in range(budget)])
    return cand, False


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-id", required=True)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--calib-dir", required=True,
                    help="B1 output dir (layer_<li>.pt + meta.json)")
    ap.add_argument("--r", type=int, required=True,
                    help="kept experts per layer (Qwen3 16 / GPT-OSS 4 / Mixtral 1)")
    ap.add_argument("--num-candidates", type=int, default=100_000)
    ap.add_argument("--tokens", type=int, default=2048,
                    help="calibration-token subsample used for the search")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cand-chunk", type=int, default=128)
    ap.add_argument("--token-chunk", type=int, default=512)
    ap.add_argument("--out", default=None,
                    help="default: output/naee/<tag>_r<r>.json")
    args = ap.parse_args()

    calib = Path(args.calib_dir)
    meta = json.loads((calib / "meta.json").read_text())
    tag = args.model_id.split("/")[-1]
    out_path = Path(args.out or f"output/naee/{tag}_r{args.r}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[D1] loading {args.model_id} (hf) ...", flush=True)
    model, _ = load_model(args.model_id, dtype=torch.bfloat16,
                          device_map="auto")
    adapter = (get_adapter(args.adapter) if args.adapter
               else adapter_for_config(model.config))
    device = get_model_device(model)
    blocks = list(adapter.iter_moe(model))
    n = adapter.num_experts(blocks[0][1])

    cands, exact = make_candidates(n, args.r, args.num_candidates, args.seed)
    print(f"[D1] {'EXACT enumeration' if exact else 'sampled search'}: "
          f"{cands.shape[0]} candidates of size {args.r} from {n}", flush=True)

    g = torch.Generator()
    g.manual_seed(args.seed)
    layers = {}
    t0 = time.perf_counter()
    for li, block in blocks:
        d = torch.load(calib / f"layer_{li}.pt", weights_only=True)
        T_all = d["input"].shape[0]
        sel = (torch.randperm(T_all, generator=g)[:args.tokens].sort().values
               if args.tokens < T_all else torch.arange(T_all))
        x = d["input"][sel].to(device, torch.bfloat16)
        E = all_expert_outputs(adapter, block, x)
        losses = naee_losses(
            E, d["router_logits"][sel], cands,
            top_k=getattr(block, "top_k", None) or
                  adapter.default_count_top_k(model),
            ref=d["output"][sel],
            norm_topk_prob=getattr(block, "norm_topk_prob", True),
            cand_chunk=args.cand_chunk, token_chunk=args.token_chunk)
        best = int(losses.argmin())
        layers[str(li)] = {
            "kept": cands[best].tolist(),
            "loss": float(losses[best]),
            "loss_mean": float(losses.mean()),
            "loss_min5": [float(v) for v in losses.topk(5, largest=False).values],
        }
        del E
        torch.cuda.empty_cache()
        print(f"[D1] layer {li}: best loss {losses[best]:.4f} "
              f"(mean {losses.mean():.4f}) kept={cands[best].tolist()[:8]}... "
              f"({time.perf_counter() - t0:.0f}s)", flush=True)

    spec = {
        "model_id": args.model_id, "r": args.r,
        "search": {"exact": exact, "num_candidates": int(cands.shape[0]),
                   "tokens": args.tokens, "seed": args.seed,
                   "calibration": {k: meta[k] for k in
                                   ("n_seq", "seq_len", "seed", "c4_file",
                                    "timestamp")},
                   "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")},
        "layers": layers,
    }
    out_path.write_text(json.dumps(spec))
    print(f"[D1] spec → {out_path} ({time.perf_counter() - t0:.0f}s)")


if __name__ == "__main__":
    main()
