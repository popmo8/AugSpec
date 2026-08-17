#!/usr/bin/env python
"""MC-SMoE offline grouping (M-SMoE merging stage; Table 1 merge-static
baseline, tab_main_baselines.tex).

Reads B1's calibration (`expert_stats.pt` freq + per-layer
`layer_<li>.pt` router_logits) and emits the grouping spec consumed by
the `mc_smoe` draft:
  * dominant experts — adaptive layer-wise ratio (layer-max-normalised
    frequency, global top L*K; ≥1 per layer);
  * groups — each non-dominant expert joins its most similar dominant,
    similarity = cosine of router-logit vectors over the calibration
    tokens (M-SMoE Eq.1).
The merge itself (permutation alignment + frequency-weighted averaging)
needs the model weights and runs in the draft's `prepare()` at load time.

Pure CPU, seconds — login-node safe.

  python scripts/build_mc_smoe.py \\
      --calib-dir output/calibration/Qwen3-30B-A3B-Base --K 16
  # → output/mc_smoe/Qwen3-30B-A3B-Base_K16.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from aug_spec.clustering.mc_smoe import (group_by_similarity,
                                         router_logits_similarity,
                                         select_dominant_adaptive)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--calib-dir", required=True,
                    help="B1 output dir (expert_stats.pt + layer_<li>.pt)")
    ap.add_argument("--K", type=int, required=True,
                    help="AVERAGE clusters per layer — the global dominant "
                         "budget is num_layers*K, split adaptively "
                         "(Qwen3 16 / GPT-OSS 4 / Mixtral 1)")
    ap.add_argument("--out", default=None,
                    help="default: output/mc_smoe/<tag>_K<K>.json")
    args = ap.parse_args()

    calib = Path(args.calib_dir)
    meta = json.loads((calib / "meta.json").read_text())
    stats = torch.load(calib / "expert_stats.pt", weights_only=True)

    tag = meta["model_id"].split("/")[-1]
    out = Path(args.out or f"output/mc_smoe/{tag}_K{args.K}.json")
    out.parent.mkdir(parents=True, exist_ok=True)

    lis = meta["layer_indices"]
    freq = torch.stack([stats["freq"][li] for li in lis]).long()   # [L, n]
    dominant = select_dominant_adaptive(freq, args.K)

    layers = {}
    for row, li in enumerate(lis):
        logits = torch.load(calib / f"layer_{li}.pt",
                            weights_only=True)["router_logits"]    # [T, n]
        sim = router_logits_similarity(logits)
        groups = group_by_similarity(sim, dominant[row])
        layers[str(li)] = {"groups": groups,
                           "dominant": dominant[row],
                           "freq": freq[row].tolist()}
        sizes = sorted((len(g) for g in groups), reverse=True)
        print(f"[MC] layer {li}: {len(groups)} dominant, sizes {sizes}")

    counts = [len(d) for d in dominant]
    print(f"[MC] dominant/layer: min {min(counts)} max {max(counts)} "
          f"total {sum(counts)} (= {len(lis)}*{args.K})")

    spec = {
        "model_id": meta["model_id"],
        "method": "mc_smoe",
        "K": args.K,
        "count_top_k": stats["count_top_k"],
        "calibration": {k: meta[k] for k in
                        ("n_seq", "seq_len", "stored_tokens", "seed",
                         "c4_file", "timestamp")},
        "layers": layers,
    }
    out.write_text(json.dumps(spec))
    print(f"[MC] spec → {out}")


if __name__ == "__main__":
    main()
