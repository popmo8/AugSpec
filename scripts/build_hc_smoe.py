#!/usr/bin/env python
"""C1 — HC-SMoE offline clustering (baseline_tables_plan.md WS-C).

Reads B1's `expert_stats.pt` (per-layer mean expert outputs `o` + top-k
`freq`) and clusters every layer's experts into K groups by
average-linkage agglomerative clustering on Euclidean distances between
expert outputs — HC-SMoE's method (deterministic, no initialisation).
Emits the grouping spec consumed by the `static_merge` draft.

Pure CPU, seconds — login-node safe.

  python scripts/build_hc_smoe.py \\
      --calib-dir output/calibration/Qwen3-30B-A3B-Base --K 16
  # → output/hc_smoe/Qwen3-30B-A3B-Base_K16.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from aug_spec.clustering.agglomerative import average_linkage_groups


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--calib-dir", required=True,
                    help="B1 output dir (holds expert_stats.pt + meta.json)")
    ap.add_argument("--K", type=int, required=True,
                    help="clusters per layer (Qwen3 16 / GPT-OSS 4 / Mixtral 1)")
    ap.add_argument("--out", default=None,
                    help="default: output/hc_smoe/<tag>_K<K>.json")
    args = ap.parse_args()

    calib = Path(args.calib_dir)
    meta = json.loads((calib / "meta.json").read_text())
    stats = torch.load(calib / "expert_stats.pt", weights_only=True)

    tag = meta["model_id"].split("/")[-1]
    out = Path(args.out or f"output/hc_smoe/{tag}_K{args.K}.json")
    out.parent.mkdir(parents=True, exist_ok=True)

    layers = {}
    for li in meta["layer_indices"]:
        o = stats["o"][li].float()                      # [n, D]
        D = torch.cdist(o, o)                           # Euclidean, HC-SMoE
        groups = average_linkage_groups(D, args.K)
        freq = stats["freq"][li].tolist()
        layers[str(li)] = {"groups": groups, "freq": freq}
        sizes = sorted((len(g) for g in groups), reverse=True)
        print(f"[C1] layer {li}: {len(groups)} groups, sizes {sizes}")

    spec = {
        "model_id": meta["model_id"],
        "K": args.K,
        "count_top_k": stats["count_top_k"],
        "calibration": {k: meta[k] for k in
                        ("n_seq", "seq_len", "stored_tokens", "seed",
                         "c4_file", "timestamp")},
        "layers": layers,
    }
    out.write_text(json.dumps(spec))
    print(f"[C1] spec → {out}")


if __name__ == "__main__":
    main()
