"""Routing-drift analysis (2026-07-27): quantify dynamic-vs-static routing
from routing_trace npz files. Three metrics, all per (question, layer),
averaged per task:

1. drift(w)   = 1 - Jaccard(top32 set of decode window w, prefill top32 set)
                w = 8 EQUAL-fraction windows of each question's decode
                (relative position, so short outputs still contribute).
2. distinct(t)= |union of experts routed up to decode token t| / 128.
3. fix16      = routing-mass share of the best FIXED 16-expert set chosen
                post-hoc from the whole decode (upper bound for any static
                keep-16 policy, i.e. SpecMoE's ceiling).
"""
import sys
from collections import Counter
from pathlib import Path

import numpy as np

NW = 8
TOPK = 32

def top_set(ids_flat, k=TOPK):
    c = Counter(ids_flat.tolist())
    return set(e for e, _ in c.most_common(k))

def analyze(files):
    drift, distinct, fix16, lens = [], [], [], []
    for f in files:
        z = np.load(f)
        pre, dec = z["prefill_top8"], z["decode_top8"]   # [L,P,8],[L,T,8]
        L, T, _ = dec.shape
        if T < 4 * NW:
            continue
        bounds = np.linspace(0, T, NW + 1).astype(int)
        d_layers = np.zeros((L, NW)); dist_layers = np.zeros((L, NW))
        f16 = np.zeros(L)
        for li in range(L):
            s0 = top_set(pre[li].ravel())
            seen = set()
            for w in range(NW):
                seg = dec[li, bounds[w]:bounds[w+1]].ravel()
                sw = top_set(seg)
                d_layers[li, w] = 1 - len(sw & s0) / len(sw | s0)
                seen |= set(seg.tolist())
                dist_layers[li, w] = len(seen) / 128.0
            c = Counter(dec[li].ravel().tolist())
            tot = sum(c.values())
            f16[li] = sum(n for _, n in c.most_common(16)) / tot
        drift.append(d_layers.mean(0))
        distinct.append(dist_layers.mean(0))
        fix16.append(f16.mean())
        lens.append(T)
    return (np.mean(drift, axis=0), np.mean(distinct, axis=0),
            float(np.mean(fix16)), len(fix16), float(np.mean(lens)))

def main():
    td = Path(sys.argv[1] if len(sys.argv) > 1 else "output/routing_trace")
    for cat in ("translation", "rag"):
        files = sorted(td.glob(f"{cat}_*.npz"))
        if not files:
            print(f"{cat}: (no files)"); continue
        dr, di, f16, n, mt = analyze(files)
        print(f"=== {cat} (n={n}, mean decode len={mt:.0f}) ===")
        print("  drift vs prefill (window 1..8): "
              + " ".join(f"{x:.3f}" for x in dr))
        print("  distinct experts /128 (cum):    "
              + " ".join(f"{x:.3f}" for x in di))
        print(f"  best-fixed-16 coverage: {f16:.3f}")

if __name__ == "__main__":
    main()
