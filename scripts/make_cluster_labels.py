#!/usr/bin/env python3
"""Build STATIC per-layer cluster-label files for the partition A/B test.

Reads an AUG_DUMP_ACTIVE_SET JSONL (per (layer,question,cycle) selected expert
set) and emits, for each requested method, a JSON file consumed by
ScoreBasedAvgDraft via AUG_CLUSTER_LABELS:

    {"K": 16, "method": "...", "n": 128,
     "labels": {"<layer>": [cluster_label for expert 0..n-1], ...}}

Methods
  cooccur  : agglomerative (average-linkage) on cosine co-occurrence.
             similarity(i,j) = C[i,j]/sqrt(C[i,i]C[j,j]), C[i,j] = #cycles both
             i and j are in the selected set. Frequency-weighted (cosine, not
             lift) so high-mass experts drive the grouping. -> K clusters.
  random   : random balanced partition of n experts into K groups (seeded).
             The floor control: does the partition even matter?
  statfreq : static frequency-slice — rank experts by total appearance count,
             cut into K contiguous slices. Isolates "static vs dynamic" from
             "which signal", since the live baseline (freq_slice) is dynamic.

Experts never seen active get label 0 (they are ~never grouped at run time).

Usage: make_cluster_labels.py <active_set.jsonl> <out_dir> [K]
"""
import collections
import json
import sys
from pathlib import Path

import numpy as np
from scipy.cluster.hierarchy import fcluster, leaves_list, linkage
from scipy.spatial.distance import squareform


def _cooccur_linkage(C):
    """Return (linkage Z, seen-expert indices, distance matrix) for the cosine
    co-occurrence of the seen experts, or (None, idx, None) if too few seen."""
    n = C.shape[0]
    d = np.diag(C).astype(float)
    idx = np.where(d > 0)[0]
    if len(idx) < 3:
        return None, idx, None
    sim = np.zeros((n, n))
    denom = np.sqrt(np.outer(d, d))
    nz = denom > 0
    sim[nz] = C[nz] / denom[nz]
    np.clip(sim, 0.0, 1.0, out=sim)
    dist = 1.0 - sim
    np.fill_diagonal(dist, 0.0)
    dist = (dist + dist.T) / 2.0           # enforce exact symmetry
    sub = dist[np.ix_(idx, idx)]
    Z = linkage(squareform(sub, checks=False), method="average")
    return Z, idx, sub


def cooccur_labels(C, K):
    """Plan B2 as written: agglomerative average-linkage on cosine
    co-occurrence, cut to K clusters (maxclust). Tends to be UNBALANCED."""
    Z, idx, _ = _cooccur_linkage(C)
    labels = np.zeros(C.shape[0], dtype=int)
    if Z is None:
        for c, i in enumerate(idx):
            labels[i] = c % K
        return labels.tolist()
    sub_lab = fcluster(Z, t=K, criterion="maxclust") - 1
    for i, lab in zip(idx, sub_lab):
        labels[i] = int(lab)
    return labels.tolist()


def cooccur_bal_labels(C, K):
    """Balanced co-occurrence: take the dendrogram leaf order (places
    co-occurring experts adjacent) and cut it into K equal contiguous slices.
    Uses co-occurrence structure but is balanced like random/statfreq, so the
    A/B vs those isolates the *signal* from the *balance*."""
    Z, idx, _ = _cooccur_linkage(C)
    labels = np.zeros(C.shape[0], dtype=int)
    if Z is None:
        for c, i in enumerate(idx):
            labels[i] = c % K
        return labels.tolist()
    order = idx[leaves_list(Z)]            # seen experts in dendrogram order
    m = len(order)
    for slot, i in enumerate(order):
        labels[i] = slot * K // m
    return labels.tolist()


def random_labels(n, K, seed):
    rng = np.random.default_rng(seed)
    lab = np.tile(np.arange(K), (n + K - 1) // K)[:n]
    rng.shuffle(lab)
    return lab.tolist()


def statfreq_labels(freq, K):
    """Rank experts by appearance count desc, cut into K contiguous slices."""
    n = len(freq)
    order = sorted(range(n), key=lambda i: (-freq[i], i))
    labels = [0] * n
    for slot, i in enumerate(order):
        labels[i] = slot * K // n
    return labels


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)
    dump = Path(sys.argv[1])
    out_dir = Path(sys.argv[2])
    K = int(sys.argv[3]) if len(sys.argv) > 3 else 16
    out_dir.mkdir(parents=True, exist_ok=True)

    # per-layer co-occurrence matrices + appearance counts
    n = 0
    layers = {}
    with open(dump) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r.get("degenerate"):
                continue
            n = max(n, r["n"])
            li = r["layer"]
            if li not in layers:
                layers[li] = None
            a = r["active"]
            if layers[li] is None:
                layers[li] = np.zeros((r["n"], r["n"]), dtype=np.int64)
            M = layers[li]
            for x in a:
                M[x, x] += 1
            for ii in range(len(a)):
                for jj in range(ii + 1, len(a)):
                    M[a[ii], a[jj]] += 1
                    M[a[jj], a[ii]] += 1

    layer_ids = sorted(layers)
    print(f"[labels] layers={len(layer_ids)} n_experts={n} K={K}")

    methods = {"cooccur": {}, "cooccur_bal": {}, "random": {}, "statfreq": {}}
    for li in layer_ids:
        C = layers[li]
        freq = np.diag(C).tolist()
        methods["cooccur"][str(li)] = cooccur_labels(C, K)
        methods["cooccur_bal"][str(li)] = cooccur_bal_labels(C, K)
        methods["random"][str(li)] = random_labels(n, K, seed=1000 + li)
        methods["statfreq"][str(li)] = statfreq_labels(freq, K)

    for method, labmap in methods.items():
        # report cluster-size distribution sanity
        sizes = []
        for li in layer_ids:
            cnt = collections.Counter(labmap[str(li)])
            sizes.append(len(cnt))            # # distinct clusters used
        path = out_dir / f"labels_{method}_K{K}.json"
        path.write_text(json.dumps(
            {"K": K, "method": method, "n": n, "labels": labmap}) + "\n")
        print(f"[labels] {method:9s} -> {path}  "
              f"(clusters/layer: mean={np.mean(sizes):.1f} "
              f"min={min(sizes)} max={max(sizes)})")


if __name__ == "__main__":
    main()
