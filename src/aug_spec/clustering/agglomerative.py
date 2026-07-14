"""Offline average-linkage agglomerative clustering (HC-SMoE, C1).

Not a `ClusterMethod` — the per-cycle registry methods partition the
*active* set online; this is the one-shot, whole-expert-set clustering the
HC-SMoE baseline runs offline on calibration expert outputs
(baseline_tables_plan.md WS-C). Deterministic (no initialisation), pure
CPU, n≤128 so the naive O(n³) Lance-Williams loop is instant.
"""

from __future__ import annotations

from typing import List

import torch


def average_linkage_groups(D: torch.Tensor, K: int) -> List[List[int]]:
    """Cluster n items into K groups by average-linkage agglomeration.

    Args:
        D: [n, n] symmetric pairwise distance matrix (e.g. Euclidean
            between per-expert mean outputs), zero diagonal.
        K: target number of clusters (1 <= K <= n).

    Returns:
        K groups (sorted member lists, ordered by smallest member) that
        partition range(n). Average linkage: d(A, B) = mean over all
        cross pairs of D — maintained exactly via pair-sum bookkeeping.
        Ties break on the smallest (i, j) index pair, so the result is
        fully deterministic.
    """
    n = D.shape[0]
    if D.shape != (n, n):
        raise ValueError(f"D must be square, got {tuple(D.shape)}")
    if not 1 <= K <= n:
        raise ValueError(f"K must be in [1, {n}], got {K!r}")

    D = D.double()
    members: List[List[int]] = [[i] for i in range(n)]
    # pair_sum[x, y] = sum of D over all cross pairs of clusters x, y;
    # average distance = pair_sum / (|x| * |y|).
    pair_sum = D.clone()
    alive = list(range(n))

    while len(alive) > K:
        best = None          # (avg_dist, i, j) — smallest wins, index tie-break
        for ai in range(len(alive)):
            for aj in range(ai + 1, len(alive)):
                x, y = alive[ai], alive[aj]
                avg = pair_sum[x, y].item() / (len(members[x]) * len(members[y]))
                key = (avg, x, y)
                if best is None or key < best:
                    best = key
        _, x, y = best
        # Merge y into x (Lance-Williams for average linkage: sums add).
        for z in alive:
            if z == x or z == y:
                continue
            s = pair_sum[x, z] + pair_sum[y, z]
            pair_sum[x, z] = s
            pair_sum[z, x] = s
        members[x] = sorted(members[x] + members[y])
        alive.remove(y)

    groups = sorted((members[x] for x in alive), key=lambda g: g[0])
    return [list(g) for g in groups]
