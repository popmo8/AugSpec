"""Frequency-slice clustering — the original `_assign_clusters`, verbatim.

Sort active experts by descending weight, cut into K contiguous, near-equal
slices. Groups experts of comparable activation frequency; calibration-free.
Default cluster method.
"""

from __future__ import annotations

from typing import List

from .base import ClusterContext, ClusterMethod


class FreqSliceCluster(ClusterMethod):
    def assign(self, ctx: ClusterContext, K: int) -> List[List[int]]:
        ordered = sorted(ctx.active, key=lambda i: -ctx.weights[i])
        m = len(ordered)
        k = min(K, m)
        return [ordered[j * m // k:(j + 1) * m // k] for j in range(k)]
