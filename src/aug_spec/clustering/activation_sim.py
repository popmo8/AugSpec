"""Activation-similarity pair clustering (output-hidden-state cosine, Sub-MoE).

Same greedy max-value, max-size-2 pairing as cooccur_pair, but ranks pairs by
the mean cosine similarity of the two experts' OUTPUT hidden states over the
tokens that routed to both (`ctx.pair_sim`, accumulated by the draft from the
C++ engine's captured per-expert outputs). Unvisited pairs are -1 (cosine min),
so they rank last and are only used to reach K when nothing better remains.

Unlike cooccur (merge what *fires together*), this merges what is *functionally
redundant* — the theoretically-right merge criterion.
"""

from __future__ import annotations

from typing import List

from .base import ClusterContext, ClusterMethod, greedy_pair


class ActivationSimCluster(ClusterMethod):
    needs_activation_sim = True

    def __init__(self, metric: str = "cosine"):
        if metric not in ("cosine", "l2"):
            raise ValueError(f"activation_similarity metric must be cosine|l2, "
                             f"got {metric!r}")
        # The draft reads self.metric to build ctx.pair_sim: cosine of expert
        # outputs, or NEGATIVE L2 distance (so greedy's max = closest outputs).
        self.metric = metric

    def assign(self, ctx: ClusterContext, K: int) -> List[List[int]]:
        return greedy_pair(ctx.pair_sim, list(ctx.active), K)
