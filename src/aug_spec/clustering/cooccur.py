"""Co-occurrence pair clustering.

Greedy maximum-co-occurrence matching on the active experts: repeatedly merge
the highest-co-occurrence still-unpaired pair until only K clusters remain, so
every cluster is a pair or a singleton (max size 2). Uses the question-level
co-occurrence table the draft accumulates (`ctx.cooccur`, [n, n]).

Note: this is the "merge what fires together" (must-link) direction. Prior
partition A/B (2026-06-29) found must-link co-occurrence underperforms random on
acceptance; the motivation here is cache-ability (uniform within-weight + stable
pairs → reusable merges), so evaluate it on cache hit-rate / speed, not just
acceptance.
"""

from __future__ import annotations

from typing import List

from .base import ClusterContext, ClusterMethod, greedy_group


class CooccurPairCluster(ClusterMethod):
    needs_cooccur = True

    def assign(self, ctx: ClusterContext, K: int) -> List[List[int]]:
        return greedy_group(ctx.cooccur, list(ctx.active), K, ctx.group_cap)
