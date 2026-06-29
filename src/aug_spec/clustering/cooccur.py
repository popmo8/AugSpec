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

import itertools
from typing import List

from .base import ClusterContext, ClusterMethod


class CooccurPairCluster(ClusterMethod):
    needs_cooccur = True

    def assign(self, ctx: ClusterContext, K: int) -> List[List[int]]:
        active = list(ctx.active)
        C = ctx.cooccur
        # No table yet (e.g. very first refresh before any capture) or already
        # at/under K experts → every active expert is its own cluster.
        if C is None or len(active) <= K:
            return [[e] for e in active]

        # Rank all active pairs by co-occurrence desc, then greedily accept the
        # highest while each expert stays unpaired (enforces max cluster size 2),
        # stopping once enough merges land us at K clusters.
        pairs = sorted(
            ((float(C[i, j]), i, j) for i, j in itertools.combinations(active, 2)),
            reverse=True)
        need = len(active) - K          # merges required to reach K clusters
        used: set = set()
        groups: List[List[int]] = []
        for _, i, j in pairs:
            if need <= 0:
                break
            if i in used or j in used:
                continue
            groups.append([i, j])
            used.add(i)
            used.add(j)
            need -= 1
        groups.extend([e] for e in active if e not in used)
        return groups
