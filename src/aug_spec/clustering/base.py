"""Clustering strategy ABC + the stats bag it consumes.

A *cluster method* partitions a layer's active experts into at most K groups;
each group is then merged into one dense expert (see `merging/`). The only
member today is frequency-slice (the original `_assign_clusters`); co-occurrence
clustering plugs in later (B2) without touching the merge/draft code.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import torch


@dataclass
class ClusterContext:
    """Per-cycle statistics a cluster method may use. Methods take only what
    they need — freq_slice uses `weights`; cooccur_pair uses `cooccur`;
    activation_similarity uses `pair_sim`."""
    active: List[int]                       # expert ids with non-zero weight
    weights: List[float]                    # this cycle's per-expert weights
    layer_idx: int = -1
    cooccur: Optional[torch.Tensor] = None  # [n,n] co-occurrence count (B1)
    pair_sim: Optional[torch.Tensor] = None  # [n,n] activation cosine, -1 = unvisited
    l2dist: Optional[torch.Tensor] = None   # specmoe expert distances (future)


class ClusterMethod:
    """Override `assign`; `prepare` is an optional once-per-layer hook for
    methods that precompute on a coarser cadence than per-cycle (e.g. a
    windowed co-occurrence matrix). Default `prepare` is a no-op."""

    # Per-forward stats the draft must accumulate for this method (gated so
    # other methods pay zero cost): needs_cooccur -> ctx.cooccur,
    # needs_activation_sim -> ctx.pair_sim (the heavier expert-output capture).
    needs_cooccur: bool = False
    needs_activation_sim: bool = False

    def prepare(self, adapter, blocks) -> None:
        pass

    def assign(self, ctx: ClusterContext, K: int) -> List[List[int]]:
        raise NotImplementedError


def greedy_pair(table, active: List[int], K: int) -> List[List[int]]:
    """Greedy maximum-value matching shared by cooccur_pair / activation_similarity.

    Rank all active pairs by `table[i][j]` descending; greedily accept the
    highest while each expert stays unpaired (enforces max cluster size 2),
    stopping once enough merges land at K clusters. Remaining experts are
    singletons. `table` is [n,n] (co-occurrence count or pairwise similarity);
    a -1 sentinel for unvisited pairs simply ranks them last. None / too-few
    active → every expert its own cluster.
    """
    import itertools
    if table is None or len(active) <= K:
        return [[e] for e in active]
    pairs = sorted(
        ((float(table[i][j]), i, j) for i, j in itertools.combinations(active, 2)),
        reverse=True)
    need = len(active) - K
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
