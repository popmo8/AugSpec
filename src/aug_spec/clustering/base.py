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
    group_cap: int = 2                      # max experts per merged group
                                            # (= ceil(M/K); 2 = classic pairing)


class ClusterMethod:
    """Override `assign`; `prepare` is an optional once-per-layer hook for
    methods that precompute on a coarser cadence than per-cycle (e.g. a
    windowed co-occurrence matrix). Default `prepare` is a no-op."""

    # Per-forward stats the draft must accumulate for this method (gated so
    # other methods pay zero cost): needs_cooccur -> ctx.cooccur,
    # needs_activation_sim -> ctx.pair_sim (the heavier expert-output capture).
    needs_cooccur: bool = False
    needs_activation_sim: bool = False
    # act_sim_prefill_only: the offload-merge engine turns the C++ expert-output
    # capture OFF at each question's first draft start (prefill→draft boundary)
    # and back ON at question start — ctx.pair_sim freezes at the prefill state
    # and decode cycles pay no capture latency (hybrid).
    act_sim_prefill_only: bool = False

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


def greedy_group(table, active: List[int], K: int,
                 cap: int = 2) -> List[List[int]]:
    """Capacity-generalised `greedy_pair` (M>2K support, 2026-07-25).

    cap <= 2 delegates to `greedy_pair` — bit-identical behaviour for every
    pre-existing config. cap > 2 runs greedy single-linkage agglomeration on
    the same descending pair ranking, merging two groups only while the
    combined size stays <= cap, with multi-pass sweeps until K groups are
    reached; if the pass stalls (all cross-group pairs capacity-blocked) the
    smallest group is dissolved and its members re-seated one by one into the
    highest-affinity group with spare capacity. Feasibility is guaranteed by
    the top-M cutoff: |active| <= M = cap*K, so K groups of cap always fit.
    """
    if cap <= 2:
        return greedy_pair(table, active, K)
    if table is None or len(active) <= K:
        return [[e] for e in active]
    import itertools
    pairs = sorted(
        ((float(table[i][j]), i, j)
         for i, j in itertools.combinations(active, 2)),
        reverse=True)
    parent = {e: e for e in active}
    size = {e: 1 for e in active}
    members = {e: [e] for e in active}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    n_groups = len(active)
    progress = True
    while n_groups > K and progress:
        progress = False
        for _, i, j in pairs:
            if n_groups <= K:
                break
            ri, rj = find(i), find(j)
            if ri == rj or size[ri] + size[rj] > cap:
                continue
            parent[rj] = ri
            size[ri] += size[rj]
            members[ri].extend(members.pop(rj))
            del size[rj]
            n_groups -= 1
            progress = True
    while n_groups > K:                    # stalled → dissolve-and-reseat
        gid = min(members, key=lambda g: (len(members[g]), g))
        loose = members.pop(gid)
        del size[gid]
        n_groups -= 1
        for m in loose:
            best_g, best_v = None, None
            for g, mem in members.items():
                if size[g] >= cap:
                    continue
                v = max(float(table[m][x]) for x in mem)
                if best_g is None or v > best_v:
                    best_g, best_v = g, v
            if best_g is None:
                # No spare capacity anywhere — infeasible input
                # (|active| > cap*K, impossible under a top-M cutoff).
                # Degrade gracefully: smallest group takes the overflow.
                best_g = min(members, key=lambda g: (size[g], g))
            members[best_g].append(m)
            size[best_g] += 1
    return list(members.values())
