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
    they need — freq_slice uses `weights`; co-occurrence will use `cooccur`."""
    active: List[int]                       # expert ids with non-zero weight
    weights: List[float]                    # this cycle's per-expert weights
    layer_idx: int = -1
    cooccur: Optional[torch.Tensor] = None  # [n,n] co-occurrence (B1 fills)
    l2dist: Optional[torch.Tensor] = None   # specmoe expert distances (future)


class ClusterMethod:
    """Override `assign`; `prepare` is an optional once-per-layer hook for
    methods that precompute on a coarser cadence than per-cycle (e.g. a
    windowed co-occurrence matrix). Default `prepare` is a no-op."""

    def prepare(self, adapter, blocks) -> None:
        pass

    def assign(self, ctx: ClusterContext, K: int) -> List[List[int]]:
        raise NotImplementedError
