"""Random balanced partition — a control / baseline cluster method.

Mirrors freq_slice's per-cycle, balanced structure but cuts a *random* shuffle
of the active experts instead of the frequency-sorted order. Shuffle is keyed by
(seed, layer_idx), so the same active set in the same layer partitions
reproducibly.
"""

from __future__ import annotations

import random
from typing import List

from .base import ClusterContext, ClusterMethod


class RandomCluster(ClusterMethod):
    def __init__(self, seed: int = 0):
        self.seed = seed

    def assign(self, ctx: ClusterContext, K: int) -> List[List[int]]:
        active = list(ctx.active)
        random.Random(self.seed + ctx.layer_idx).shuffle(active)
        m = len(active)
        k = min(K, m)
        return [active[j * m // k:(j + 1) * m // k] for j in range(k)]
