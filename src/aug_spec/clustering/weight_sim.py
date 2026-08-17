"""Weight-similarity pair clustering (static expert-weight similarity).

Same greedy max-value, max-size-2 pairing as cooccur_pair / activation_similarity,
but the table is the pairwise similarity of the experts' WEIGHTS — a static
property of the model. So it is computed ONCE in `prepare` (no per-cycle capture,
no prefill accumulation) and optionally cached to disk.

`metric`:
  "cosine" — cosine similarity of flattened expert weights (high = similar).
  "l2"     — NEGATIVE L2 distance (so greedy's max = min distance = most similar).
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional

import torch

from .base import ClusterContext, ClusterMethod, greedy_group


def weight_sim_table(flats: List[torch.Tensor], metric: str) -> torch.Tensor:
    """[n, n] pairwise weight similarity from per-expert flattened weights.
    cosine → u·uᵀ; l2 → −cdist (negated so larger = more similar)."""
    S = torch.stack([f.float().flatten() for f in flats])      # [n, D]
    if metric == "cosine":
        u = S / S.norm(dim=1, keepdim=True).clamp_min(1e-8)
        return (u @ u.t()).cpu()
    return (-torch.cdist(S.unsqueeze(0), S.unsqueeze(0)).squeeze(0)).cpu()


class WeightSimCluster(ClusterMethod):
    def __init__(self, metric: str = "cosine", cache: Optional[str] = None):
        if metric not in ("cosine", "l2"):
            raise ValueError(f"weight_similarity metric must be cosine|l2, "
                             f"got {metric!r}")
        self.metric = metric
        self.cache = cache
        self.tables: Dict[int, torch.Tensor] = {}

    def prepare(self, adapter, blocks) -> None:
        """Compute the per-layer weight-similarity tables ONCE (static). Loads
        from `cache` if present, else computes from the experts' weights (the
        CPU source on offload) and saves. Reused unchanged for the whole run."""
        if self.tables:
            return
        if self.cache and os.path.exists(self.cache):
            self.tables = torch.load(self.cache)
            return
        for li, block in blocks:
            src = getattr(block, "_cpu_merge_source", block)
            self.tables[li] = weight_sim_table(
                adapter.expert_flat_weights(src), self.metric)
        if self.cache:
            os.makedirs(os.path.dirname(self.cache) or ".", exist_ok=True)
            torch.save(self.tables, self.cache)

    def assign(self, ctx: ClusterContext, K: int) -> List[List[int]]:
        return greedy_group(self.tables.get(ctx.layer_idx), list(ctx.active), K, ctx.group_cap)
