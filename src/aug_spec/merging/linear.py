"""The single linear weighted-average merge entry.

Converges the two existing merge call sites — the offload-merge engine's
`engine.build` (when one is attached to the block) and the adapter's
`build_weighted_avg` — behind one function, so a cache (B3) can wrap exactly
here. Behaviour-preserving: `_build_one` already routed through this same
two-way branch (the SVD path was removed earlier).
"""

from __future__ import annotations

from typing import Any, Dict, List

import torch


def linear_merge(adapter, block, member_ids: List[int],
                 weights: List[float]) -> Dict[str, torch.Tensor]:
    """Linear weighted average of `block`'s experts into one dense expert.

    `weights` is the full per-expert weight vector (length = num_experts);
    `member_ids` lists the experts with non-zero weight — i.e. the merge's
    members. `member_ids` is redundant with the non-zero entries of `weights`
    but is passed explicitly so a cache (B3) can key on the member set without
    re-deriving it. Routes through the offload-merge engine when the block
    carries one, else the adapter's CPU merge.
    """
    engine = getattr(block, "_merge_engine", None)
    if engine is not None:
        return engine.build(block, weights)
    return adapter.build_weighted_avg(block, weights)
