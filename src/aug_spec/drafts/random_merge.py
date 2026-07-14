"""Random balanced-merge draft (Table 1 "Random (merge)" baseline).

The "random" policy of the {random, static, dynamic} axis: ALL n experts
of every MoE layer are partitioned into K random balanced groups once per
run (seeded, in `prepare`), each group is merged with uniform weights
(1/|group|), and the K merged experts stay frozen for the whole run.
Routing goes through the shared gate-remap path (`_route_multi_expert`,
`indices` covering all n experts) — the same kernel the dynamic merge
drafts use, so only the partition policy and the freeze differ.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch

from aug_spec.merging import linear_merge

from .base import DraftStrategy


class RandomMergeDraft(DraftStrategy):
    """K random-group merged experts per layer, frozen for the run.

    Args:
        K: merged experts kept per layer (groups are balanced slices of a
            seeded shuffle of all n experts).
        draft_top_k: clusters activated per token in the draft forward.
            None → the model's native top-k (resolved in the adapter).
        seed: partition RNG seed — reproducible per (seed, layer order).
    """

    cache_kind = "averaged"

    # Caches K merged dense experts per layer, so the offload-merge engine
    # would have to reserve their VRAM (acceptance runs use hf, where this
    # is inert but keeps the budget math honest).
    holds_merged_residency = True

    def __init__(self, K: int = 1, draft_top_k: Optional[int] = None,
                 seed: int = 0):
        if K < 1:
            raise ValueError(f"K must be >= 1, got {K!r}")
        if draft_top_k is not None and draft_top_k < 1:
            raise ValueError(
                f"draft_top_k must be >= 1, got {draft_top_k!r}")
        self.K = K
        self.draft_top_k = draft_top_k
        self.seed = seed
        # layer_idx → frozen "multi" cache dict, built once in prepare().
        self._built: Dict[int, Dict[str, Any]] = {}

    def prepare(self, adapter, blocks) -> None:
        if self._built:
            return
        g = torch.Generator()
        g.manual_seed(self.seed)
        for li, block in blocks:
            n = adapter.num_experts(block)
            perm = torch.randperm(n, generator=g).tolist()
            k = min(self.K, n)
            groups = [sorted(perm[j * n // k:(j + 1) * n // k])
                      for j in range(k)]
            experts = []
            for group in groups:
                weights = [0.0] * n
                for i in group:
                    weights[i] = 1.0 / len(group)
                experts.append(linear_merge(adapter, block, group, weights))
            # Cluster mass = group size share (uniform prior — no routing
            # stats by construction). Only used for the descending order the
            # multi-cache contract expects; routing is per-token gate remap.
            masses = [len(group) / n for group in groups]
            order = sorted(range(k), key=lambda j: -masses[j])
            self._built[li] = {
                "kind":    "multi",
                "experts": [experts[j] for j in order],
                "weights": [masses[j] for j in order],
                "indices": [groups[j] for j in order],
            }

    def prepopulate(self, adapter, blocks, draft_cache):
        # Per-question reset clears draft_cache; re-insert the same frozen
        # dicts (the bmm-stack memo on them stays valid — weights never
        # change for the whole run).
        draft_cache.clear()
        draft_cache.update(self._built)

    def lazy_build(self, layer_idx, block, adapter):
        # The compile-warmup generate runs BEFORE any question, i.e. before
        # the first controller.reset()/prepopulate — the averaged forward
        # then asks lazy_build for the missing cache (same mechanism as
        # UniformDraft). Without this the C-BOOT fail-fast kills the warmup
        # (job 259444). prepare() has already built everything.
        return self._built.get(layer_idx)

    # refresh / capture: inherited no-ops — the partition is frozen.
