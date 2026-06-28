"""Random single-expert mask draft (baseline)."""

from __future__ import annotations

import torch

from .base import DraftStrategy


class RandomMaskDraft(DraftStrategy):
    """One uniformly random expert per layer, refreshed every cycle.

    Pre-populated on `reset()` so the FIRST cycle of every question is
    already random (no top-k warmup).
    """

    cache_kind = "masked"

    # Needs the layer's expert count auto-filled into draft args when the
    # config omits num_experts.
    needs_num_experts = True

    def __init__(self, num_experts: int, seed: int):
        self.num_experts = num_experts
        self.seed = seed
        self.generator = torch.Generator()
        self.generator.manual_seed(seed)

    def _random_mask(self) -> torch.Tensor:
        e = int(torch.randint(
            0, self.num_experts, (1,), generator=self.generator).item())
        m = torch.zeros(self.num_experts, dtype=torch.bool)
        m[e] = True
        return m

    def prepopulate(self, adapter, blocks, draft_cache):
        draft_cache.clear()
        for li, _ in blocks:
            draft_cache[li] = self._random_mask()

    def refresh(self, adapter, blocks, draft_cache):
        for li, _ in blocks:
            draft_cache[li] = self._random_mask()
