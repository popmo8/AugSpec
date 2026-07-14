"""Random expert-mask draft (Table 1 "Random (prune)" baseline).

Keeps `num_keep` uniformly random experts per layer. Default is the
"random" policy of the {random, static, dynamic} axis: the kept set is
drawn ONCE per run (in `prepare`, seeded) and stays fixed for every
question and cycle — only SpecMoE and the count-merge drafts are
dynamic. `per_cycle: true` restores the legacy behaviour (redraw the
mask every cycle), kept for the historical single-expert sweeps.
"""

from __future__ import annotations

from typing import Dict

import torch

from .base import DraftStrategy


class RandomMaskDraft(DraftStrategy):
    """`num_keep` random experts per layer as a boolean mask.

    Args:
        num_experts: experts per layer (auto-filled by the CLI when the
            config omits it).
        seed: RNG seed — masks are reproducible per (seed, layer order).
        num_keep: how many experts stay active per layer (default 1).
        per_cycle: legacy mode — redraw every cycle instead of drawing
            once per run.
    """

    cache_kind = "masked"

    # Needs the layer's expert count auto-filled into draft args when the
    # config omits num_experts.
    needs_num_experts = True

    def __init__(self, num_experts: int, seed: int, num_keep: int = 1,
                 per_cycle: bool = False):
        if not 1 <= num_keep <= num_experts:
            raise ValueError(
                f"num_keep must be in [1, {num_experts}], got {num_keep!r}")
        self.num_experts = num_experts
        self.num_keep = num_keep
        self.per_cycle = per_cycle
        self.seed = seed
        self.generator = torch.Generator()
        self.generator.manual_seed(seed)
        # layer_idx → fixed mask (static mode; drawn once in prepare()).
        self._masks: Dict[int, torch.Tensor] = {}

    def _random_mask(self) -> torch.Tensor:
        kept = torch.randperm(self.num_experts,
                              generator=self.generator)[:self.num_keep]
        m = torch.zeros(self.num_experts, dtype=torch.bool)
        m[kept] = True
        return m

    def prepare(self, adapter, blocks) -> None:
        # Static (default): one draw per run. The draw must happen HERE —
        # prepopulate() runs on every controller.reset(), i.e. per question,
        # and drawing there would silently turn the baseline per-question.
        if self.per_cycle or self._masks:
            return
        for li, _ in blocks:
            self._masks[li] = self._random_mask()

    def prepopulate(self, adapter, blocks, draft_cache):
        draft_cache.clear()
        for li, _ in blocks:
            draft_cache[li] = (self._random_mask() if self.per_cycle
                               else self._masks[li])

    def refresh(self, adapter, blocks, draft_cache):
        if not self.per_cycle:
            return          # static: the run-fixed masks stay as-is
        for li, _ in blocks:
            draft_cache[li] = self._random_mask()
