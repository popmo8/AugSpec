"""Generic speculative-decoding controller wiring adapter × draft strategy.

One controller class drives every (model family, draft strategy) pair.
The pair (`adapter`, `draft`) determines:

  * adapter — what model family + how to swap MoE-block forwards
  * draft   — what goes into `controller.draft_cache` (averaged-expert
              dicts or per-layer boolean masks), how it's refreshed per
              cycle, what is captured target-side

The surface exposed — `install`, `uninstall`, `reset`, `update_masks`,
`in_draft_phase`, `cycle_misses` — matches what
`aug_spec.runtime.phase.specbench_callbacks` and
`shared_model_phase_patch` expect.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import torch.nn as nn

from aug_spec.adapters import MoEAdapter
from aug_spec.drafts import DraftStrategy


class Controller:
    """Wires an adapter and a draft strategy onto a model.

    `draft_cache` is the per-layer state used in draft phase:
      * "averaged" drafts store a `Dict[str, Tensor]` (the averaged expert)
      * "masked"   drafts store a 1-D bool Tensor (the one-hot mask)
    """

    def __init__(self, model: nn.Module,
                 adapter: MoEAdapter,
                 draft: DraftStrategy):
        self.model = model
        self.adapter = adapter
        self.draft = draft

        self.blocks: List[Tuple[int, nn.Module]] = list(adapter.iter_moe(model))
        if not self.blocks:
            raise RuntimeError(
                f"No MoE layers found for adapter '{adapter.name}'.")
        self.num_moe_layers = len(self.blocks)

        self.draft_cache: Dict[int, Any] = {}
        self.update_count: int = 0
        self.cycle_misses: List[int] = []  # populated by drafts that track it

        self.in_draft_phase: bool = False
        self._installed: bool = False

    # ── install / uninstall ────────────────────────────────────────────
    def install(self) -> None:
        if self._installed:
            return
        if self.draft.cache_kind == "averaged":
            factory = self.adapter.make_averaged_forward
        elif self.draft.cache_kind == "masked":
            factory = self.adapter.make_masked_forward
        else:
            raise ValueError(
                f"Unknown draft.cache_kind {self.draft.cache_kind!r}")

        for layer_idx, block in self.blocks:
            if hasattr(block, "_aug_spec_orig_forward"):
                block.forward = block._aug_spec_orig_forward
                del block._aug_spec_orig_forward
            block._aug_spec_orig_forward = block.forward
            fn = factory(self, layer_idx, block)
            block.forward = (
                lambda hidden_states, *a, _fn=fn, _block=block, **kw:
                _fn(_block, hidden_states))
        self._installed = True

    def uninstall(self) -> None:
        for _, block in self.blocks:
            if hasattr(block, "_aug_spec_orig_forward"):
                block.forward = block._aug_spec_orig_forward
                del block._aug_spec_orig_forward
        self._installed = False

    # ── specbench callback surface ─────────────────────────────────────
    def update_masks(self) -> None:
        """`on_cycle` hook: refresh the draft cache from the latest
        target-side capture (whatever the draft stores)."""
        self.draft.refresh(self.adapter, self.blocks, self.draft_cache)
        self.update_count += 1

    def reset(self) -> None:
        """`on_question_start` hook: per-question clear. Forward
        replacements stay installed. Does NOT touch `in_draft_phase`
        (controlled by the phase patch).
        """
        self.draft.reset()
        self.draft_cache.clear()
        self.update_count = 0
        self.draft.prepopulate(self.adapter, self.blocks, self.draft_cache)
