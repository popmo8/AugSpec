"""Static kept-mask draft (Table 1 "Enumerate"/NAEE baseline, prune–static).

Loads an offline-searched kept-set spec (`scripts/search_naee.py` → one
json per model) and freezes each layer's boolean expert mask for the whole
run. The draft forward is the existing masked forward (-inf on non-kept
logits → native top-k within the kept set) — identical mechanics to
`random_mask`, only the (offline, reconstruction-loss-optimal) set differs.

Spec schema (json):
    {"model_id": ..., "r": 16,
     "layers": {"<layer_idx>": {"kept": [expert ids], "loss": float}}}
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import torch

from .base import DraftStrategy


class StaticMaskDraft(DraftStrategy):
    """Frozen per-layer kept-expert mask from an NAEE search spec.

    Args:
        spec_path: json produced by `scripts/search_naee.py`.
    """

    cache_kind = "masked"

    def __init__(self, spec_path: str):
        self.spec_path = Path(spec_path)
        # layer_idx → frozen bool mask, built once in prepare().
        self._masks: Dict[int, torch.Tensor] = {}

    def prepare(self, adapter, blocks) -> None:
        if self._masks:
            return
        spec = json.loads(self.spec_path.read_text())
        layers = spec["layers"]
        for li, block in blocks:
            entry = layers.get(str(li))
            if entry is None:
                raise ValueError(
                    f"static_mask: spec {self.spec_path} has no entry for "
                    f"MoE layer {li} — was it built for this model?")
            n = adapter.num_experts(block)
            kept = sorted(int(i) for i in entry["kept"])
            if (not kept or kept[0] < 0 or kept[-1] >= n
                    or len(set(kept)) != len(kept)):
                raise ValueError(
                    f"static_mask: layer {li} kept set invalid for n={n}: "
                    f"{kept}")
            m = torch.zeros(n, dtype=torch.bool)
            m[torch.tensor(kept)] = True
            self._masks[li] = m

    def prepopulate(self, adapter, blocks, draft_cache):
        draft_cache.clear()
        for li, _ in blocks:
            draft_cache[li] = self._masks[li]

    # refresh: inherited no-op — the kept sets are frozen. (The masked
    # forward silently routes unmasked during the compile warmup, before
    # the first prepopulate — untimed, harmless.)
