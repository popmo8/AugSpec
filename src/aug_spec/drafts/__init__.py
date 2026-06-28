"""Draft-strategy registry.

Add a new strategy:
  1. Drop a module exporting a `DraftStrategy` subclass.
  2. Register it in `_REGISTRY` with a stable string key (used in
     `configs/*.yaml`'s `draft.name` field).
"""

from __future__ import annotations

from typing import Any, Dict, Type

from .base import DraftStrategy, ScoreBasedAvgDraft
from .count import CountDraft, PrunedCountDraft
from .prefill_count import PrefillCountDraft
from .prefill_topm_count import PrefillTopMCountDraft
from .random_mask import RandomMaskDraft
from .softmax import SoftmaxDraft
from .specmoe import SpecMoeDraft
from .topm_count import TopMCountDraft
from .uniform import UniformDraft


_REGISTRY: Dict[str, Type[DraftStrategy]] = {
    "uniform": UniformDraft,
    "count": CountDraft,
    "pruned_count": PrunedCountDraft,
    "topm_count": TopMCountDraft,
    "prefill_count": PrefillCountDraft,
    "prefill_topm_count": PrefillTopMCountDraft,
    "softmax": SoftmaxDraft,
    "random_mask": RandomMaskDraft,
    "specmoe": SpecMoeDraft,
}


def get_draft_class(name: str) -> Type[DraftStrategy]:
    """Look up a draft class by registered name *without* instantiating it.

    Lets the CLI read class-level facts (holds_merged_residency /
    needs_count_top_k / needs_num_experts) before the draft args (and the
    model needed to fill them) are resolved."""
    if name not in _REGISTRY:
        known = ", ".join(sorted(_REGISTRY))
        raise KeyError(f"unknown draft {name!r}; known: {known}")
    return _REGISTRY[name]


def get_draft(name: str, **kwargs: Any) -> DraftStrategy:
    """Instantiate a draft strategy by registered name with **kwargs."""
    return get_draft_class(name)(**kwargs)


__all__ = [
    "DraftStrategy",
    "ScoreBasedAvgDraft",
    "UniformDraft",
    "CountDraft",
    "PrunedCountDraft",
    "TopMCountDraft",
    "PrefillCountDraft",
    "PrefillTopMCountDraft",
    "SoftmaxDraft",
    "RandomMaskDraft",
    "SpecMoeDraft",
    "get_draft",
    "get_draft_class",
]
