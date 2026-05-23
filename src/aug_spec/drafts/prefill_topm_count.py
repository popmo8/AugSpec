"""Prefill-only count-weighted averaged-expert draft with a top-M cutoff.

Cross of `PrefillCountDraft` and `TopMCountDraft`:

* Like ``prefill_count``: capture the target router's count distribution
  **once** during the prefill target forward; build the merged expert
  lazily on first draft access; never refresh again.
* Like ``topm_count``: before building, keep only the top-`M` experts
  by vote count (others zeroed, remainder renormalised). `M` defaults
  to the model's routing top-k (= ``count_top_k``).

The strongest offload assumption: the only experts ever read at
``build_weighted_avg`` time are the M experts with the largest mass in
the **prompt's** routing distribution, and we read them exactly once
per question.
"""

from __future__ import annotations

from typing import List, Optional

from .prefill_count import PrefillCountDraft
from .topm_count import keep_top_m


class PrefillTopMCountDraft(PrefillCountDraft):
    """Prefill-only count merge with a fixed top-M cutoff."""

    def __init__(self, count_top_k: int, M: Optional[int] = None):
        super().__init__(count_top_k=count_top_k)
        if M is not None and M < 1:
            raise ValueError(f"M must be >= 1, got {M!r}")
        self.M = M

    def _postprocess_weights(self, weights: List[float]) -> List[float]:
        m = self.M if self.M is not None else self.count_top_k
        return keep_top_m(weights, m)
