"""Top-M count-weighted averaged-expert draft.

Same per-cycle count machinery as `CountDraft`, but before each rebuild
keeps only the **top M experts by count** (everything else zeroed and
the rest renormalised). Compared with `PrunedCountDraft`, the budget is
a fixed integer M instead of a cumulative-mass threshold — useful when
you want a deterministic upper bound on how many experts contribute to
the merged dense expert each cycle.

`M` defaults to `count_top_k`, i.e. the model's native routing top-k.
That matches "merge only what the router itself would have picked per
token", and the corresponding offload-backend property: the experts
touched at refresh time form a subset of the experts the most recent
verify forward already fetched.
"""

from __future__ import annotations

from typing import List, Optional

from .count import CountDraft


def keep_top_m(weights: List[float], m: int) -> List[float]:
    """Keep the top-`m` entries of `weights` by descending value, zero the
    rest, and renormalise. Ties are broken by ascending index for
    determinism. If `m >= len(weights)` the input is returned unchanged.

    Module-level so that other draft variants (e.g. PrefillTopMCount)
    can reuse the same cutoff logic without inheriting from this class.
    """
    n = len(weights)
    if m >= n:
        return weights
    order = sorted(range(n), key=lambda i: (-weights[i], i))
    kept = set(order[:m])
    pruned = [w if i in kept else 0.0 for i, w in enumerate(weights)]
    s = sum(pruned)
    if s <= 0.0:
        return [1.0 / n] * n
    return [w / s for w in pruned]


class TopMCountDraft(CountDraft):
    """Count-weighted averaged expert with a fixed top-M cutoff.

    Args:
        count_top_k: per-position top-k for the underlying count scorer
            (defaults to adapter.default_count_top_k(model) at CLI level).
        M: number of experts kept after sorting by count desc. ``None``
            means "use count_top_k" — typically the value you want.
        record_history: dump raw per-cycle count vectors to
            ``expert_weights_history.json`` for offline analysis.
    """

    def __init__(self, count_top_k: int,
                 M: Optional[int] = None,
                 record_history: bool = False):
        super().__init__(count_top_k=count_top_k,
                         record_history=record_history)
        if M is not None and M < 1:
            raise ValueError(f"M must be >= 1, got {M!r}")
        self.M = M

    def _postprocess_weights(self, weights: List[float]) -> List[float]:
        m = self.M if self.M is not None else self.count_top_k
        return keep_top_m(weights, m)
