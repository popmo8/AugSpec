"""Count-weighted averaged-expert drafts (with optional pruning)."""

from __future__ import annotations

from typing import List

from aug_spec.runtime.scorers import make_count_scorer

from .base import ScoreBasedAvgDraft


class CountDraft(ScoreBasedAvgDraft):
    """Per-cycle count-weighted averaged expert.

    For each captured forward pass, each expert's weight is the number of
    positions where it landed in the router's top-k.
    """

    history_value_kind = "int"

    def __init__(self, count_top_k: int, record_history: bool = False):
        super().__init__(record_history=record_history)
        self.count_top_k = count_top_k
        self._scorer = make_count_scorer(count_top_k=count_top_k)

    def _score_vector_from_logits(self, router_logits):
        scores = router_logits.softmax(dim=-1)
        return self._scorer(scores)

    def _score_vector_from_softmax(self, softmax):
        return self._scorer(softmax)


class PrunedCountDraft(CountDraft):
    """Count-weighted averaged expert with cumulative-frequency pruning.

    Before each per-cycle weighted average is built, low-frequency
    experts are dropped:

        sort experts by normalised count desc;
        walk the sorted list summing weights;
        the smallest prefix that reaches `cumulative_threshold` is KEPT;
        experts outside that prefix have their weight zeroed;
        renormalise the kept weights to sum to 1.

    With `cumulative_threshold=0.9` (default), the result averages over
    the experts that explain the top 90% of activation mass for this
    cycle and ignores the long tail. Motivation: on imbalanced routers
    (gpt-oss-20b, Qwen3) the bottom ~half of experts contribute <1% each
    — pulling them into the average mostly injects noise.

    Edge cases:
      * `cumulative_threshold == 1.0` → no pruning (= plain CountDraft).
      * single expert already ≥ threshold → only that expert is kept.
      * all-zero count vector → base class falls back to uniform before
        we see it; pruning then keeps experts up to the threshold, which
        is still uniform over the kept subset.
    """

    def __init__(self, count_top_k: int,
                 cumulative_threshold: float = 0.9,
                 record_history: bool = False):
        super().__init__(count_top_k=count_top_k,
                         record_history=record_history)
        if not (0.0 < cumulative_threshold <= 1.0):
            raise ValueError(
                f"cumulative_threshold must be in (0, 1], got "
                f"{cumulative_threshold!r}")
        self.cumulative_threshold = float(cumulative_threshold)

    def _postprocess_weights(self, weights: List[float]) -> List[float]:
        threshold = self.cumulative_threshold
        if threshold >= 1.0:
            return weights

        n = len(weights)
        # Sort by weight desc; break ties by ascending index (deterministic).
        order = sorted(range(n), key=lambda i: (-weights[i], i))
        kept = [False] * n
        cum = 0.0
        for idx in order:
            kept[idx] = True
            cum += weights[idx]
            if cum >= threshold:
                break

        pruned = [w if k else 0.0 for w, k in zip(weights, kept)]
        s = sum(pruned)
        if s <= 0.0:
            # Shouldn't trigger — the top-weight expert is always kept.
            return [1.0 / n] * n
        return [w / s for w in pruned]
