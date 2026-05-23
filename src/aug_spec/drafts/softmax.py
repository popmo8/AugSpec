"""Softmax-sum weighted averaged-expert draft."""

from __future__ import annotations

from .base import ScoreBasedAvgDraft


class SoftmaxDraft(ScoreBasedAvgDraft):
    """Per-cycle softmax-sum weighted averaged expert.

    For each expert, sum its softmax routing score across every captured
    position; normalise across experts to get per-expert weights.
    """

    history_value_kind = "float"

    def __init__(self, record_history: bool = False):
        super().__init__(record_history=record_history)

    def _score_vector_from_logits(self, router_logits):
        scores = router_logits.softmax(dim=-1)
        return scores.sum(dim=0).float()

    def _score_vector_from_softmax(self, softmax):
        return softmax.sum(dim=0).float()
