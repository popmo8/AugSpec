"""Uniform-averaged draft: fixed 1/n weights, built lazily once per layer."""

from __future__ import annotations

from .base import DraftStrategy


class UniformDraft(DraftStrategy):
    """1/n averaged expert, built lazily on first draft-phase use.

    No capture, no per-cycle refresh — the result is fixed for the whole
    run. Memory note: keeping the build lazy means the transient fp32
    accumulator is only ever held for one layer at a time, not all
    layers simultaneously.
    """

    cache_kind = "averaged"

    # Builds a merged dense expert (the 1/n average) like the score-based
    # family, so it also reserves merged residency — even though it is not a
    # ScoreBasedAvgDraft subclass.
    holds_merged_residency = True

    def lazy_build(self, layer_idx, block, adapter):
        n = adapter.num_experts(block)
        return adapter.build_weighted_avg(block, [1.0 / n] * n)
