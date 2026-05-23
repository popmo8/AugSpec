"""Prefill-only count-weighted averaged-expert draft.

Captures the target router's count distribution ONCE — during the
prefill target forward at the start of each question — and never
refreshes the merged expert again for the rest of that question.

This is the most aggressive offload-friendly merge variant: the only
moment we read expert weights to build the merged expert is right
after prefill, and the merged expert is then reused unchanged for
every speculative-decoding cycle.

Behaviour summary
-----------------
* On `reset()` (question start): `target_score` and `draft_cache` are
  cleared.
* During the prefill target forward: `capture()` populates
  `target_score[layer_idx]` for every layer.
* During the first draft forward (per layer): `lazy_build()` reads the
  captured prefill score, builds the merged expert, caches it.
* Subsequent verify forwards still call `capture()` but they are
  no-ops here (we keep the prefill capture intact).
* Per-cycle `refresh()` is also a no-op.

Caveat: this draft does not benefit from temporal-locality updates to
the routing distribution. It is the simplest possible merge baseline
and the cheapest to run on offload. Compare against `count` /
`topm_count` to see how much per-cycle adaptation buys you.
"""

from __future__ import annotations

from .count import CountDraft


class PrefillCountDraft(CountDraft):
    """Count-weighted merged expert, frozen after the prefill capture."""

    def __init__(self, count_top_k: int):
        # record_history is not supported here — only one effective
        # state per question, no per-cycle evolution to log.
        super().__init__(count_top_k=count_top_k, record_history=False)

    def capture(self, layer_idx, router_logits):
        # Keep only the FIRST capture per layer (= prefill target forward).
        # Verify-cycle captures are dropped on the floor.
        if layer_idx in self.target_score:
            return
        super().capture(layer_idx, router_logits)

    def capture_softmax(self, layer_idx, softmax):
        # Same single-shot semantics for the GPT-OSS softmax path.
        if layer_idx in self.target_score:
            return
        super().capture_softmax(layer_idx, softmax)

    def refresh(self, adapter, blocks, draft_cache):
        # No per-cycle refresh — the merged expert is built lazily once
        # by `lazy_build` on the first draft forward per layer.
        return

    def lazy_build(self, layer_idx, block, adapter):
        score_vec = self.target_score.get(layer_idx)
        if score_vec is None:
            # Prefill hasn't captured this layer yet. Shouldn't normally
            # happen — the very first target forward (prefill) traverses
            # every MoE block — but degrade gracefully and let the
            # adapter fall through to standard routing for now.
            return None
        n = adapter.num_experts(block)
        total = float(score_vec.sum().item())
        if total <= 0:
            weights = [1.0 / n] * n
        else:
            weights = (score_vec.float() / total).tolist()
        # Same hook as ScoreBasedAvgDraft.refresh — subclasses can prune.
        weights = self._postprocess_weights(weights)
        return adapter.build_weighted_avg(block, weights)
