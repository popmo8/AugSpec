"""Draft strategy ABC + shared machinery for score-based averaging drafts.

A *draft strategy* decides what the draft side uses each cycle:

  cache_kind = "averaged"  → `controller.draft_cache[layer_idx]` holds a
                             Dict[str, Tensor] (the averaged-expert
                             weights for that layer)
  cache_kind = "masked"    → `controller.draft_cache[layer_idx]` holds a
                             1-D bool Tensor (one-hot expert mask)

The Controller calls these in this order per question:
    draft.reset()
    draft.prepopulate(adapter, blocks, draft_cache)    [optional]
    [per cycle:]
      target forward → draft.capture(layer_idx, router_logits)
      after verify   → draft.refresh(adapter, blocks, draft_cache)
    [draft forward may also call draft.lazy_build for "averaged"
     strategies that haven't been refreshed yet — UniformDraft uses this]
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch


class DraftStrategy:
    """Abstract draft strategy. Override only what you need."""

    cache_kind: str = "averaged"  # or "masked"

    def reset(self) -> None:
        pass

    def prepopulate(self, adapter, blocks, draft_cache: Dict[int, Any]) -> None:
        pass

    def capture(self, layer_idx: int, router_logits: torch.Tensor) -> None:
        pass

    def capture_softmax(self, layer_idx: int, softmax: torch.Tensor) -> None:
        """GPT-OSS path passes pre-computed softmax to avoid recomputing it."""
        pass

    def refresh(self, adapter, blocks, draft_cache: Dict[int, Any]) -> None:
        pass

    def lazy_build(self, layer_idx: int, block, adapter):
        return None


# =========================================================================
# Shared base: per-cycle weighted average from captured target-side scores
# =========================================================================

class ScoreBasedAvgDraft(DraftStrategy):
    """Common machinery for count / softmax weighted averages.

    Subclasses override `_score_vector_from_logits(router_logits)` (or
    `_score_vector_from_softmax(softmax)` for GPT-OSS), returning a
    `[num_experts]` fp32 tensor that gets normalised and used as
    per-expert weights for `adapter.build_weighted_avg`.
    """

    cache_kind = "averaged"

    # History encoding for `expert_weights_history.json`. Count-style
    # scorers produce integer-valued tensors; subclasses with continuous
    # scores override to "float". Only consulted when record_history=True.
    history_value_kind: str = "float"

    def __init__(self, record_history: bool = False):
        # layer_idx → CPU fp32 tensor [num_experts]
        self.target_score: Dict[int, torch.Tensor] = {}

        # Optional per-cycle history of the raw target-side score vector
        # for each MoE layer, captured BEFORE normalisation. Populated
        # only when record_history=True; exp scripts dump it to JSON for
        # offline statistical analysis (per-cycle per-layer per-expert
        # values). The CLI tags qid/category/cycle telemetry on it from
        # on_cycle.
        self.record_history: bool = record_history
        self.history: List[Dict[str, Any]] = []
        self._cycle_in_question: int = -1

    # --- history helpers ------------------------------------------------
    def _encode_score_vec(self, score_vec: torch.Tensor) -> List[Any]:
        if self.history_value_kind == "int":
            return [int(v) for v in score_vec.tolist()]
        return [float(v) for v in score_vec.tolist()]

    def _snapshot_history(self, layer_order: List[int]) -> None:
        """Append one per-cycle snapshot to `self.history`. `layer_order`
        is the canonical MoE-layer ordering (used as row index), so the
        resulting 2-D array is consistently shaped across cycles."""
        scores: List[List[Any]] = []
        for li in layer_order:
            vec = self.target_score.get(li)
            if vec is None:
                scores.append([])
            else:
                scores.append(self._encode_score_vec(vec))
        self.history.append({
            "cycle_idx_in_q": self._cycle_in_question,
            "scores": scores,
        })

    def make_on_cycle_tagger(self) -> Optional[Any]:
        """Build an `on_cycle_extra(qres, cs)` callback that tags the most
        recent history snapshot with question / cycle metadata.

        Returns None when `record_history=False` so callers can pass the
        result through unconditionally.
        """
        if not self.record_history:
            return None

        def tag(qres, cs):
            if not self.history:
                return
            snap = self.history[-1]
            snap["question_id"] = qres.qid
            snap["category"] = qres.category
            snap["actual_T"] = int(cs.actual_T)
            snap["num_matches"] = int(cs.num_matches)
        return tag

    def export_history(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """JSON-friendly dict bundling history + run metadata. `metadata`
        is shallow-copied; `value_kind` / `cycles` are overwritten so the
        schema is self-describing."""
        out = dict(metadata)
        out.setdefault("schema_version", 1)
        out["value_kind"] = self.history_value_kind
        out["cycles"] = self.history
        return out

    # --- subclass hooks -------------------------------------------------
    def _score_vector_from_logits(
        self, router_logits: torch.Tensor,
    ) -> torch.Tensor:
        raise NotImplementedError

    def _score_vector_from_softmax(
        self, softmax: torch.Tensor,
    ) -> torch.Tensor:
        raise NotImplementedError

    # --- DraftStrategy overrides ---------------------------------------
    def reset(self):
        self.target_score.clear()
        self._cycle_in_question = -1

    def capture(self, layer_idx, router_logits):
        score_vec = self._score_vector_from_logits(router_logits)
        self.target_score[layer_idx] = score_vec.float().detach().cpu()

    def capture_softmax(self, layer_idx, softmax):
        score_vec = self._score_vector_from_softmax(softmax)
        self.target_score[layer_idx] = score_vec.float().detach().cpu()

    def refresh(self, adapter, blocks, draft_cache):
        if not self.target_score:
            return
        layer_to_block = dict(blocks)
        if self.record_history:
            self._cycle_in_question += 1
            self._snapshot_history([li for li, _ in blocks])
        for li, score_vec in self.target_score.items():
            # Drop old cache for this layer first so peak transient VRAM
            # during the rebuild stays bounded by one layer's worth.
            draft_cache.pop(li, None)
            block = layer_to_block[li]
            n = adapter.num_experts(block)
            total = float(score_vec.sum().item())
            if total <= 0:
                weights = [1.0 / n] * n
            else:
                weights = (score_vec.float() / total).tolist()
            weights = self._postprocess_weights(weights)
            draft_cache[li] = adapter.build_weighted_avg(block, weights)

    def _postprocess_weights(self, weights: List[float]) -> List[float]:
        """Hook called once per layer per cycle, right before
        `adapter.build_weighted_avg`. Subclasses override to e.g. prune
        low-mass experts. Default = identity passthrough."""
        return weights
