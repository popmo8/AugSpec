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

import os
from typing import Any, Dict, List, Optional

import torch


class DraftStrategy:
    """Abstract draft strategy. Override only what you need."""

    cache_kind: str = "averaged"  # or "masked" / "substitute"

    def prepare(self, adapter, blocks) -> None:
        """One-time setup before any inference (called once by the CLI after
        the controller is built). Use for static, model-derived precomputation
        that must not repeat per question — e.g. pairwise expert distances.
        Default is a no-op."""
        pass

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
    per-expert weights for the chosen merge method.

    Set `K>1` to keep K merged experts per layer (a mini-MoE) instead of a
    single dense expert. Active experts are partitioned into K clusters and
    each cluster is merged independently; the draft forward then runs the
    `draft_top_k` heaviest clusters and combines their outputs. This avoids
    collapsing a large, diverse expert pool (e.g. Qwen3's 128) into one
    expert. `K=1` reproduces the single-merged-expert behaviour exactly.
    """

    cache_kind = "averaged"

    # History encoding for `expert_weights_history.json`. Count-style
    # scorers produce integer-valued tensors; subclasses with continuous
    # scores override to "float". Only consulted when record_history=True.
    history_value_kind: str = "float"

    def __init__(self, record_history: bool = False,
                 K: int = 1,
                 draft_top_k: Optional[int] = None):
        # layer_idx → CPU fp32 tensor [num_experts]
        self.target_score: Dict[int, torch.Tensor] = {}

        # K: number of merged experts cached per layer.
        #   K == 1 → single dense expert (a plain Dict[str, Tensor] cache).
        #   K  > 1 → K cluster-merged experts (a "multi" cache dict).
        if K < 1:
            raise ValueError(f"K must be >= 1, got {K!r}")
        self.K = K

        # draft_top_k: how many of the K clusters the draft forward actually
        #   runs. None → resolved to the model's native top-k inside each
        #   adapter forward. Ignored when K == 1.
        if draft_top_k is not None and draft_top_k < 1:
            raise ValueError(f"draft_top_k must be >= 1, got {draft_top_k!r}")
        self.draft_top_k = draft_top_k

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
        # merge_during_verify (verify_merge_plan.md P1): the offload-merge engine
        # already built draft_cache[li] per layer *during* verify (on_verify_layer,
        # experts still resident → 0 re-fetch). Skip the after-verify rebuild;
        # telemetry above still runs. No engine / during_verify=False → unchanged.
        engine = getattr(blocks[0][1], "_merge_engine", None) if blocks else None
        if engine is not None and getattr(engine, "during_verify", False):
            return
        for li, score_vec in self.target_score.items():
            self._refresh_layer(adapter, layer_to_block[li], li, score_vec,
                                draft_cache)

    def _refresh_layer(self, adapter, block, li, score_vec, draft_cache):
        """Build (or rebuild) the merged draft for one layer from its captured
        count vector. Shared by `refresh` (after-verify, all layers) and the
        offload-merge engine's `on_verify_layer` (during-verify, one layer)."""
        # Drop old cache for this layer first so peak transient VRAM during the
        # rebuild stays bounded by one layer's worth.
        draft_cache.pop(li, None)
        n = adapter.num_experts(block)
        total = float(score_vec.sum().item())
        if total <= 0:
            weights = [1.0 / n] * n
        else:
            weights = (score_vec.float() / total).tolist()
        weights = self._postprocess_weights(weights)
        if self.K > 1:
            draft_cache[li] = self._cluster_and_build(adapter, block, weights)
        else:
            draft_cache[li] = self._build_one(adapter, block, weights)

    def _build_one(self, adapter, block,
                   weights: List[float]) -> Dict[str, torch.Tensor]:
        """Merge `weights` into a single expert by linear weighted average.

        The offload-merge engine owns the merge when merge_offload built one
        (tagged onto the block by the controller); otherwise the plain adapter
        merge. The engine's build delegates to build_weighted_avg, so this is
        behaviour-preserving.
        """
        engine = getattr(block, "_merge_engine", None)
        if engine is not None:
            return engine.build(block, weights)
        return adapter.build_weighted_avg(block, weights)

    def _cluster_and_build(self, adapter, block,
                           weights: List[float]) -> Dict[str, Any]:
        """Partition active experts into K clusters and merge each one.

        Clustering (frequency-slice): active experts are sorted by weight
        descending and cut into K contiguous slices, so each cluster groups
        experts of comparable activation frequency. This is a fast,
        calibration-free proxy — a stronger functional-similarity metric
        (e.g. gate-vector clustering) can replace `_assign_clusters` later
        without touching the rest of the pipeline.

        Returns a "multi" cache dict consumed by `adapter._route_multi_expert`::

            {
              "kind":     "multi",
              "experts":  [expert_0, ..., expert_{K'-1}],  # K' = min(K, #active)
              "weights":  [w_0,      ..., w_{K'-1}],         # cluster mass, sum=1
              "indices":  [[orig expert ids in cluster 0], ...],  # for gate remap
            }

        Clusters are ordered by descending mass. `indices[k]` lists the
        original expert ids in cluster k, so the draft forward can remap a
        token's gate probabilities onto the K clusters.
        """
        n = len(weights)
        active = [i for i, w in enumerate(weights) if w > 0.0]
        groups = self._assign_clusters(active, weights)

        # AUG_CLUSTER_UNIFORM experiment: merge each cluster with EQUAL weights
        # (1/|group|) instead of frequency-proportional ones. The slicing
        # (_assign_clusters) and the cross-cluster mass (`masses`) stay
        # frequency-based — only the within-cluster combine changes.
        uniform_merge = os.environ.get("AUG_CLUSTER_UNIFORM") is not None

        experts: List[Dict[str, torch.Tensor]] = []
        masses: List[float] = []
        for group in groups:
            group_mass = sum(weights[i] for i in group)
            # Renormalise within the cluster so each merge gets weights summing
            # to 1; the cluster's share of the whole is tracked in `masses`.
            cluster_weights = [0.0] * n
            if uniform_merge:
                w_each = 1.0 / len(group)
                for i in group:
                    cluster_weights[i] = w_each
            else:
                for i in group:
                    cluster_weights[i] = weights[i] / group_mass
            experts.append(
                self._build_one(adapter, block, cluster_weights))
            masses.append(group_mass)

        total = sum(masses)
        order = sorted(range(len(experts)), key=lambda k: -masses[k])
        return {
            "kind":    "multi",
            "experts": [experts[k] for k in order],
            "weights": [masses[k] / total for k in order],
            "indices": [groups[k] for k in order],
        }

    def _assign_clusters(self, active: List[int],
                         weights: List[float]) -> List[List[int]]:
        """Group active expert indices into at most K clusters.

        Frequency-slice strategy: sort by weight descending, then cut into K
        contiguous, near-equal-size slices. Returns a list of non-empty
        index groups (fewer than K only when active experts < K).
        """
        ordered = sorted(active, key=lambda i: -weights[i])
        m = len(ordered)
        k = min(self.K, m)
        return [ordered[j * m // k:(j + 1) * m // k] for j in range(k)]

    def _postprocess_weights(self, weights: List[float]) -> List[float]:
        """Hook called once per layer per cycle, right before
        `adapter.build_weighted_avg`. Subclasses override to e.g. prune
        low-mass experts. Default = identity passthrough."""
        return weights
