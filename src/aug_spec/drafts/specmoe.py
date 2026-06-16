"""SpecMoE draft: top-k routing with L2-nearest expert substitution.

Port of `thesis_experiment/.../exp_mixtral_specmoe_specbench.py` into the
adapter × draft architecture.

Mechanism (cache_kind = "substitute"):

  * Both phases route the natural top-`route_top_k` experts (overriding the
    model's native top-k).
  * TARGET phase routes those winners directly and captures the per-position
    router softmax (count scorer) to refresh the kept-expert mask.
  * DRAFT phase remaps each winner through a per-layer substitute table:
    a winner inside the kept mask maps to itself; a winner outside maps to
    its L2-nearest kept neighbour. The table lives in `draft_cache[li]`.

Per verify cycle the kept mask (top-N by count) and the substitute table are
rebuilt. Pairwise expert L2 distances are static, so they are computed ONCE
up front in `prepare` and never recomputed.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import torch

from aug_spec.adapters.base import _pairwise_l2
from aug_spec.runtime.scorers import make_count_scorer

from .base import DraftStrategy


class SpecMoeDraft(DraftStrategy):
    """SpecMoE single-model draft (cache_kind="substitute").

    Args:
        N: kept-mask size — the top-N most-voted experts stay active in the
            draft each cycle. Must satisfy 1 <= N <= num_experts.
        route_top_k: experts routed per token in both phases (replaces the
            model's native top-k). Defaults to 1 (the original SpecMoE).
        count_top_k: per-position vote size for the count scorer that picks
            the top-N kept experts. Defaults to `route_top_k`.
    """

    cache_kind = "substitute"

    def __init__(self, N: int, route_top_k: int = 1,
                 count_top_k: Optional[int] = None):
        if N < 1:
            raise ValueError(f"N must be >= 1, got {N!r}")
        if route_top_k < 1:
            raise ValueError(f"route_top_k must be >= 1, got {route_top_k!r}")
        self.N = N
        self.route_top_k = route_top_k
        self.count_top_k = count_top_k if count_top_k is not None else route_top_k
        self._scorer = make_count_scorer(count_top_k=self.count_top_k)

        # layer_idx → CPU fp32 count vector captured in the target phase
        self.target_score: Dict[int, torch.Tensor] = {}
        # layer_idx → CPU bool kept-mask [num_experts] (telemetry + table build)
        self._draft_mask: Dict[int, torch.Tensor] = {}
        # layer_idx → CPU fp32 [num_experts, num_experts] L2 distances.
        # Static; built once in prepare(), never cleared by reset().
        self._expert_dist: Dict[int, torch.Tensor] = {}
        self.num_experts: int = 0

        # Run-level: one int per verify cycle = how many of the top-route_top_k
        # target experts fell outside the OLD mask, summed over layers.
        self.cycle_misses: List[int] = []

    # ── one-time setup ──────────────────────────────────────────────────
    def prepare(self, adapter, blocks) -> None:
        if self._expert_dist:
            return
        for li, block in blocks:
            # Offload: block.experts are placeholders — read real weights from
            # the CPU source the controller attached (hf: src is block itself).
            src = getattr(block, "_cpu_merge_source", block)
            self._expert_dist[li] = _pairwise_l2(
                adapter.expert_flat_weights(src))
        first = next(iter(self._expert_dist.values()))
        self.num_experts = first.shape[0]
        if self.N > self.num_experts:
            raise ValueError(
                f"N={self.N} must be <= num_experts={self.num_experts}")
        if self.route_top_k > self.num_experts:
            raise ValueError(
                f"route_top_k={self.route_top_k} must be <= "
                f"num_experts={self.num_experts}")

    # ── per-question reset ──────────────────────────────────────────────
    def reset(self) -> None:
        # Per-question state clears; distances and the run-level miss log
        # persist across questions.
        self.target_score.clear()
        self._draft_mask.clear()

    # ── target-phase capture (forward passes the full softmax) ──────────
    def capture(self, layer_idx: int, softmax: torch.Tensor) -> None:
        score_vec = self._scorer(softmax)
        self.target_score[layer_idx] = score_vec.float().detach().cpu()

    # ── per-cycle mask + substitute-table refresh ───────────────────────
    def refresh(self, adapter, blocks, draft_cache: Dict[int, torch.Tensor]) -> None:
        if not self.target_score:
            return

        # Miss telemetry against the OLD masks, before they are overwritten.
        cycle_miss = 0
        for li, score_vec in self.target_score.items():
            old_mask = self._draft_mask.get(li)
            if old_mask is None:
                continue
            top = torch.topk(score_vec, self.route_top_k).indices
            cycle_miss += int((~old_mask[top]).sum().item())
        self.cycle_misses.append(cycle_miss)

        for li, score_vec in self.target_score.items():
            top_n = torch.topk(score_vec, self.N).indices
            mask = torch.zeros(self.num_experts, dtype=torch.bool)
            mask[top_n] = True
            self._draft_mask[li] = mask
            draft_cache[li] = self._build_substitute_table(li, mask)

    def _build_substitute_table(self, layer_idx: int,
                                mask: torch.Tensor) -> torch.Tensor:
        """[num_experts] long: each expert → the kept expert it routes to.
        Kept experts map to themselves (distance diagonal is zero)."""
        D = self._expert_dist[layer_idx]
        kept = torch.nonzero(mask, as_tuple=False).flatten()
        nearest_pos = D[:, kept].argmin(dim=1)        # [num_experts]
        return kept[nearest_pos].long()
