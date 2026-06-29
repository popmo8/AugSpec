"""Per-cycle expert importance scorers.

A *scorer* maps a `[seq_len, num_experts]` softmax tensor (target router's
distribution over experts at every verified position of one forward pass)
to a `[num_experts]` importance vector. Higher = more important.

Drafts that build per-cycle weights (e.g. count-weighted averaged expert)
plug a scorer in to convert raw router output → per-expert weights.
"""

from __future__ import annotations

from typing import Any, Callable, Tuple

import torch


Scorer = Callable[[torch.Tensor], torch.Tensor]


def make_count_scorer(count_top_k: int) -> Scorer:
    """Vote-counting scorer.

    At every verified position, pick the top-`count_top_k` experts; tally
    how many positions voted for each expert. Setting `count_top_k` to
    the model's native `num_experts_per_tok` mirrors the routing the
    target itself performed.
    """
    assert count_top_k >= 1, f"count_top_k={count_top_k} must be >= 1"

    def fn(scores: torch.Tensor) -> torch.Tensor:
        num_experts = scores.shape[-1]
        top_idx = torch.topk(scores, k=count_top_k, dim=-1).indices
        counts = torch.bincount(top_idx.flatten(), minlength=num_experts)
        return counts.float()

    return fn


def make_cooccurrence_scorer(top_k: int) -> Scorer:
    """Pairwise co-occurrence scorer.

    At every position pick the top-`top_k` experts; return an ``[n, n]`` matrix
    whose entry (i, j) counts positions whose top-k contained BOTH i and j
    (diagonal = per-expert count). Symmetric. Unlike the count scorer (a 1-D
    vote tally), this captures *which experts fire together on the same token* —
    the signal co-occurrence clustering merges on. Drafts accumulate it across
    the whole question (prefill + decode).
    """
    assert top_k >= 1, f"top_k={top_k} must be >= 1"

    def fn(scores: torch.Tensor) -> torch.Tensor:
        n = scores.shape[-1]
        top = torch.topk(scores, k=min(top_k, n), dim=-1).indices   # [seq, k]
        oneh = torch.zeros(scores.shape[0], n,
                           device=scores.device, dtype=torch.float)
        oneh.scatter_(1, top, 1.0)
        return oneh.t() @ oneh                                      # [n, n]

    return fn


def make_softmax_scorer(
    weights: Tuple[float, ...] = (0.5, 0.3, 0.2),
) -> Scorer:
    """Softmax-aggregation scorer.

    Weighted sum of the last K positions' router softmax distributions;
    `weights[0]` is applied to the most recent position. Renormalised so
    fewer-than-K positions still produce a proper distribution.
    """
    weights = tuple(weights)
    assert len(weights) >= 1, "weights must be non-empty"

    def fn(scores: torch.Tensor) -> torch.Tensor:
        n = scores.shape[0]
        take = min(len(weights), n)
        w = weights[:take]
        w_sum = sum(w)
        agg = scores[n - 1] * (w[0] / w_sum)
        for i in range(1, take):
            agg = agg + scores[n - 1 - i] * (w[i] / w_sum)
        return agg

    return fn


def make_hybrid_scorer(
    count_top_k: int,
    weights: Tuple[float, ...] = (0.5, 0.3, 0.2),
    alpha: float = 0.5,
) -> Scorer:
    """Convex blend: `alpha * count + (1 - alpha) * softmax`.

    Both components are L1-normalised before mixing so `alpha` is on a
    consistent scale. `alpha=1.0` → pure count, `alpha=0.0` → pure softmax.
    """
    assert 0.0 <= alpha <= 1.0, f"alpha={alpha} not in [0, 1]"
    count = make_count_scorer(count_top_k)
    soft = make_softmax_scorer(weights)
    eps = 1e-12

    def fn(scores: torch.Tensor) -> torch.Tensor:
        c = count(scores)
        s = soft(scores)
        c_norm = c / (c.sum() + eps)
        s_norm = s / (s.sum() + eps)
        return alpha * c_norm + (1.0 - alpha) * s_norm

    return fn


def make_scorer(kind: str, **kwargs: Any) -> Scorer:
    """Dispatch by name. `kind` ∈ {count, softmax, hybrid}."""
    if kind == "count":
        return make_count_scorer(**kwargs)
    if kind == "softmax":
        return make_softmax_scorer(**kwargs)
    if kind == "hybrid":
        return make_hybrid_scorer(**kwargs)
    raise ValueError(
        f"unknown scorer kind={kind!r}; expected count, softmax, or hybrid")
