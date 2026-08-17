"""Offline MC-SMoE grouping (M-SMoE, Li et al. ICLR 2024 — merging stage).

Not a `ClusterMethod` — like `agglomerative.py` (HC-SMoE) this is the
one-shot, whole-expert-set grouping the MC-SMoE baseline runs offline on
calibration statistics (`scripts/build_mc_smoe.py`). Two pieces:

  * dominant-expert selection — M-SMoE's adaptive layer-wise ratio: each
    layer's activation frequencies are normalised by the layer max (most
    active expert → 1.0), then the global top L·K entries across all
    layers become dominant (≥1 per layer by construction);
  * grouping — every non-dominant expert joins its most similar dominant
    expert, similarity = cosine between per-expert router-logit vectors
    over the calibration tokens (M-SMoE Eq.1, their best variant).

The merge itself (permutation alignment + frequency-weighted averaging)
needs the model weights and lives in `drafts/mc_smoe.py`.
"""

from __future__ import annotations

from typing import List

import torch
import torch.nn.functional as F


def select_dominant_adaptive(freq: torch.Tensor, avg_k: int) -> List[List[int]]:
    """Pick dominant experts with M-SMoE's adaptive layer-wise ratio.

    Args:
        freq: [L, n] per-layer expert activation counts (calibration top-k
            votes), non-negative.
        avg_k: average dominant experts per layer; the global budget is
            L * avg_k (1 <= avg_k <= n).

    Returns:
        L sorted dominant-id lists. Each layer's frequencies are
        normalised by the layer max, the layer argmax is always kept
        (M-SMoE footnote 3), and the remaining budget goes to the
        globally highest normalised entries. Ties break on (layer,
        expert) index, so the result is fully deterministic.
    """
    if freq.dim() != 2:
        raise ValueError(f"freq must be [L, n], got {tuple(freq.shape)}")
    L, n = freq.shape
    if not 1 <= avg_k <= n:
        raise ValueError(f"avg_k must be in [1, {n}], got {avg_k!r}")

    row_max = freq.double().amax(dim=1, keepdim=True)
    scores = freq.double() / row_max.clamp(min=1e-12)   # all-zero layer → 0s

    dominant: List[List[int]] = [[] for _ in range(L)]
    rest = []
    for li in range(L):
        seed = int(scores[li].argmax())                 # tie → smallest id
        dominant[li].append(seed)
        rest.extend((-float(scores[li, ei]), li, ei)
                    for ei in range(n) if ei != seed)
    rest.sort()                                         # score desc, (li, ei) asc
    for _, li, ei in rest[:L * (avg_k - 1)]:
        dominant[li].append(ei)
    return [sorted(d) for d in dominant]


def router_logits_similarity(logits: torch.Tensor) -> torch.Tensor:
    """[T, n] calibration router logits → [n, n] cosine similarity between
    the per-expert logit vectors (rows of H in M-SMoE Eq.1)."""
    if logits.dim() != 2:
        raise ValueError(f"logits must be [T, n], got {tuple(logits.shape)}")
    h = F.normalize(logits.t().double(), dim=1, eps=1e-12)
    return h @ h.t()


def group_by_similarity(sim: torch.Tensor,
                        dominant: List[int]) -> List[List[int]]:
    """Assign every non-dominant expert to its most similar dominant one.

    Args:
        sim: [n, n] symmetric expert-similarity matrix.
        dominant: dominant expert ids (non-empty, unique, within range).

    Returns:
        len(dominant) groups (sorted member lists) that partition
        range(n); group j is led by sorted(dominant)[j]. Ties break on
        the smallest dominant id.
    """
    n = sim.shape[0]
    if sim.shape != (n, n):
        raise ValueError(f"sim must be square, got {tuple(sim.shape)}")
    dom = sorted(dominant)
    if not dom or len(set(dom)) != len(dom) or dom[0] < 0 or dom[-1] >= n:
        raise ValueError(f"bad dominant list {dominant!r} for n={n}")

    # argmax returns the first maximum → smallest dominant id on ties.
    choice = sim[:, dom].argmax(dim=1)
    groups = [[d] for d in dom]
    dom_set = set(dom)
    for j in range(n):
        if j not in dom_set:
            groups[int(choice[j])].append(j)
    return [sorted(g) for g in groups]
