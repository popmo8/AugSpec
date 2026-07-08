"""Hybrid α-blended relation clustering (prefill act-sim × decode co-occur).

Builds the per-layer *Expert Relation Map*

    R[i,j] = alpha * N(A[i,j]) + (1 - alpha) * N(C[i,j])

from two component maps and runs the shared greedy max-value, max-size-2
pairing on it:

  * A — activation similarity (`ctx.pair_sim`), captured during the PREFILL
    forward only: the engine turns the C++ expert-output capture off at the
    first draft start (`act_sim_prefill_only`), so decode cycles pay none of
    actsim's per-cycle capture latency. Frozen per question after prefill.
  * C — co-occurrence (`ctx.cooccur`), accumulated over the DECODE verify
    forwards only (`cooccur_scope="decode"` skips the prefill forward) and
    growing across cycles — the temporal-locality signal. `cooccur_norm=
    "cosine"` divides by the diagonal (per-expert counts) to remove the
    hot-expert frequency bias; `"raw"` keeps plain counts.

N(·) maps each map's OBSERVED active pairs onto [0, 1] (`norm="rank"` default —
scale-free, robust to the L2 long tail; `"minmax"` as ablation, both monotonic
and tie-preserving) and UNOBSERVED pairs to -1, below any observed pair — so missing evidence ranks last and is only
used to reach K, exactly like the single-signal methods' sentinels. Endpoints
therefore reproduce them: `alpha=1` ranks identically to activation_similarity
(on the prefill-only table) and `alpha=0` with `cooccur_norm="raw"` ranks
identically to cooccur_pair.
"""

from __future__ import annotations

import itertools
from typing import List, Optional

import torch

from .base import ClusterContext, ClusterMethod, greedy_pair

_NO_DATA = -1.0


def norm01(values: torch.Tensor, mask: torch.Tensor,
           norm: str) -> torch.Tensor:
    """Map `values[mask]` monotonically (tie-preserving) onto [0, 1] and
    `~mask` entries to the -1 no-data sentinel. 1-D in, 1-D out.

    minmax — affine stretch of the observed range (all-equal → 0.5).
    rank   — dense rank of the observed values / (#distinct - 1); scale-free,
             robust to outliers (single distinct value → 0.5).
    """
    out = torch.full_like(values, _NO_DATA, dtype=torch.float32)
    vals = values[mask].float()
    if vals.numel() == 0:
        return out
    if norm == "minmax":
        vmin, vmax = vals.min(), vals.max()
        scored = (torch.full_like(vals, 0.5) if vmax == vmin
                  else (vals - vmin) / (vmax - vmin))
    else:  # rank
        uniq, inv = torch.unique(vals, return_inverse=True)
        scored = (torch.full_like(vals, 0.5) if len(uniq) == 1
                  else inv.float() / (len(uniq) - 1))
    out[mask] = scored
    return out


class HybridRelationCluster(ClusterMethod):
    needs_cooccur = True
    needs_activation_sim = True
    act_sim_prefill_only = True

    def __init__(self, alpha: float = 0.5, metric: str = "l2",
                 norm: str = "rank", cooccur_norm: str = "cosine",
                 cooccur_scope: str = "decode"):
        if not 0.0 <= float(alpha) <= 1.0:
            raise ValueError(f"hybrid alpha must be in [0, 1], got {alpha!r}")
        if metric not in ("cosine", "l2"):
            raise ValueError(f"hybrid metric must be cosine|l2, got {metric!r}")
        if norm not in ("minmax", "rank"):
            raise ValueError(f"hybrid norm must be minmax|rank, got {norm!r}")
        if cooccur_norm not in ("cosine", "raw"):
            raise ValueError(f"hybrid cooccur_norm must be cosine|raw, "
                             f"got {cooccur_norm!r}")
        if cooccur_scope not in ("decode", "all"):
            raise ValueError(f"hybrid cooccur_scope must be decode|all, "
                             f"got {cooccur_scope!r}")
        self.alpha = float(alpha)
        # The draft reads self.metric to build ctx.pair_sim (same contract as
        # activation_similarity): cosine → mean output-cosine, sentinel -1;
        # l2 → NEGATIVE mean L2 distance, sentinel -inf.
        self.metric = metric
        self.norm = norm
        self.cooccur_norm = cooccur_norm
        # Read by the draft's _accumulate_cooccur: "decode" skips the prefill
        # forward so C reflects only decode-time temporal locality.
        self.cooccur_scope = cooccur_scope

    def assign(self, ctx: ClusterContext, K: int) -> List[List[int]]:
        return greedy_pair(self._blend(ctx), list(ctx.active), K)

    # ── relation-map assembly ────────────────────────────────────────────
    def _blend(self, ctx: ClusterContext) -> Optional[torch.Tensor]:
        """R over the active pairs (others stay at the -1 sentinel); None when
        neither component map has been captured yet (greedy_pair falls back to
        singletons, same as the single-signal methods)."""
        if ctx.pair_sim is None and ctx.cooccur is None:
            return None
        pairs = list(itertools.combinations(ctx.active, 2))
        if not pairs:
            return None
        ii = torch.tensor([i for i, _ in pairs])
        jj = torch.tensor([j for _, j in pairs])
        sa = self._act_scores(ctx.pair_sim, ii, jj)
        sc = self._cooccur_scores(ctx.cooccur, ii, jj)
        r = self.alpha * sa + (1.0 - self.alpha) * sc
        n = len(ctx.weights)
        R = torch.full((n, n), _NO_DATA)
        R[ii, jj] = r
        R[jj, ii] = r
        return R

    def _act_scores(self, pair_sim: Optional[torch.Tensor],
                    ii: torch.Tensor, jj: torch.Tensor) -> torch.Tensor:
        """N(A) over the pair list. Unvisited pairs carry the table's sentinel
        (l2 → -inf, cosine → -1, both strictly below any observed value)."""
        if pair_sim is None:
            return torch.full((len(ii),), _NO_DATA)
        vals = pair_sim[ii, jj].float()
        if self.metric == "l2":
            mask = torch.isfinite(vals)
        else:
            mask = vals > -1.0
        return norm01(vals, mask, self.norm)

    def _cooccur_scores(self, cooccur: Optional[torch.Tensor],
                        ii: torch.Tensor, jj: torch.Tensor) -> torch.Tensor:
        """N(C) over the pair list. Count 0 = never co-fired = no data. cosine
        divides by sqrt(diag_i * diag_j) (diagonal = per-expert counts), which
        is > 0 whenever the pair count is."""
        if cooccur is None:
            return torch.full((len(ii),), _NO_DATA)
        C = cooccur.float()
        vals = C[ii, jj]
        mask = vals > 0
        if self.cooccur_norm == "cosine":
            d = C.diagonal()
            vals = vals / (d[ii] * d[jj]).clamp_min(1e-12).sqrt()
        return norm01(vals, mask, self.norm)
