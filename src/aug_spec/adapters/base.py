"""Per-model-family adapter ABC.

An adapter encapsulates everything specific to one MoE family:

  * `iter_moe(model)` — yield `(layer_idx, block)` for every MoE layer
  * `num_experts(block)` — expert count for that block
  * `default_count_top_k(model)` — used by count-based drafts when CLI/YAML
                                   leaves it unset
  * `build_weighted_avg(block, weights)` — `sum_e w_e * expert[e].*` per layer
  * `make_averaged_forward(controller, layer_idx, block)` — forward callable
        that runs the dense averaged expert in draft phase, captures /
        routes normally in target phase
  * `make_masked_forward(controller, layer_idx, block)` — forward callable
        that masks gate logits to a single expert in draft phase, routes
        normally otherwise
  * `post_load(model, tokenizer, args)` — model/tokenizer setup hook
        (e.g. GPT-OSS `reasoning_effort`)
"""

from __future__ import annotations

from typing import Any, Dict, Iterator, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# A cached SVD basis for one weight-matrix type: (US, V_blocks).
#   US       : [O, q]  — the U @ diag(S) factor, precomputed.
#   V_blocks : list of n tensors [I, q] — one V block per expert.
SvdBasis = Tuple[torch.Tensor, List[torch.Tensor]]


def _svd_decompose(matrices: List[torch.Tensor], rank: int,
                   store_dtype: torch.dtype) -> SvdBasis:
    """Joint SVD over ALL experts for one weight-matrix type (Sub-MoE §3.3).

    Because the expert weights are static, this is computed ONCE per layer
    and cached; `_svd_remerge` then reuses it every cycle. The shared basis
    satisfies ``W_i ≈ US @ V_i^T`` for every expert i, so any frequency-
    weighted combination is just ``US @ (Σ w_i V_i)^T`` — no re-SVD.

    Steps:
      1. Horizontal concat:  [W_0 | … | W_{n-1}]  →  O × nI
      2. Randomised SVD (rank q):  A ≈ U diag(S) V^T
      3. Precompute  US = U ⊙ S  and split V into n blocks V_i ∈ ℝ^{I×q}

    Args:
        matrices:    n float32 tensors, each [O, I] (one per expert).
        rank:        SVD rank q. Clamped to min(rank, O, n*I).
        store_dtype: dtype the cached factors are stored in (e.g. bfloat16
                     to bound cache memory; the per-cycle merge upcasts).
    """
    n = len(matrices)
    O, I_dim = matrices[0].shape

    A = torch.cat(matrices, dim=1)                    # [O, nI]
    q = min(rank, O, n * I_dim)
    U, S, V = torch.pca_lowrank(A, q=q, center=False, niter=2)

    US = (U * S.unsqueeze(0)).to(store_dtype).contiguous()           # [O, q]
    V_blocks = [vb.to(store_dtype).contiguous()
                for vb in V.split(I_dim, dim=0)]      # n × [I, q]
    return US, V_blocks


def _svd_remerge(basis: SvdBasis, weights: List[float]) -> torch.Tensor:
    """Frequency-weighted merge + reconstruct from a cached SVD basis.

    Returns the fp32 reconstruction ``US @ (Σ w_i V_i)^T`` of shape [O, I].
    Cheap: one [I, q] accumulation plus one [O, q] × [q, I] matmul per call.
    Zero-weight experts are skipped. The caller casts to the target dtype.
    """
    US, V_blocks = basis
    I_dim, q = V_blocks[0].shape
    V_merged = torch.zeros(I_dim, q, dtype=torch.float32, device=US.device)
    for w, Vb in zip(weights, V_blocks):
        if w == 0.0:
            continue
        V_merged.add_(Vb.float(), alpha=w)
    return US.float() @ V_merged.t()                  # [O, I]


def _weighted_sum(tensors: List[torch.Tensor],
                  weights: List[float]) -> torch.Tensor:
    """fp32 frequency-weighted sum of per-expert tensors (e.g. biases)."""
    out = torch.zeros_like(tensors[0], dtype=torch.float32)
    for t, w in zip(tensors, weights):
        if w == 0.0:
            continue
        out.add_(t.float(), alpha=w)
    return out


def _pairwise_l2(flats: List[torch.Tensor]) -> torch.Tensor:
    """Pairwise L2 distance matrix [n, n] from per-expert flattened weights.

    `flats` is a list of n 1-D tensors (each expert's concatenated weights).
    Returns a CPU fp32 [n, n] matrix with a zero diagonal, so an argmin over
    a kept-expert column maps an in-mask expert to itself. Computed via
    `torch.cdist` on the stacked matrix (one pass on the experts' device).
    """
    stacked = torch.stack(flats, dim=0)               # [n, D]
    D = torch.cdist(stacked.unsqueeze(0), stacked.unsqueeze(0)).squeeze(0)
    D.fill_diagonal_(0.0)
    return D.float().cpu()


def _topk_substitute_forward(controller, layer_idx: int, block: nn.Module):
    """SpecMoE forward for the standard `block.gate` + `block.experts[e]`
    layout (Mixtral / Qwen3).

    Both phases route top-`controller.draft.route_top_k` (overriding the
    model's native top-k). In draft phase each natural winner is remapped
    through the substitute table cached at `draft_cache[layer_idx]`
    (in-mask → itself, out-of-mask → L2-nearest in-mask neighbour). In
    target phase the natural winners are routed and the per-position
    softmax captured for the next mask refresh.
    """

    def fwd(block, hidden_states):
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hs_flat = hidden_states.view(-1, hidden_dim)
        router_logits = block.gate(hs_flat)                     # [T, n_experts]
        full_softmax = F.softmax(router_logits, dim=1, dtype=torch.float)

        k = controller.draft.route_top_k
        routing_weights, selected = torch.topk(full_softmax, k, dim=-1)  # [T, k]

        if controller.in_draft_phase:
            sub_table = controller.draft_cache.get(layer_idx)
            if sub_table is not None:
                selected = sub_table.to(selected.device)[selected]   # remap
        else:
            controller.draft.capture(layer_idx, full_softmax)

        if getattr(block, "norm_topk_prob", True):
            routing_weights = routing_weights / routing_weights.sum(
                dim=-1, keepdim=True)
        routing_weights = routing_weights.to(hs_flat.dtype)

        final = torch.zeros(
            (batch_size * sequence_length, hidden_dim),
            dtype=hs_flat.dtype, device=hs_flat.device)
        expert_mask = F.one_hot(
            selected, num_classes=block.num_experts).permute(2, 1, 0)
        expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()
        for expert_idx in expert_hit:
            expert_layer = block.experts[expert_idx]
            idx, top_x = torch.where(expert_mask[expert_idx].squeeze(0))
            current_state = hs_flat[None, top_x].reshape(-1, hidden_dim)
            current_hidden = (
                expert_layer(current_state)
                * routing_weights[top_x, idx, None])
            final.index_add_(0, top_x, current_hidden.to(hs_flat.dtype))
        return final.reshape(batch_size, sequence_length, hidden_dim), router_logits

    return fwd


class MoEAdapter:
    """Abstract MoE family adapter. See module docstring."""

    name: str = "abstract"

    def iter_moe(self, model: nn.Module) -> Iterator[Tuple[int, nn.Module]]:
        raise NotImplementedError

    def num_experts(self, block: nn.Module) -> int:
        raise NotImplementedError

    def default_count_top_k(self, model: nn.Module) -> int:
        raise NotImplementedError

    def build_weighted_avg(
        self, block: nn.Module, weights: List[float],
    ) -> Dict[str, torch.Tensor]:
        raise NotImplementedError

    def _route_multi_expert(self, cache: Dict[str, Any],
                            gate_probs: torch.Tensor,
                            hs_flat: torch.Tensor, top_k: int) -> torch.Tensor:
        """Gate-remap routing over the K merged experts of a "multi" cache.

        Unlike a static cluster mix, each token routes using *its own* gate:

          1. Remap gate probs onto clusters: ``cluster_score[t,k] = Σ_{i∈k} g[t,i]``
          2. Per token, keep the `top_k` highest-scoring clusters, renormalise.
          3. Dispatch each token to its selected clusters, run the merged
             expert on that subset, and scatter-add the weighted output.

        `cache` comes from `ScoreBasedAvgDraft._cluster_and_build`: `experts`
        are the merged dense experts and `indices[k]` the original expert ids
        in cluster k (used to aggregate gate mass).

        Args:
            cache:      multi-expert cache dict.
            gate_probs: softmax of the router logits, shape [T, n_experts].
            hs_flat:    flattened hidden states, shape [T, hidden_dim].
            top_k:      clusters activated per token (native num_experts_per_tok).
        """
        experts = cache["experts"]
        indices = cache["indices"]
        K = len(experts)
        k = min(top_k, K)
        T = hs_flat.shape[0]

        # Aggregate gate mass per cluster → [T, K].
        cluster_scores = torch.stack(
            [gate_probs[:, idx].sum(dim=-1) for idx in indices], dim=1)

        # Per-token top-k clusters, renormalised. Tokens whose gate mass lands
        # entirely outside every cluster (all-zero row) fall back to uniform.
        top_scores, top_cidx = cluster_scores.topk(k, dim=-1)        # [T, k]
        denom = top_scores.sum(dim=-1, keepdim=True)
        top_scores = torch.where(
            denom > 0, top_scores / denom,
            top_scores.new_full(top_scores.shape, 1.0 / k))

        # Dense [T, K] weight matrix (0 for non-selected clusters).
        weight = hs_flat.new_zeros(T, K)
        weight.scatter_(1, top_cidx, top_scores.to(weight.dtype))

        # Dispatch per cluster: run only the tokens that selected it.
        out = torch.zeros_like(hs_flat)
        for ki in range(K):
            col = weight[:, ki]
            mask = col > 0
            if not mask.any():
                continue
            expert_out = self._run_dense_expert(experts[ki], hs_flat[mask])
            out[mask] += expert_out * col[mask].unsqueeze(-1)
        return out

    def build_svd_basis(self, block: nn.Module, rank: int = 256,
                        store_dtype: torch.dtype = torch.bfloat16) -> Dict[str, Any]:
        """Decompose every expert in `block` into a cached SVD basis.

        Computed ONCE per layer (expert weights are static) and reused by
        `build_svd_from_basis` every cycle. Returns a dict keyed per weight-
        matrix type (`SvdBasis` tuples) plus any static extras (e.g. biases)
        and a `"dtype"` entry for the reconstruction's target dtype.
        """
        raise NotImplementedError

    def build_svd_from_basis(self, basis: Dict[str, Any],
                             weights: List[float]) -> Dict[str, torch.Tensor]:
        """Frequency-weighted Sub-MoE merge from a cached `build_svd_basis`.

        Cheap per-cycle path: only V-merge + reconstruction, no re-SVD.
        Returns a dict with the same keys as `build_weighted_avg`, so the
        existing `_run_dense_expert` methods work unchanged.
        """
        raise NotImplementedError

    def make_averaged_forward(self, controller, layer_idx: int, block: nn.Module):
        raise NotImplementedError

    def make_masked_forward(self, controller, layer_idx: int, block: nn.Module):
        raise NotImplementedError

    def make_substitute_forward(self, controller, layer_idx: int,
                                block: nn.Module):
        """SpecMoE forward: top-k routing with L2-nearest substitution in the
        draft phase (cache_kind="substitute")."""
        raise NotImplementedError

    def expert_flat_weights(self, block: nn.Module) -> List[torch.Tensor]:
        """Per-expert flattened fp32 weight vectors for L2 distance.

        Returns a list of n 1-D tensors (one per expert), each the
        concatenation of that expert's weight matrices. Used once by
        `SpecMoeDraft.prepare` to build the pairwise distance matrix.
        """
        raise NotImplementedError

    def post_load(self, model: nn.Module, tokenizer: Any, args: Any) -> None:
        pass
