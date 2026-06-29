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

import os
from typing import Any, Dict, Iterator, List, Optional, Tuple

import torch
import torch.nn as nn

from aug_spec.kernels.bmm import bmm_swiglu


# Draft compute backend for the merged "multi" experts on the offload engine:
#   "engine_bmm" (default) → C++ DispatchBmm — the optimised resident-expert
#                            path that topm and SpecMoE share;
#   "dispatch"             → archer per-expert MoEMLP::forward (A/B vs the bmm
#                            kernel, same machinery);
#   "bmm"                  → Python torch.bmm.
# The hf backend always uses bmm (no dispatcher).
_MERGED_BACKEND = os.environ.get("AUG_MERGED_BACKEND", "engine_bmm").lower()

# SpecMoE early-pin (Stage 1): pin each layer's kept-N during verify instead of
# in refresh, so they stay resident for the next draft (draft re-fetch → 0).
# 0=off, 1=pin-now, 2=keep last cycle's kept until after this layer's compute.
_EARLY_PIN = int(os.environ.get("AUG_EARLY_PIN", "0"))


def apply_offload_settings(merged_backend: Optional[str] = None,
                           early_pin: Optional[int] = None) -> None:
    """Apply YAML-sourced overrides for the two module-level offload knobs that
    used to be import-time-only env reads (A4). The env vars still win — set
    only when the corresponding env var is absent, so `AUG_MERGED_BACKEND` /
    `AUG_EARLY_PIN` remain runtime overrides. `None` means "leave as is".
    The CLI calls this once after parsing the config, before any forward."""
    global _MERGED_BACKEND, _EARLY_PIN
    if merged_backend is not None and "AUG_MERGED_BACKEND" not in os.environ:
        _MERGED_BACKEND = str(merged_backend).lower()
    if early_pin is not None and "AUG_EARLY_PIN" not in os.environ:
        _EARLY_PIN = int(early_pin)


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
                            hs_flat: torch.Tensor, top_k: int,
                            block=None) -> torch.Tensor:
        """Gate-remap routing over the K merged experts of a "multi" cache.

        Unlike a static cluster mix, each token routes using *its own* gate:

          1. Remap gate probs onto clusters: ``cluster_score[t,k] = Σ_{i∈k} g[t,i]``
          2. Per token, keep the `top_k` highest-scoring clusters, renormalise.
          3. Run the K merged experts on their selected tokens and combine.

        Step 3 has three backends, all numerically equivalent:
          * offload — `dispatch_merged_local`: the K merged experts go through
            the archer engine's MoEMLP::forward, the SAME kernel the SpecMoE
            (substitute) draft dispatches, so the comparison isolates the
            algorithm rather than the kernel.
          * hf SwiGLU adapters — one batched `bmm` (`_dense_experts_batched`).
          * otherwise — a per-cluster Python loop.

        `cache` comes from `ScoreBasedAvgDraft._cluster_and_build`: `experts`
        are the merged dense experts and `indices[k]` the original expert ids
        in cluster k (used to aggregate gate mass).

        Args:
            cache:      multi-expert cache dict.
            gate_probs: softmax of the router logits, shape [T, n_experts].
            hs_flat:    flattened hidden states, shape [T, hidden_dim].
            top_k:      clusters activated per token (native num_experts_per_tok).
            block:      the MoE block (offload dispatch needs its engine handle).
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

        disp = self._merge_dispatcher(block) if block is not None else None

        # engine_bmm (offload): pre-stacked weights → the engine's batched bmm
        # (C++ DispatchBmm). Same op sequence as the Python bmm below, but the
        # resident-expert compute is unified inside the engine — the home for the
        # eventual CUTLASS grouped-GEMM upgrade. SpecMoE's draft calls the same
        # entry, so a bmm-vs-bmm comparison isolates the algorithm.
        if disp is not None and _MERGED_BACKEND == "engine_bmm":
            stk = self._swiglu_stack(cache, experts)
            if stk is not None:
                gw, uw, dw = stk
                return disp.dispatch_bmm(
                    hs_flat, gw, uw, dw, weight, hs_flat.device.index)

        # dispatch (offload, default): run the K merged experts through the archer
        # engine's per-expert MoEMLP::forward — identical to SpecMoE's draft
        # execution, so a TPS comparison there isolates the algorithm, not the
        # kernel. The C++ side does the weighted token-combine and returns [T, D].
        if disp is not None and _MERGED_BACKEND == "dispatch":
            lists = self._merged_tensor_lists(experts)
            if lists is not None:
                return disp.dispatch_merged_local(
                    hs_flat, weight, lists, hs_flat.device.index)

        # Python bmm (hf, or AUG_MERGED_BACKEND=bmm on offload): adapters with a
        # uniform SwiGLU layout run all K merged experts in 3 batched bmm
        # launches/layer instead of a K-iteration Python loop of tiny per-expert
        # F.linear. Dense-over-all-K only costs K/k× the selected-token FLOPs, but
        # the GEMMs are launch-bound at draft batch sizes so it is far cheaper.
        eo = self._dense_experts_batched(cache, experts, hs_flat)  # [K, T, D] | None
        if eo is not None:
            # out[t] = Σ_k weight[t,k] · expert_k(hs[t]).
            return (eo * weight.t().unsqueeze(-1)).sum(dim=0)      # [T, D]

        # Generic fallback (e.g. gptoss): dispatch per cluster, running only the
        # tokens that selected it.
        out = torch.zeros_like(hs_flat)
        for ki in range(K):
            col = weight[:, ki]
            mask = col > 0
            if not mask.any():
                continue
            expert_out = self._run_dense_expert(experts[ki], hs_flat[mask])
            out[mask] += expert_out * col[mask].unsqueeze(-1)
        return out

    def _swiglu_stack(self, cache: Dict[str, Any],
                      experts: List[Dict[str, torch.Tensor]]):
        """Stack the K merged experts' SwiGLU weights for bmm, memoised on the
        per-cycle `cache` dict — returns ``(gate, up, down)`` as
        ``[K, D, I], [K, D, I], [K, I, D]``, or ``None`` when the adapter has no
        uniform SwiGLU layout (e.g. gptoss). Overridden by qwen3 / mixtral.
        Shared by the Python bmm and the engine_bmm (C++ DispatchBmm) paths."""
        return None

    def _dense_experts_batched(self, cache: Dict[str, Any],
                               experts: List[Dict[str, torch.Tensor]],
                               hs_flat: torch.Tensor):
        """All K merged experts applied to every token via one batched bmm,
        returning ``[K, T, D]`` — or ``None`` (no batched layout) to fall back
        to the per-cluster loop."""
        stk = self._swiglu_stack(cache, experts)
        if stk is None:
            return None
        return bmm_swiglu(hs_flat, *stk)

    @staticmethod
    def _merge_dispatcher(block):
        """The archer expert dispatcher for an offload block, or None (hf /
        no engine) — `dispatch_merged_local` lives on it."""
        ex = getattr(block, "expert_executor", None)
        return getattr(ex, "expert_dispatcher", None) if ex is not None else None

    def _merged_tensor_lists(self, experts: List[Dict[str, torch.Tensor]]):
        """K × [w0, w1, w2] GPU tensors in tensor-id order, ready for
        `dispatch_merged_local` (→ MoEMLP::forward). None when the adapter's
        merged experts can't run through that kernel (e.g. gptoss biases).
        Overridden by adapters whose experts match the MoEMLP layout."""
        return None

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
