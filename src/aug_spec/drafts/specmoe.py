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
import torch.nn as nn
import torch.nn.functional as F

# Read the offload knobs at call time (apply_offload_settings mutates them, A4),
# so reference attributes on the module rather than binding the values.
from aug_spec.adapters import base as _adapter_base
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
                 count_top_k: Optional[int] = None, pin: bool = False):
        if N < 1:
            raise ValueError(f"N must be >= 1, got {N!r}")
        if route_top_k < 1:
            raise ValueError(f"route_top_k must be >= 1, got {route_top_k!r}")
        self.N = N
        self.route_top_k = route_top_k
        self.count_top_k = count_top_k if count_top_k is not None else route_top_k
        self._scorer = make_count_scorer(count_top_k=self.count_top_k)
        # specmoe_pin_plan.md: pin the kept-N in GPU (offload) so the draft reads
        # them resident (0 PCIe) — the faithful SpecMoE memory behaviour. The
        # kept-N then occupy ~N×L×expert of the archer pool (no merged-reserve;
        # they ARE experts in the pool). No-op on hf / when pin=False.
        self.pin = pin

        # layer_idx → CPU fp32 count vector captured in the target phase
        self.target_score: Dict[int, torch.Tensor] = {}
        # layer_idx → CPU bool kept-mask [num_experts] (telemetry + table build)
        self._draft_mask: Dict[int, torch.Tensor] = {}
        # layer_idx → CPU fp32 [num_experts, num_experts] L2 distances.
        # Static; built once in prepare(), never cleared by reset().
        self._expert_dist: Dict[int, torch.Tensor] = {}
        self.num_experts: int = 0
        # layer_idx → (gw, uw, dw, kept_to_col) for the engine_bmm draft path,
        # gathered from the resident pinned kept-N and memoised per cycle (the
        # kept set changes every refresh, so it is cleared there).
        self._kept_bmm: Dict[int, tuple] = {}

        # Run-level: one int per verify cycle = how many of the top-route_top_k
        # target experts fell outside the OLD mask, summed over layers.
        self.cycle_misses: List[int] = []
        # Run-level: per cycle, how many kept-N experts are NEW vs last cycle
        # (summed over layers) — AUG_PROFILE churn metric, compared against the
        # draft re-fetch count to size the "pin earlier" opportunity.
        self.kept_changed: List[int] = []

    # ── one-time setup ──────────────────────────────────────────────────
    def prepare(self, adapter, blocks) -> None:
        if self._expert_dist:
            return
        for li, block in blocks:
            # Offload: block.experts are placeholders — read real weights from
            # the CPU source the controller attached (hf: src is block itself).
            src = getattr(block, "_cpu_merge_source", block)
            self._expert_dist[li] = pairwise_l2(
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
        self._kept_bmm.clear()

    # ── target-phase capture (forward passes the full softmax) ──────────
    def capture(self, layer_idx: int, softmax: torch.Tensor) -> None:
        score_vec = self._scorer(softmax)
        self.target_score[layer_idx] = score_vec.float().detach().cpu()

    # ── per-cycle mask + substitute-table refresh ───────────────────────
    def refresh(self, adapter, blocks, draft_cache: Dict[int, torch.Tensor]) -> None:
        if not self.target_score:
            return

        # New kept sets → invalidate the engine_bmm gathered/stacked weights.
        self._kept_bmm.clear()

        # Miss telemetry against the OLD masks, before they are overwritten.
        cycle_miss = 0
        for li, score_vec in self.target_score.items():
            old_mask = self._draft_mask.get(li)
            if old_mask is None:
                continue
            top = torch.topk(score_vec, self.route_top_k).indices
            cycle_miss += int((~old_mask[top]).sum().item())
        self.cycle_misses.append(cycle_miss)

        cycle_changed = 0
        layer_to_block = dict(blocks)
        for li, score_vec in self.target_score.items():
            top_n = torch.topk(score_vec, self.N).indices
            mask = torch.zeros(self.num_experts, dtype=torch.bool)
            mask[top_n] = True
            old_mask = self._draft_mask.get(li)
            if old_mask is not None:
                cycle_changed += int((mask & ~old_mask).sum().item())
            self._draft_mask[li] = mask
            draft_cache[li] = self._build_substitute_table(li, mask)
            # specmoe_pin_plan.md: pin this layer's kept-N so the draft reads
            # them resident (FindExpertEvict skips pinned). Offload-only; the
            # draft's batch_size==1 dispatch then fetches + keeps them.
            if self.pin:
                block = layer_to_block[li]
                ex = getattr(block, "expert_executor", None)
                disp = getattr(ex, "expert_dispatcher", None) if ex else None
                if disp is not None:
                    disp.set_pinned(li, top_n.tolist(), 0)
        self.kept_changed.append(cycle_changed)

    def _dispatcher(self, block):
        ex = getattr(block, "expert_executor", None)
        return getattr(ex, "expert_dispatcher", None) if ex else None

    def early_pin(self, layer_idx, block, mode=1) -> None:
        """Pin this layer's next-draft kept-N during verify, the moment its count
        is captured (before this layer's dispatch), so they are not evicted
        before the draft. mode 2 also keeps last cycle's kept-N pinned through
        this layer's compute (dropped later in `late_unpin`) so an old kept-N
        that this verify still routes is not evicted mid-compute. Offload + pin
        only; no-op otherwise."""
        if not self.pin:
            return
        score = self.target_score.get(layer_idx)   # just set by capture()
        if score is None or self.num_experts == 0:
            return
        disp = self._dispatcher(block)
        if disp is None:
            return
        ids = set(torch.topk(score, self.N).indices.tolist())   # new kept-N
        if mode == 2:
            old_mask = self._draft_mask.get(layer_idx)          # last cycle's
            if old_mask is not None:
                ids |= set(torch.nonzero(old_mask, as_tuple=False)
                           .flatten().tolist())
        disp.set_pinned(layer_idx, list(ids), 0)

    def late_unpin(self, layer_idx, block) -> None:
        """mode 2: after this layer's experts are computed, re-pin only the new
        kept-N (drops the old kept-N the next draft won't use)."""
        if not self.pin:
            return
        score = self.target_score.get(layer_idx)
        if score is None or self.num_experts == 0:
            return
        disp = self._dispatcher(block)
        if disp is not None:
            top_n = torch.topk(score, self.N).indices
            disp.set_pinned(layer_idx, top_n.tolist(), 0)

    def _build_substitute_table(self, layer_idx: int,
                                mask: torch.Tensor) -> torch.Tensor:
        """[num_experts] long: each expert → the kept expert it routes to.
        Kept experts map to themselves (distance diagonal is zero)."""
        D = self._expert_dist[layer_idx]
        kept = torch.nonzero(mask, as_tuple=False).flatten()
        nearest_pos = D[:, kept].argmin(dim=1)        # [num_experts]
        return kept[nearest_pos].long()

    def kept_bmm_state(self, layer_idx, dispatcher, gpu, device):
        """For the engine_bmm draft path: the kept-N experts' weights stacked
        for `DispatchBmm` plus a kept-id→column map, gathered from the resident
        pinned weights and memoised per cycle. Returns
        ``(gw [N,D,I], uw [N,D,I], dw [N,I,D], kept_to_col [num_experts])`` or
        ``None`` when any kept expert isn't GPU-resident (caller then falls back
        to the per-expert dispatch). Requires pin=true so the kept-N stay
        resident across the draft."""
        st = self._kept_bmm.get(layer_idx)
        if st is not None:
            return st
        mask = self._draft_mask.get(layer_idx)
        if mask is None:
            return None
        kept = torch.nonzero(mask, as_tuple=False).flatten().tolist()  # ascending
        gates, ups, downs = [], [], []
        n_res = 0
        for e in kept:
            w = dispatcher.get_resident_expert_weights(layer_idx, e, gpu)
            if len(w) >= 3:
                n_res += 1
                gates.append(w[0]); ups.append(w[1]); downs.append(w[2])
        # AUG_PROFILE diagnostic: how many of this layer's kept-N are actually
        # GPU-resident at draft time (need 100% for the bmm path to engage).
        self._bmm_calls = getattr(self, "_bmm_calls", 0) + 1
        self._bmm_res_sum = getattr(self, "_bmm_res_sum", 0) + n_res
        self._bmm_kept_sum = getattr(self, "_bmm_kept_sum", 0) + len(kept)
        if n_res < len(kept):
            return None  # not all resident → caller uses per-expert dispatch
        # tensor-id order [gate[I,D], up[I,D], down[D,I]] → bmm operands.
        gw = torch.stack(gates).transpose(1, 2).contiguous()  # [N, D, I]
        uw = torch.stack(ups).transpose(1, 2).contiguous()    # [N, D, I]
        dw = torch.stack(downs).transpose(1, 2).contiguous()  # [N, I, D]
        kept_to_col = torch.zeros(self.num_experts, dtype=torch.long, device=device)
        for col, e in enumerate(kept):
            kept_to_col[e] = col
        st = (gw, uw, dw, kept_to_col)
        self._kept_bmm[layer_idx] = st
        return st


# ── SpecMoE substitute forward (moved out of adapters/base.py, A5) ──────────
# This is SpecMoE-draft logic, not model-adapter logic: the adapter only exposes
# the generic `gate` / `_dispatch_selected` hooks these call through. Moved here
# next to SpecMoeDraft. The adapter `make_substitute_forward` lazy-imports
# `topk_substitute_forward` (avoids an adapters<->drafts import cycle).

def pairwise_l2(flats: List[torch.Tensor]) -> torch.Tensor:
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


def specmoe_engine_bmm(controller, layer_idx, block, hs_flat,
                       selected, routing_weights):
    """Run SpecMoE's pinned kept-N experts through the engine's batched bmm
    (`DispatchBmm`). The draft supplies a per-cycle-memoised stack of the kept
    weights plus a kept-id→column map; `selected` (already substituted to kept)
    and `routing_weights` are scattered into a [T, N] routing matrix. Returns
    None when the kept experts aren't resident, so the caller falls back to the
    per-expert dispatch."""
    disp = block.expert_executor.expert_dispatcher
    gpu = hs_flat.device.index
    st = controller.draft.kept_bmm_state(
        layer_idx, disp, gpu, hs_flat.device)
    if st is None:
        return None
    gw, uw, dw, kept_to_col = st
    cols = kept_to_col[selected]                          # [T, k]
    routing = hs_flat.new_zeros(hs_flat.shape[0], gw.shape[0])  # [T, N]
    routing.scatter_add_(1, cols, routing_weights)
    return disp.dispatch_bmm(hs_flat, gw, uw, dw, routing, gpu)


def topk_substitute_forward(controller, layer_idx: int, block: nn.Module):
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
            # AUG_EARLY_PIN: pin this layer's next-draft kept-N NOW (count is known
            # the moment the gate runs) so they stay resident for the draft
            # (draft_fetch → 0). Default (0) pins only in refresh (after verify).
            #   mode 1 = pin new kept-N now (unpins old-dropped immediately).
            #   mode 2 = also keep last cycle's kept pinned through this layer's
            #            compute, then drop the dropped ones in late_unpin (after
            #            dispatch) so an old-kept this verify still uses isn't
            #            evicted early.
            early_pin = _adapter_base._EARLY_PIN
            if early_pin and hasattr(controller.draft, "early_pin"):
                controller.draft.early_pin(layer_idx, block, early_pin)

        if getattr(block, "norm_topk_prob", True):
            routing_weights = routing_weights / routing_weights.sum(
                dim=-1, keepdim=True)
        routing_weights = routing_weights.to(hs_flat.dtype)

        # engine_bmm (offload, draft): run the resident pinned kept-N experts
        # through the engine's batched bmm — the SAME entry topm's merged draft
        # uses, so a bmm-vs-bmm comparison isolates the algorithm. Falls back to
        # the per-expert dispatch below if the kept experts aren't resident.
        if (controller.in_draft_phase
                and _adapter_base._MERGED_BACKEND == "engine_bmm"
                and hasattr(block, "expert_executor")
                and hasattr(controller.draft, "kept_bmm_state")):
            out = specmoe_engine_bmm(
                controller, layer_idx, block, hs_flat, selected, routing_weights)
            if out is not None:
                return (out.reshape(batch_size, sequence_length, hidden_dim),
                        router_logits)

        # Offload: experts are placeholders — run the (remapped) winners
        # through the archer engine's dispatch instead of the per-expert loop.
        if hasattr(block, "expert_executor"):
            final = controller.adapter._dispatch_selected(
                block, hs_flat, selected, routing_weights)
            # mode 2: this layer's experts are now computed → drop the old kept-N
            # that the next draft won't use (kept pinned until here so they were
            # not evicted mid-compute).
            if (_adapter_base._EARLY_PIN == 2 and not controller.in_draft_phase
                    and hasattr(controller.draft, "late_unpin")):
                controller.draft.late_unpin(layer_idx, block)
            return (final.reshape(batch_size, sequence_length, hidden_dim),
                    router_logits)

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
