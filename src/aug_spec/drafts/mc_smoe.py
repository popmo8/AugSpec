"""MC-SMoE static merged-expert draft (Table 1 "MC-SMoE" baseline,
merge–static; M-SMoE, Li et al. ICLR 2024).

Same frozen static-merge machinery as HC-SMoE (`static_merge.py`) — the
grouping spec comes from `scripts/build_mc_smoe.py` (dominant experts by
adaptive layer-wise frequency, grouping by router-logits cosine) and the
whole thing is built once per run. The single behavioural difference is
the merge itself: before the frequency-weighted average, every group
member is neuron-permutation-aligned to the group's dominant expert by
weight matching (Hungarian on the Frobenius inner-product cost, M-SMoE's
Git-Re-Basin step, extended to the SwiGLU triple). We implement the
merging stage only — MC-SMoE's post-merge low-rank compression and KD
fine-tuning have no zero-shot counterpart under the expert-count budget.

Spec schema = static_merge's plus a per-layer "dominant" list parallel to
"groups" (group j's alignment reference / merge anchor).
"""

from __future__ import annotations

from typing import Tuple

import torch

from aug_spec.merging import linear_merge

from .static_merge import StaticMergeDraft

# Per-family (gate, up, down) attribute names of the raw SwiGLU expert
# modules — same mapping as collect_calibration.stack_expert_weights.
# Permutation alignment needs the raw matrices, so families without them
# (gptoss's fused clamped-GLU) are not supported here.
_SWIGLU_KEYS = {
    "qwen3_moe": ("gate_proj", "up_proj", "down_proj"),
    "mixtral": ("w1", "w3", "w2"),
    "deepseek_moe": ("gate_proj", "up_proj", "down_proj"),
}


def _expert_weights(experts, idx: int,
                    keys: Tuple[str, str, str]) -> Tuple[torch.Tensor, ...]:
    e = experts[idx]
    return tuple(getattr(e, k).weight.detach() for k in keys)


def match_permutation(ref: Tuple[torch.Tensor, ...],
                      tgt: Tuple[torch.Tensor, ...]) -> torch.Tensor:
    """Intermediate-neuron permutation aligning expert `tgt` to `ref`.

    `ref`/`tgt` are (gate [I, D], up [I, D], down [D, I]) triples. Returns
    perm [I] such that (gate[perm], up[perm], down[:, perm]) is `tgt`
    brought into `ref`'s neuron order — the maximiser of the summed
    Frobenius inner products (M-SMoE's linear assignment problem, solved
    by the Hungarian algorithm)."""
    # Lazy import: keeps scipy out of the drafts registry import chain.
    from scipy.optimize import linear_sum_assignment

    g_r, u_r, d_r = (t.float() for t in ref)
    g_t, u_t, d_t = (t.to(g_r.device).float() for t in tgt)
    cost = g_r @ g_t.t() + u_r @ u_t.t() + d_r.t() @ d_t     # [I, I]
    _, perm = linear_sum_assignment(cost.cpu().numpy(), maximize=True)
    return torch.from_numpy(perm)


@torch.no_grad()
def permuted_linear_merge(adapter, block, member_ids, weights, ref_id):
    """Weighted average of `block`'s experts with every non-reference
    member permutation-aligned to expert `ref_id` first. Same contract as
    `linear_merge` (full-length `weights`, fp32 accumulation, output in
    the experts' dtype/keys), plus the alignment step."""
    keys = _SWIGLU_KEYS.get(adapter.name)
    if keys is None:
        raise NotImplementedError(
            f"mc_smoe: adapter {adapter.name!r} has no raw SwiGLU expert "
            f"modules to permutation-align (supported: "
            f"{sorted(_SWIGLU_KEYS)})")
    # Offload blocks hold placeholder experts; merge from the CPU-resident
    # source the controller attached (same pattern as build_weighted_avg).
    src = getattr(block, "_cpu_merge_source", block)
    ref = _expert_weights(src.experts, ref_id, keys)
    dtype = ref[0].dtype
    sums = [torch.zeros_like(t, dtype=torch.float32) for t in ref]

    for i in member_ids:
        w = float(weights[i])
        if w == 0.0:
            continue
        if i == ref_id:
            for s, t in zip(sums, ref):
                s.add_(t.float(), alpha=w)
            continue
        tgt = _expert_weights(src.experts, i, keys)
        perm = match_permutation(ref, tgt).to(tgt[0].device)
        sums[0].add_(tgt[0][perm].to(sums[0].device).float(), alpha=w)
        sums[1].add_(tgt[1][perm].to(sums[1].device).float(), alpha=w)
        sums[2].add_(tgt[2][:, perm].to(sums[2].device).float(), alpha=w)

    out = {k: s.to(dtype) for k, s in zip(keys, sums)}
    merge_device = getattr(block, "_merge_device", None)
    if merge_device is not None:
        out = {k: v.to(merge_device) for k, v in out.items()}
    return out


class MCSMoEDraft(StaticMergeDraft):
    """K offline-grouped, permutation-aligned, frequency-weighted merged
    experts per layer (frozen for the whole run).

    Args:
        spec_path: json produced by `scripts/build_mc_smoe.py`.
        draft_top_k: clusters activated per token in the draft forward.
    """

    def _merge_group(self, adapter, block, entry, gi, group, weights):
        dominants = entry.get("dominant")
        if dominants is None or len(dominants) != len(entry["groups"]):
            raise ValueError(
                f"mc_smoe: spec {self.spec_path} has no per-group "
                f"'dominant' list — was it built by "
                f"scripts/build_mc_smoe.py (not build_hc_smoe.py)?")
        dominant = int(dominants[gi])
        if dominant not in group:
            raise ValueError(
                f"mc_smoe: group {gi} dominant {dominant} is not a member "
                f"of {group}")
        if len(group) == 1:
            return linear_merge(adapter, block, group, weights)
        return permuted_linear_merge(adapter, block, group, weights,
                                     dominant)
