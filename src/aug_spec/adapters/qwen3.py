"""Qwen3-MoE adapter (Qwen3-30B-A3B-Base et al.).

Block path:    `model.model.layers[i].mlp` (filter by presence of
               `.gate` + `.experts` — Qwen3 reuses `layer.mlp` for the
               dense fallback)
Expert MLP:    `gate_proj` / `up_proj` / `down_proj` (SwiGLU, same shape
               as Mixtral but different attribute names)
Native top-k:  `config.num_experts_per_tok` (= 8 for A3B)
Note:          `block.norm_topk_prob` toggles the routing-weight renorm
               (True on A3B; HF default is False — match the block flag).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .base import (
    MoEAdapter,
    _stack_swiglu_weights,
    _topk_substitute_forward,
)

# moe_infinity is an optional dependency: an hf-only environment must still
# be able to import this adapter. When absent, _is_offload_block is always
# False and only the hf expert-loop path runs.
try:
    from moe_infinity.models.qwen import Qwen3MoEBlock as _Qwen3OffloadBlock
except ImportError:
    _Qwen3OffloadBlock = None


def _is_offload_block(block) -> bool:
    """True when `block` is moe_infinity's offloaded Qwen3 MoE block, whose
    experts are placeholders and must be routed through the archer engine's
    `dispatch_local` rather than the hf expert loop."""
    return _Qwen3OffloadBlock is not None and isinstance(block, _Qwen3OffloadBlock)


class Qwen3MoeAdapter(MoEAdapter):
    name = "qwen3_moe"

    def iter_moe(self, model):
        if not (hasattr(model, "model") and hasattr(model.model, "layers")):
            raise TypeError("Expected a Qwen3-MoE model with .model.layers.")
        for i, layer in enumerate(model.model.layers):
            block = getattr(layer, "mlp", None)
            if block is None:
                continue
            if hasattr(block, "gate") and hasattr(block, "experts"):
                yield i, block

    def num_experts(self, block):
        return len(block.experts)

    def default_count_top_k(self, model):
        return getattr(model.config, "num_experts_per_tok", 8)

    def build_weighted_avg(self, block, weights):
        # M9b GPU merge: merge the experts directly on GPU through the archer
        # dispatcher. Sources a recent verify left resident cost zero PCIe; any
        # non-resident source is copied host->GPU transiently inside the engine
        # (the dispatcher cache is untouched). Bit-exact with the CPU path (C2),
        # so the M7 precision baseline still holds.
        if getattr(block, "_merge_offload", False) and hasattr(block, "expert_executor"):
            nz_idx = [i for i, w in enumerate(weights) if w != 0.0]
            if nz_idx:  # empty would crash the C++ side; CPU path handles it
                nz_w = [float(weights[i]) for i in nz_idx]
                out = block.expert_executor.expert_dispatcher.merge_experts_local(
                    block.layer_id, nz_idx, nz_w, 0)
                return {"gate_proj": out[0], "up_proj": out[1],
                        "down_proj": out[2]}

        # Offload backend: `block.experts` are placeholders, so merge from the
        # CPU-resident source the controller attached, then move the result to
        # GPU (M4's "(d) CPU merge" — fp32 accumulate on host, ship one expert).
        # hf backend: src is block itself and _merge_device is None, so the
        # accumulation runs on-GPU and nothing moves — behaviour unchanged.
        src = getattr(block, "_cpu_merge_source", block)
        merge_device = getattr(block, "_merge_device", None)

        ref_g = src.experts[0].gate_proj.weight
        ref_u = src.experts[0].up_proj.weight
        ref_d = src.experts[0].down_proj.weight
        dtype = ref_g.dtype

        g_sum = torch.zeros_like(ref_g, dtype=torch.float32)
        u_sum = torch.zeros_like(ref_u, dtype=torch.float32)
        d_sum = torch.zeros_like(ref_d, dtype=torch.float32)

        # Iterate weights (not experts) so zero-weight experts cost nothing —
        # on offload that skips touching the placeholder/source module entirely.
        for e_idx, w in enumerate(weights):
            if w == 0.0:
                continue
            expert = src.experts[e_idx]
            g_sum.add_(expert.gate_proj.weight.float(), alpha=w)
            u_sum.add_(expert.up_proj.weight.float(), alpha=w)
            d_sum.add_(expert.down_proj.weight.float(), alpha=w)

        out = {
            "gate_proj": g_sum.to(dtype),
            "up_proj": u_sum.to(dtype),
            "down_proj": d_sum.to(dtype),
        }
        del g_sum, u_sum, d_sum
        if merge_device is not None:
            out = {k: v.to(merge_device) for k, v in out.items()}
        return out

    def _run_dense_expert(self, avg, hs_flat):
        # Qwen3MoeMLP: SiLU(gate_proj · h) ⊙ (up_proj · h) → down_proj(...).
        gate = F.linear(hs_flat, avg["gate_proj"])
        up = F.linear(hs_flat, avg["up_proj"])
        hidden = F.silu(gate) * up
        return F.linear(hidden, avg["down_proj"])

    def _swiglu_stack(self, cache, experts):
        # gate_proj / up_proj / down_proj, stacked + transposed for bmm.
        return _stack_swiglu_weights(
            cache, experts, "gate_proj", "up_proj", "down_proj")

    def _merged_tensor_lists(self, experts):
        # Tensor-id order = [gate, up, down], matching MergeExpertsLocal's
        # output and what MoEMLP::forward expects for this expert type.
        # .contiguous() guards the device-to-device memcpy in SetTensorsDirect
        # (no-op for the freshly merged weights, which already are contiguous).
        return [[e["gate_proj"].contiguous(), e["up_proj"].contiguous(),
                 e["down_proj"].contiguous()] for e in experts]

    def _route_offload(self, block, hs_flat, gate_logits):
        """Offload-backend verify routing (replaces the hf expert loop —
        offloaded experts are placeholders). Builds per-token
        (num_tokens, num_experts) masks and hands them to the archer
        engine's batched `dispatch_local`, which fetches each needed
        expert on demand and returns the weighted-combined result.

        The top-k softmax + `norm_topk_prob` renorm here reproduces
        moe_infinity's fused `lib.topk_softmax` bit-for-bit (validated in
        M3, offload_plan.md). Accepting `gate_logits` (rather than calling
        the kernel) is what lets masked drafts feed -inf'd logits through —
        the kernel is fixed top-k and won't take a mask.
        """
        routing_weights = F.softmax(gate_logits, dim=1, dtype=torch.float)
        topk_w, topk_idx = torch.topk(routing_weights, block.top_k, dim=-1)
        if getattr(block, "norm_topk_prob", True):
            topk_w = topk_w / topk_w.sum(dim=-1, keepdim=True)
        topk_w = topk_w.to(hs_flat.dtype)

        num_tokens = gate_logits.shape[0]
        router_mask = torch.zeros(
            num_tokens, block.num_experts,
            dtype=torch.bool, device=hs_flat.device)
        router_mask.scatter_(1, topk_idx, True)
        weights_mask = torch.zeros(
            num_tokens, block.num_experts,
            dtype=hs_flat.dtype, device=hs_flat.device)
        weights_mask.scatter_add_(1, topk_idx, topk_w)

        block.expert_executor.dispatch_local(
            block.layer_id, hs_flat, router_mask, weights_mask)
        out = block.expert_executor.wait_dispatch_local()

        # This layer's experts are now GPU-resident (post-dispatch, pre-evict):
        # the offload-merge engine's window to merge-while-resident / overlap
        # with the next layer's fetch. No-op unless merge_offload built one.
        engine = getattr(block, "_merge_engine", None)
        if engine is not None:
            engine.on_verify_layer(block.layer_id, block)

        return out.to(hs_flat.dtype)   # wait returns fp32; match hf .to(dtype)

    def _dispatch_selected(self, block, hs_flat, selected, weights):
        """Offload expert execution from explicit per-token (selected, weights)
        — used by the SpecMoE substitute forward, whose `selected` is remapped
        (winner → L2-nearest kept expert) rather than the gate's top-k. Builds
        the (num_tokens, num_experts) masks and dispatches; `scatter_add_` on
        weights correctly sums the case where two winners remap to one expert
        (matches the hf loop's index_add_). Returns (num_tokens, hidden)."""
        num_tokens = selected.shape[0]
        router_mask = torch.zeros(
            num_tokens, block.num_experts,
            dtype=torch.bool, device=hs_flat.device)
        router_mask.scatter_(1, selected, True)
        weights_mask = torch.zeros(
            num_tokens, block.num_experts,
            dtype=hs_flat.dtype, device=hs_flat.device)
        weights_mask.scatter_add_(1, selected, weights.to(hs_flat.dtype))
        block.expert_executor.dispatch_local(
            block.layer_id, hs_flat, router_mask, weights_mask)
        return block.expert_executor.wait_dispatch_local().to(hs_flat.dtype)

    def _standard_routing(self, block, hs_flat, gate_logits,
                          batch_size, sequence_length, hidden_dim):
        if _is_offload_block(block):
            out = self._route_offload(block, hs_flat, gate_logits)
            return out.reshape(batch_size, sequence_length, hidden_dim)

        # ── hf backend: per-expert loop (experts hold real weights) ──
        routing_weights = F.softmax(gate_logits, dim=1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(
            routing_weights, block.top_k, dim=-1)
        # Mixtral always renorms; Qwen3 has a config flag. Match the
        # block attr exactly; default True if somehow missing.
        if getattr(block, "norm_topk_prob", True):
            routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        routing_weights = routing_weights.to(hs_flat.dtype)

        final = torch.zeros(
            (batch_size * sequence_length, hidden_dim),
            dtype=hs_flat.dtype, device=hs_flat.device)
        expert_mask = F.one_hot(
            selected_experts, num_classes=block.num_experts).permute(2, 1, 0)
        expert_hit = torch.greater(
            expert_mask.sum(dim=(-1, -2)), 0).nonzero()
        for expert_idx in expert_hit:
            expert_layer = block.experts[expert_idx]
            idx, top_x = torch.where(expert_mask[expert_idx].squeeze(0))
            current_state = hs_flat[None, top_x].reshape(-1, hidden_dim)
            current_hidden = (
                expert_layer(current_state)
                * routing_weights[top_x, idx, None])
            final.index_add_(0, top_x, current_hidden.to(hs_flat.dtype))
        return final.reshape(batch_size, sequence_length, hidden_dim)

    def make_averaged_forward(self, controller, layer_idx, block):
        adapter = self

        def fwd(block, hidden_states):
            batch_size, sequence_length, hidden_dim = hidden_states.shape
            hs_flat = hidden_states.view(-1, hidden_dim)
            router_logits = block.gate(hs_flat)

            if controller.in_draft_phase:
                avg = controller.draft_cache.get(layer_idx)
                if avg is None:
                    avg = controller.draft.lazy_build(layer_idx, block, adapter)
                    if avg is not None:
                        controller.draft_cache[layer_idx] = avg
                if avg is not None:
                    if avg.get("kind") == "multi":
                        top_k = controller.draft.draft_top_k or block.top_k
                        gate_probs = router_logits.softmax(dim=-1)
                        out = adapter._route_multi_expert(
                            avg, gate_probs, hs_flat, top_k, block)
                    else:
                        out = adapter._run_dense_expert(avg, hs_flat)
                    return out.reshape(
                        batch_size, sequence_length, hidden_dim), router_logits
                # First cycle, no cache → fall through to standard routing.
            else:
                controller.draft.capture(layer_idx, router_logits)

            final = adapter._standard_routing(
                block, hs_flat, router_logits,
                batch_size, sequence_length, hidden_dim)
            return final, router_logits

        return fwd

    def make_masked_forward(self, controller, layer_idx, block):
        adapter = self

        def fwd(block, hidden_states):
            batch_size, sequence_length, hidden_dim = hidden_states.shape
            hs_flat = hidden_states.view(-1, hidden_dim)
            router_logits = block.gate(hs_flat)

            if controller.in_draft_phase:
                mask = controller.draft_cache.get(layer_idx)
                if mask is not None:
                    inactive = (~mask).to(router_logits.device)
                    gate_logits = router_logits.masked_fill(
                        inactive.unsqueeze(0), float("-inf"))
                else:
                    gate_logits = router_logits
            else:
                gate_logits = router_logits  # target verify: no capture

            final = adapter._standard_routing(
                block, hs_flat, gate_logits,
                batch_size, sequence_length, hidden_dim)
            return final, router_logits

        return fwd

    def make_substitute_forward(self, controller, layer_idx, block):
        return _topk_substitute_forward(controller, layer_idx, block)

    def expert_flat_weights(self, block):
        return [
            torch.cat([
                e.gate_proj.weight.detach().flatten().float(),
                e.up_proj.weight.detach().flatten().float(),
                e.down_proj.weight.detach().flatten().float(),
            ])
            for e in block.experts
        ]
