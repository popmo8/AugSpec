"""Qwen3-MoE adapter (Qwen3-30B-A3B-Base et al.).

Block path:    `model.model.layers[i].mlp` (filter by presence of
               `.gate` + `.experts` — Qwen3 reuses `layer.mlp` for the
               dense fallback)
Expert MLP:    `gate_proj` / `up_proj` / `down_proj` (SwiGLU, same shape
               as Mixtral but different attribute names)
Native top-k:  `config.num_experts_per_tok` (= 8 for A3B)
Note:          `block.norm_topk_prob` toggles the routing-weight renorm
               (True on A3B; HF default is False — match the block flag).

Backend branching (in `_standard_routing`):
  * HF backend → `Qwen3MoeSparseMoeBlock` → `_route_hf` (hand-rolled
                 per-expert loop)
  * Offload    → `Qwen3MoEBlock`          → `_route_offload`
                 (uses block.lib.topk_softmax CUDA kernel + dispatch_local)
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .base import MoEAdapter


def _is_offload_block(block) -> bool:
    """Detect moe_infinity's offloaded Qwen3 MoE block by class name."""
    return type(block).__name__ == "Qwen3MoEBlock"


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

    def build_weighted_avg(self, block, weights, *, cpu_block=None):
        """Build the merged dense expert. See base.py for contract.

        On offload backend (`cpu_block` set): stream from CPU into a fp32
        accumulator on GPU; the offloaded `block.experts[*]` weights are
        shape-(1,) placeholders.
        """
        if cpu_block is not None:
            source = cpu_block
            device = torch.device("cuda", torch.cuda.current_device())
        else:
            source = block
            device = block.experts[0].gate_proj.weight.device

        ref_g = source.experts[0].gate_proj.weight
        ref_u = source.experts[0].up_proj.weight
        ref_d = source.experts[0].down_proj.weight
        dtype = ref_g.dtype

        g_sum = torch.zeros(ref_g.shape, dtype=torch.float32, device=device)
        u_sum = torch.zeros(ref_u.shape, dtype=torch.float32, device=device)
        d_sum = torch.zeros(ref_d.shape, dtype=torch.float32, device=device)

        for e_idx, w in enumerate(weights):
            if w == 0.0:
                continue
            g = source.experts[e_idx].gate_proj.weight.to(device, non_blocking=True).float()
            u = source.experts[e_idx].up_proj.weight.to(device, non_blocking=True).float()
            d = source.experts[e_idx].down_proj.weight.to(device, non_blocking=True).float()
            g_sum.add_(g, alpha=w)
            u_sum.add_(u, alpha=w)
            d_sum.add_(d, alpha=w)
            del g, u, d

        out = {
            "gate_proj": g_sum.to(dtype),
            "up_proj": u_sum.to(dtype),
            "down_proj": d_sum.to(dtype),
        }
        del g_sum, u_sum, d_sum
        return out

    def _run_dense_expert(self, avg, hs_flat):
        # Qwen3MoeMLP: SiLU(gate_proj · h) ⊙ (up_proj · h) → down_proj(...).
        gate = F.linear(hs_flat, avg["gate_proj"])
        up = F.linear(hs_flat, avg["up_proj"])
        hidden = F.silu(gate) * up
        return F.linear(hidden, avg["down_proj"])

    def _standard_routing(self, block, hs_flat, gate_logits,
                          batch_size, sequence_length, hidden_dim):
        """Dispatch tokens through real experts. Branches on backend."""
        if _is_offload_block(block):
            return self._route_offload(
                block, hs_flat, gate_logits,
                batch_size, sequence_length, hidden_dim)
        return self._route_hf(
            block, hs_flat, gate_logits,
            batch_size, sequence_length, hidden_dim)

    def _route_hf(self, block, hs_flat, gate_logits,
                  batch_size, sequence_length, hidden_dim):
        """HF-backend routing: hand-rolled per-expert loop."""
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

    def _route_offload(self, block, hs_flat, gate_logits,
                       batch_size, sequence_length, hidden_dim):
        """Offload-backend routing: mirror of Qwen3MoEBlock.forward.

        Uses moe_infinity's `block.lib.topk_softmax` CUDA kernel to
        build (router_mask, routing_weights_mask) in one go — this is
        what their validated Qwen3 path uses. Then delegate to
        `block.expert_executor.dispatch_local`.
        """
        router_mask, routing_weights_mask = block.lib.topk_softmax(gate_logits)
        block.expert_executor.dispatch_local(
            block.layer_id, hs_flat, router_mask, routing_weights_mask)
        final = block.expert_executor.wait_dispatch_local()
        return final.view(
            batch_size, sequence_length, hidden_dim).to(hs_flat.dtype)

    def make_averaged_forward(self, controller, layer_idx, block):
        adapter = self

        def fwd(block, hidden_states):
            batch_size, sequence_length, hidden_dim = hidden_states.shape
            hs_flat = hidden_states.view(-1, hidden_dim)
            router_logits = block.gate(hs_flat)

            if controller.in_draft_phase:
                avg = controller.draft_cache.get(layer_idx)
                if avg is None:
                    cpu_block = (
                        controller.cpu_blocks.get(layer_idx)
                        if controller.cpu_blocks else None)
                    avg = controller.draft.lazy_build(
                        layer_idx, block, adapter, cpu_block=cpu_block)
                    if avg is not None:
                        controller.draft_cache[layer_idx] = avg
                if avg is not None:
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
