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

from .base import MoEAdapter


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
        ref_g = block.experts[0].gate_proj.weight
        ref_u = block.experts[0].up_proj.weight
        ref_d = block.experts[0].down_proj.weight
        dtype = ref_g.dtype

        g_sum = torch.zeros_like(ref_g, dtype=torch.float32)
        u_sum = torch.zeros_like(ref_u, dtype=torch.float32)
        d_sum = torch.zeros_like(ref_d, dtype=torch.float32)

        for e_idx, expert in enumerate(block.experts):
            w = weights[e_idx]
            if w == 0.0:
                continue
            g_sum.add_(expert.gate_proj.weight.float(), alpha=w)
            u_sum.add_(expert.up_proj.weight.float(), alpha=w)
            d_sum.add_(expert.down_proj.weight.float(), alpha=w)

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
