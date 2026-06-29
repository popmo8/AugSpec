"""Mixtral adapter (Mixtral-8x7B, Mixtral-8x22B, …).

Block path:    `model.model.layers[i].block_sparse_moe`
Expert MLP:    `block.experts[e].{w1, w2, w3}` (SwiGLU)
Native top-k:  `config.num_experts_per_tok` (= 2)
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from aug_spec.kernels.bmm import stack_swiglu_weights

from .base import MoEAdapter


class MixtralAdapter(MoEAdapter):
    name = "mixtral"

    def iter_moe(self, model):
        if not (hasattr(model, "model") and hasattr(model.model, "layers")):
            raise TypeError("Expected a Mixtral-style model with .model.layers.")
        for i, layer in enumerate(model.model.layers):
            block = getattr(layer, "block_sparse_moe", None)
            if block is None:
                continue
            if hasattr(block, "gate") and hasattr(block, "experts"):
                yield i, block

    def num_experts(self, block):
        return len(block.experts)

    def default_count_top_k(self, model):
        return getattr(model.config, "num_experts_per_tok", 2)

    def build_weighted_avg(self, block, weights):
        ref_w1 = block.experts[0].w1.weight
        ref_w2 = block.experts[0].w2.weight
        ref_w3 = block.experts[0].w3.weight
        dtype = ref_w1.dtype

        w1_sum = torch.zeros_like(ref_w1, dtype=torch.float32)
        w2_sum = torch.zeros_like(ref_w2, dtype=torch.float32)
        w3_sum = torch.zeros_like(ref_w3, dtype=torch.float32)

        for e_idx, expert in enumerate(block.experts):
            w = weights[e_idx]
            if w == 0.0:
                continue
            w1_sum.add_(expert.w1.weight.float(), alpha=w)
            w2_sum.add_(expert.w2.weight.float(), alpha=w)
            w3_sum.add_(expert.w3.weight.float(), alpha=w)

        out = {
            "w1": w1_sum.to(dtype),
            "w2": w2_sum.to(dtype),
            "w3": w3_sum.to(dtype),
        }
        del w1_sum, w2_sum, w3_sum
        return out

    def _run_dense_expert(self, avg, hs_flat):
        # Mixtral expert: SiLU(w1 · h) ⊙ (w3 · h) → w2(...).
        gate = F.linear(hs_flat, avg["w1"])
        up = F.linear(hs_flat, avg["w3"])
        hidden = F.silu(gate) * up
        return F.linear(hidden, avg["w2"])

    def _swiglu_stack(self, cache, experts):
        # gate=w1, up=w3, down=w2, stacked + transposed for bmm.
        return stack_swiglu_weights(cache, experts, "w1", "w3", "w2")

    def _standard_routing(self, block, hs_flat, gate_logits,
                          batch_size, sequence_length, hidden_dim):
        routing_weights = F.softmax(gate_logits, dim=1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(
            routing_weights, block.top_k, dim=-1)
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
        # Lazy import: the SpecMoE forward lives in drafts/specmoe.py (A5);
        # importing it at module top would cycle (adapters <-> drafts).
        from aug_spec.drafts.specmoe import topk_substitute_forward
        return topk_substitute_forward(controller, layer_idx, block)

    def expert_flat_weights(self, block):
        return [
            torch.cat([
                e.w1.weight.detach().flatten().float(),
                e.w2.weight.detach().flatten().float(),
                e.w3.weight.detach().flatten().float(),
            ])
            for e in block.experts
        ]
