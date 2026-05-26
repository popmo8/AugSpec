"""Mixtral adapter (Mixtral-8x7B, Mixtral-8x22B, …).

Block path:    `model.model.layers[i].block_sparse_moe`
Expert MLP:    `block.experts[e].{w1, w2, w3}` (SwiGLU)
Native top-k:  `config.num_experts_per_tok` (= 2)

Backend branching (in `_standard_routing`):
  * HF backend  → `MixtralSparseMoeBlock`         → `_route_hf`
                  (hand-rolled per-expert for-loop reading real weights)
  * Offload     → `SyncMixtralSparseMoeBlock`     → `_route_offload`
                  (delegates to `block.expert_executor.dispatch_local`,
                   which streams experts through the archer cache)
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .base import MoEAdapter


def _is_offload_block(block) -> bool:
    """Detect moe_infinity's offloaded MoE block by name (avoids
    unconditionally importing moe_infinity from the GPU-backend path)."""
    return type(block).__name__ == "SyncMixtralSparseMoeBlock"


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

    def build_weighted_avg(self, block, weights, *, cpu_block=None):
        """Build the merged dense expert for one MoE layer.

        - HF backend (`cpu_block=None`): read real expert weights from
          `block.experts[e].w*.weight` (already on GPU), accumulate fp32.
        - Offload backend (`cpu_block` set): read from `cpu_block` (CPU
          RAM, real bf16 weights), stream CPU→GPU per non-zero-weight
          expert, accumulate into fp32 buffer on the offloaded block's
          target device.

        Peak GPU memory during the call: 1 fp32 accumulator (3 weights)
        + 1 bf16 transient (3 weights) ≈ 1.75 expert sizes — independent
        of how many experts contribute (streaming pattern).
        """
        # Pick the source for expert weight reads and the target device
        # for the merged tensor.
        if cpu_block is not None:
            source = cpu_block
            # Merged expert must live on GPU (where draft forward consumes
            # it). In offload mode moe_infinity moves ALL dense params
            # (including `block.gate.weight`) to CPU — they're staged back
            # to GPU only inside forward via a just-in-time hook. So at
            # build-time both `block.experts[*].w1` (placeholder) and
            # `block.gate.weight` (CPU) lie. Single-GPU constraint
            # (IMPLEMENTATION_PLAN §1) makes this unambiguous: target cuda.
            device = torch.device("cuda", torch.cuda.current_device())
        else:
            source = block
            device = block.experts[0].w1.weight.device

        ref_w1 = source.experts[0].w1.weight
        ref_w2 = source.experts[0].w2.weight
        ref_w3 = source.experts[0].w3.weight
        dtype = ref_w1.dtype

        w1_sum = torch.zeros(ref_w1.shape, dtype=torch.float32, device=device)
        w2_sum = torch.zeros(ref_w2.shape, dtype=torch.float32, device=device)
        w3_sum = torch.zeros(ref_w3.shape, dtype=torch.float32, device=device)

        for e_idx, w in enumerate(weights):
            if w == 0.0:
                continue
            # In HF mode source is on GPU; .to(device) is a no-op.
            # In offload mode source is on CPU; .to(device) is the H2D copy.
            w1 = source.experts[e_idx].w1.weight.to(device, non_blocking=True).float()
            w2 = source.experts[e_idx].w2.weight.to(device, non_blocking=True).float()
            w3 = source.experts[e_idx].w3.weight.to(device, non_blocking=True).float()
            w1_sum.add_(w1, alpha=w)
            w2_sum.add_(w2, alpha=w)
            w3_sum.add_(w3, alpha=w)
            del w1, w2, w3

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
        """Offload-backend routing: delegate to `block.expert_executor`.

        Mirror of `SyncMixtralSparseMoeBlock.forward` (see
        moe_infinity/moe_infinity/models/mixtral.py) but operates on
        the gate logits we've already computed upstream, and returns
        the per-token output reshaped to the original 3-D shape.
        """
        routing_weights = F.softmax(gate_logits, dim=1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(
            routing_weights, block.top_k, dim=-1)
        routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        routing_weights = routing_weights.to(hs_flat.dtype)

        router_mask = F.one_hot(
            selected_experts, num_classes=block.num_experts)
        routing_weights_mask = (
            routing_weights[:, :, None] * router_mask).permute(0, 2, 1)
        router_mask = router_mask.permute(0, 2, 1)
        # Mixtral is top-2; matches SyncMixtralSparseMoeBlock.forward.
        router_mask = torch.logical_or(
            router_mask[:, :, 0], router_mask[:, :, 1])
        routing_weights_mask = torch.sum(routing_weights_mask, dim=-1)

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
