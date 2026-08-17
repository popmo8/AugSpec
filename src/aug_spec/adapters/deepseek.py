"""DeepSeek-MoE v1 adapter (deepseek-moe-16b-base / -chat).

Block path:    `model.model.layers[i].mlp` — MoE layers only. Layer 0 is a
               dense `DeepseekMLP` without `.gate`/`.experts`
               (config.first_k_dense_replace = 1) and is skipped by the same
               presence filter qwen3 uses for its dense fallback.
Expert MLP:    `gate_proj` / `up_proj` / `down_proj` (SwiGLU; 64 fine-grained
               routed experts of moe_intermediate_size 1408).
Shared experts:`block.shared_experts` — ONE widened DeepseekMLP
               (n_shared_experts × 1408). Every token passes through it in
               BOTH phases; like attention weights it sits outside the
               routed-expert draft budget, and its output is ADDED to the
               routed output (y = routed + shared(identity)).
Gate:          `block.gate` is a MoEGate MODULE returning
               (topk_idx, topk_weight, aux_loss), NOT logits — every forward
               here recomputes the raw logits as
               `F.linear(hs.float(), gate.weight.float())`, matching
               MoEGate's fp32 scoring exactly, then softmax → top-k.
               `norm_topk_prob` lives on the gate (False on 16b); native
               top-k = config.num_experts_per_tok (6).
Return shape:  `DeepseekDecoderLayer` does `h = self.mlp(h)` — the swapped
               forwards return a SINGLE tensor (no router-logits tuple,
               unlike qwen3/mixtral).
Backend:       hf only (trust_remote_code model family; no offload port).
act-sim:       hybrid clustering's hf prefill capture is implemented inline
               in the verify expert loop — the raw per-expert outputs are
               already computed there, so capture is free (same tuple format
               as gptoss's `_fired_expert_outputs` path).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from aug_spec.kernels.bmm import stack_swiglu_weights

from .base import MoEAdapter


class DeepseekMoeAdapter(MoEAdapter):
    name = "deepseek_moe"

    def iter_moe(self, model):
        if not (hasattr(model, "model") and hasattr(model.model, "layers")):
            raise TypeError("Expected a DeepSeek-MoE model with .model.layers.")
        for i, layer in enumerate(model.model.layers):
            block = getattr(layer, "mlp", None)
            if block is None:
                continue
            if hasattr(block, "gate") and hasattr(block, "experts"):
                yield i, block

    def num_experts(self, block):
        # Routed experts only — `shared_experts` is always-dense and not part
        # of the draft/merge domain.
        return len(block.experts)

    def default_count_top_k(self, model):
        return getattr(model.config, "num_experts_per_tok", 6)

    # ── gate helpers (MoEGate is a module, not a Linear) ─────────────────
    @staticmethod
    def _gate_logits(block, hs_flat):
        # MoEGate scores in fp32 (`F.linear(h.float(), weight.float())`);
        # match it exactly so verify routing is target-exact.
        return F.linear(hs_flat.type(torch.float32),
                        block.gate.weight.type(torch.float32))

    @staticmethod
    def _native_top_k(block):
        return getattr(block.gate, "top_k",
                       getattr(block, "num_experts_per_tok", 6))

    @staticmethod
    def _norm_topk(block):
        return bool(getattr(block.gate, "norm_topk_prob", False))

    @staticmethod
    def _shared_out(block, hidden_states):
        shared = getattr(block, "shared_experts", None)
        return None if shared is None else shared(hidden_states)

    # ── merge / dense-expert kernels (SwiGLU, llama-style attr names) ────
    def build_weighted_avg(self, block, weights):
        ref_g = block.experts[0].gate_proj.weight
        ref_u = block.experts[0].up_proj.weight
        ref_d = block.experts[0].down_proj.weight
        dtype = ref_g.dtype

        g_sum = torch.zeros_like(ref_g, dtype=torch.float32)
        u_sum = torch.zeros_like(ref_u, dtype=torch.float32)
        d_sum = torch.zeros_like(ref_d, dtype=torch.float32)
        for e_idx, w in enumerate(weights):
            if w == 0.0:
                continue
            expert = block.experts[e_idx]
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
        gate = F.linear(hs_flat, avg["gate_proj"])
        up = F.linear(hs_flat, avg["up_proj"])
        hidden = F.silu(gate) * up
        return F.linear(hidden, avg["down_proj"])

    def _swiglu_stack(self, cache, experts):
        return stack_swiglu_weights(
            cache, experts, "gate_proj", "up_proj", "down_proj")

    def expert_flat_weights(self, block):
        return [
            torch.cat([
                e.gate_proj.weight.detach().flatten().float(),
                e.up_proj.weight.detach().flatten().float(),
                e.down_proj.weight.detach().flatten().float(),
            ])
            for e in block.experts
        ]

    # ── expert execution ─────────────────────────────────────────────────
    def _expert_loop(self, block, hs_flat, selected, routing_weights,
                     act_sim_capture=None, layer_idx=-1):
        """hf per-expert loop over explicit (selected, weights) — shared by
        standard routing and the SpecMoE substitute path. When
        `act_sim_capture` is a list, the raw (unweighted) outputs of every
        fired expert are appended to it in the engine's captured-tuple
        format (hybrid's hf prefill act-sim, drafts/base.py)."""
        num_tokens, hidden_dim = hs_flat.shape
        final = torch.zeros(
            (num_tokens, hidden_dim),
            dtype=hs_flat.dtype, device=hs_flat.device)
        expert_mask = F.one_hot(
            selected, num_classes=len(block.experts)).permute(2, 1, 0)
        expert_hit = torch.greater(
            expert_mask.sum(dim=(-1, -2)), 0).nonzero()
        for expert_idx in expert_hit:
            expert_layer = block.experts[expert_idx]
            idx, top_x = torch.where(expert_mask[expert_idx].squeeze(0))
            current_state = hs_flat[None, top_x].reshape(-1, hidden_dim)
            raw = expert_layer(current_state)
            if act_sim_capture is not None:
                act_sim_capture.append(
                    (layer_idx, int(expert_idx), top_x, raw.detach()))
            current_hidden = raw * routing_weights[top_x, idx, None]
            final.index_add_(0, top_x, current_hidden.to(hs_flat.dtype))
        return final

    def _standard_routing(self, controller, layer_idx, block, hs_flat,
                          gate_logits, batch_size, sequence_length,
                          hidden_dim, want_act_sim=False):
        # gate_logits is already fp32 (see _gate_logits); softmax → topk →
        # optional renorm reproduces MoEGate inference routing.
        scores = F.softmax(gate_logits, dim=1, dtype=torch.float)
        routing_weights, selected = torch.topk(
            scores, self._native_top_k(block), dim=-1)
        if self._norm_topk(block):
            routing_weights = routing_weights / (
                routing_weights.sum(dim=-1, keepdim=True) + 1e-20)
        routing_weights = routing_weights.to(hs_flat.dtype)

        capture = None
        if want_act_sim:
            wants = getattr(controller.draft, "wants_prefill_act_sim", None)
            if wants is not None and wants(layer_idx):
                capture = []
        final = self._expert_loop(block, hs_flat, selected, routing_weights,
                                  act_sim_capture=capture,
                                  layer_idx=layer_idx)
        if capture:
            controller.draft.accumulate_prefill_act_sim(
                layer_idx, capture, len(block.experts))
        return final.reshape(batch_size, sequence_length, hidden_dim)

    # ── skip-return conventions (speed / draft_verify / dv_search) ───────
    def mlp_skip_output(self, hidden_states):
        # DeepseekDecoderLayer does `h = self.mlp(h)` — tensor alone, no
        # (out, router_logits) unpack.
        return torch.zeros_like(hidden_states)

    def decoder_skip_output(self, hidden_states, *args, **kwargs):
        # 4.36-style tuple contract: the model loop reads layer_outputs[0],
        # and with use_cache also [2 if output_attentions else 1] for the
        # (pass-through) cache object.
        out = (hidden_states,)
        if kwargs.get("output_attentions"):
            out += (None,)
        if kwargs.get("use_cache"):
            out += (kwargs.get("past_key_value"),)
        return out

    # ── swapped forwards (single-tensor return; shared experts added) ────
    def make_averaged_forward(self, controller, layer_idx, block):
        adapter = self

        def fwd(block, hidden_states):
            batch_size, sequence_length, hidden_dim = hidden_states.shape
            hs_flat = hidden_states.view(-1, hidden_dim)
            router_logits = adapter._gate_logits(block, hs_flat)

            if controller.in_draft_phase:
                avg = controller.draft_cache.get(layer_idx)
                if avg is None:
                    avg = controller.draft.lazy_build(layer_idx, block, adapter)
                    if avg is not None:
                        controller.draft_cache[layer_idx] = avg
                # C-BOOT: see qwen3.py — missing draft cache under
                # run.prefill_warmup is a bug; fail fast, don't fall back.
                if avg is None and getattr(controller, "prefill_warmup", False):
                    raise RuntimeError(
                        f"draft cache missing for MoE layer {layer_idx} in "
                        f"draft phase despite run.prefill_warmup=true "
                        f"(merged_cache_plan.md §2.4)")
                if avg is not None:
                    if avg.get("kind") == "multi":
                        top_k = (controller.draft.draft_top_k
                                 or adapter._native_top_k(block))
                        gate_probs = router_logits.softmax(
                            dim=-1).to(hs_flat.dtype)
                        out = adapter._route_multi_expert(
                            avg, gate_probs, hs_flat, top_k, block)
                    else:
                        out = adapter._run_dense_expert(avg, hs_flat)
                    out = out.reshape(
                        batch_size, sequence_length, hidden_dim)
                    shared = adapter._shared_out(block, hidden_states)
                    return out if shared is None else out + shared
                # prefill_warmup=false ablation: first cycle has no cache →
                # fall through to standard routing (legacy behaviour).
            else:
                controller.draft.capture(layer_idx, router_logits)

            final = adapter._standard_routing(
                controller, layer_idx, block, hs_flat, router_logits,
                batch_size, sequence_length, hidden_dim,
                want_act_sim=not controller.in_draft_phase)
            shared = adapter._shared_out(block, hidden_states)
            return final if shared is None else final + shared

        return fwd

    def make_masked_forward(self, controller, layer_idx, block):
        adapter = self

        def fwd(block, hidden_states):
            batch_size, sequence_length, hidden_dim = hidden_states.shape
            hs_flat = hidden_states.view(-1, hidden_dim)
            router_logits = adapter._gate_logits(block, hs_flat)

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
                controller, layer_idx, block, hs_flat, gate_logits,
                batch_size, sequence_length, hidden_dim)
            shared = adapter._shared_out(block, hidden_states)
            return final if shared is None else final + shared

        return fwd

    def make_substitute_forward(self, controller, layer_idx, block):
        # SpecMoE for the MoEGate layout: same semantics as
        # drafts/specmoe.topk_substitute_forward's hf path (route the natural
        # top-route_top_k in both phases; draft remaps winners through the
        # substitute table), plus the deepseek deltas — fp32 gate recompute,
        # shared-experts add, single-tensor return. No offload/bmm branches
        # (hf-only family).
        adapter = self

        def fwd(block, hidden_states):
            batch_size, sequence_length, hidden_dim = hidden_states.shape
            hs_flat = hidden_states.view(-1, hidden_dim)
            router_logits = adapter._gate_logits(block, hs_flat)
            full_softmax = F.softmax(router_logits, dim=1, dtype=torch.float)

            k = controller.draft.route_top_k
            routing_weights, selected = torch.topk(full_softmax, k, dim=-1)

            if controller.in_draft_phase:
                sub_table = controller.draft_cache.get(layer_idx)
                if sub_table is not None:
                    selected = sub_table.to(selected.device)[selected]
            else:
                controller.draft.capture(layer_idx, full_softmax)

            if adapter._norm_topk(block):
                routing_weights = routing_weights / (
                    routing_weights.sum(dim=-1, keepdim=True) + 1e-20)
            routing_weights = routing_weights.to(hs_flat.dtype)

            final = adapter._expert_loop(
                block, hs_flat, selected, routing_weights).reshape(
                batch_size, sequence_length, hidden_dim)
            shared = adapter._shared_out(block, hidden_states)
            return final if shared is None else final + shared

        return fwd
