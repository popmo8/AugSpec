"""GPT-OSS adapter (gpt-oss-20b, gpt-oss-120b).

Block path:    `model.model.layers[i].mlp` (filter by presence of `router`)
Expert MLP:    fused 3-D tensors `gate_up_proj` / `down_proj` (+ biases)
Activation:    clamped SwiGLU with `alpha` / `limit` constants
Native top-k:  `config.num_experts_per_tok` (= 4)
"""

from __future__ import annotations

from typing import Any, Optional

import torch
import torch.nn.functional as F

from .base import MoEAdapter, _svd_decompose, _svd_remerge, _weighted_sum


def _wrap_chat_template_with_reasoning(tokenizer, reasoning_effort: str):
    """gpt-oss chat templates accept a `reasoning_effort` kwarg. Specbench
    calls `apply_chat_template` without it, so wrap the method to inject
    the requested level."""
    orig = tokenizer.apply_chat_template

    def patched(*args, **kwargs):
        kwargs.setdefault("reasoning_effort", reasoning_effort)
        try:
            return orig(*args, **kwargs)
        except TypeError:
            kwargs.pop("reasoning_effort", None)
            return orig(*args, **kwargs)

    tokenizer.apply_chat_template = patched


class GptOssAdapter(MoEAdapter):
    name = "gptoss"

    def __init__(self):
        self._alpha: Optional[float] = None
        self._limit: Optional[float] = None

    def iter_moe(self, model):
        if not (hasattr(model, "model") and hasattr(model.model, "layers")):
            raise TypeError("Expected a GPT-OSS-style model with .model.layers.")
        for i, layer in enumerate(model.model.layers):
            mlp = getattr(layer, "mlp", None)
            if mlp is None:
                continue
            if hasattr(mlp, "router") and hasattr(mlp, "experts"):
                yield i, mlp

    def num_experts(self, block):
        return block.experts.gate_up_proj.shape[0]

    def default_count_top_k(self, model):
        return getattr(model.config, "num_experts_per_tok", 4)

    def post_load(self, model, tokenizer, args):
        reasoning = getattr(args, "reasoning_effort", None)
        if reasoning is not None:
            model.generation_config.reasoning_effort = reasoning
            _wrap_chat_template_with_reasoning(tokenizer, reasoning)
        # Cache activation-function constants from the first MoE layer.
        for _, mlp in self.iter_moe(model):
            self._alpha = float(getattr(mlp.experts, "alpha", 1.702))
            self._limit = float(getattr(mlp.experts, "limit", 7.0))
            break

    def build_weighted_avg(self, mlp, weights):
        experts = mlp.experts
        n_experts = experts.gate_up_proj.shape[0]

        gate_up_ref = experts.gate_up_proj
        gate_up_bias_ref = experts.gate_up_proj_bias
        down_ref = experts.down_proj
        down_bias_ref = experts.down_proj_bias
        dtype = gate_up_ref.dtype

        gate_up_sum = torch.zeros(
            gate_up_ref.shape[1:], dtype=torch.float32,
            device=gate_up_ref.device)
        gate_up_bias_sum = torch.zeros(
            gate_up_bias_ref.shape[1:], dtype=torch.float32,
            device=gate_up_bias_ref.device)
        down_sum = torch.zeros(
            down_ref.shape[1:], dtype=torch.float32,
            device=down_ref.device)
        down_bias_sum = torch.zeros(
            down_bias_ref.shape[1:], dtype=torch.float32,
            device=down_bias_ref.device)

        for e_idx in range(n_experts):
            w = weights[e_idx]
            if w == 0.0:
                continue
            gate_up_sum.add_(gate_up_ref[e_idx].float(), alpha=w)
            gate_up_bias_sum.add_(gate_up_bias_ref[e_idx].float(), alpha=w)
            down_sum.add_(down_ref[e_idx].float(), alpha=w)
            down_bias_sum.add_(down_bias_ref[e_idx].float(), alpha=w)

        out = {
            "gate_up_proj": gate_up_sum.to(dtype),
            "gate_up_proj_bias": gate_up_bias_sum.to(dtype),
            "down_proj": down_sum.to(dtype),
            "down_proj_bias": down_bias_sum.to(dtype),
        }
        del gate_up_sum, gate_up_bias_sum, down_sum, down_bias_sum
        return out

    def build_svd_basis(self, mlp, rank=256, store_dtype=torch.bfloat16):
        experts = mlp.experts
        n = experts.gate_up_proj.shape[0]
        # Weight matrices get an SVD basis; biases are 1-D (no subspace), so
        # we keep the static per-expert bias tensors for a per-cycle plain avg.
        return {
            "dtype": experts.gate_up_proj.dtype,
            "gate_up": _svd_decompose(
                [experts.gate_up_proj[e].float() for e in range(n)],
                rank, store_dtype),
            "down": _svd_decompose(
                [experts.down_proj[e].float() for e in range(n)],
                rank, store_dtype),
            "gate_up_bias": [experts.gate_up_proj_bias[e] for e in range(n)],
            "down_bias":    [experts.down_proj_bias[e] for e in range(n)],
        }

    def build_svd_from_basis(self, basis, weights):
        dtype = basis["dtype"]
        return {
            "gate_up_proj":      _svd_remerge(basis["gate_up"], weights).to(dtype),
            "gate_up_proj_bias": _weighted_sum(basis["gate_up_bias"], weights).to(dtype),
            "down_proj":         _svd_remerge(basis["down"], weights).to(dtype),
            "down_proj_bias":    _weighted_sum(basis["down_bias"], weights).to(dtype),
        }

    def _run_dense_expert(self, avg, flat):
        alpha = self._alpha
        limit = self._limit
        gate_up = flat @ avg["gate_up_proj"] + avg["gate_up_proj_bias"]
        gate = gate_up[..., ::2].clamp(max=limit)
        up = gate_up[..., 1::2].clamp(min=-limit, max=limit)
        glu = gate * torch.sigmoid(gate * alpha)
        gated = (up + 1) * glu
        return gated @ avg["down_proj"] + avg["down_proj_bias"]

    def make_averaged_forward(self, controller, layer_idx, mlp):
        adapter = self

        def fwd(mlp, hidden_states):
            batch_size, sequence_length, hidden_dim = hidden_states.shape
            flat = hidden_states.reshape(-1, hidden_dim)

            if controller.in_draft_phase:
                avg = controller.draft_cache.get(layer_idx)
                if avg is None:
                    avg = controller.draft.lazy_build(layer_idx, mlp, adapter)
                    if avg is not None:
                        controller.draft_cache[layer_idx] = avg
                if avg is not None:
                    if avg.get("kind") == "multi":
                        # Run the gate (one linear) for per-token cluster scores.
                        router = mlp.router
                        gate_logits = F.linear(flat, router.weight, router.bias)
                        gate_probs = gate_logits.softmax(dim=-1)
                        top_k = controller.draft.draft_top_k or router.top_k
                        out = adapter._route_multi_expert(
                            avg, gate_probs, flat, top_k)
                    else:
                        out = adapter._run_dense_expert(avg, flat)
                    return out.reshape(batch_size, sequence_length, hidden_dim), None
                # First cycle, no cache → fall through to mlp.router/experts.
                router_scores, router_indices = mlp.router(hidden_states)
                routed_out = mlp.experts(
                    hidden_states,
                    router_indices=router_indices,
                    routing_weights=router_scores,
                )
                return routed_out, router_scores

            # Target verify (or prefill): compute router logits once, use
            # them for both the scorer and the top-k dispatch (avoid
            # running router.weight twice on the same hidden states).
            router = mlp.router
            router_logits = F.linear(flat, router.weight, router.bias)
            full_softmax = router_logits.softmax(dim=-1)
            controller.draft.capture_softmax(layer_idx, full_softmax)

            top_val, router_indices = torch.topk(
                router_logits, router.top_k, dim=-1)
            top_val = F.softmax(top_val, dim=1, dtype=top_val.dtype)
            router_scores = torch.zeros_like(router_logits).scatter_(
                1, router_indices, top_val)
            routed_out = mlp.experts(
                hidden_states,
                router_indices=router_indices,
                routing_weights=router_scores,
            )
            return routed_out, router_scores

        return fwd

    def make_masked_forward(self, controller, layer_idx, mlp):
        raise NotImplementedError(
            "GPT-OSS masked forward is not implemented (no current experiment "
            "needs it). Add it here when one does.")
