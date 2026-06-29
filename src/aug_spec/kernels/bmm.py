"""Batched SwiGLU bmm for the merged "multi" experts.

Moved out of `adapters/base.py` (A5) — these are pure tensor kernels with no
adapter or draft knowledge, shared by the adapter merged-expert forward
(qwen3 / mixtral) and the offload path.
"""

from __future__ import annotations

from typing import Any, Dict, List

import torch
import torch.nn.functional as F


def stack_swiglu_weights(cache: Dict[str, Any],
                         experts: List[Dict[str, torch.Tensor]],
                         gate_key: str, up_key: str, down_key: str):
    """Stack K merged SwiGLU experts' weights, transposed as the right operand
    of ``bmm(hidden, w)``, memoised on the per-cycle `cache` dict:
      gate/up: [K, D, INTER]   down: [K, INTER, D].
    The cache dict is rebuilt each verify cycle, so the memo invalidates with
    the merged weights."""
    stk = cache.get("_bmm_stack")
    if stk is None:
        gate = torch.stack([e[gate_key] for e in experts]).transpose(1, 2)
        up = torch.stack([e[up_key] for e in experts]).transpose(1, 2)
        down = torch.stack([e[down_key] for e in experts]).transpose(1, 2)
        stk = (gate.contiguous(), up.contiguous(), down.contiguous())
        cache["_bmm_stack"] = stk
    return stk


def bmm_swiglu(hs_flat: torch.Tensor, gw: torch.Tensor,
               uw: torch.Tensor, dw: torch.Tensor) -> torch.Tensor:
    """[K, T, D] = SiLU(hs·gateᵀ) ⊙ (hs·upᵀ) · downᵀ for all K experts, where
    gw/uw are [K, D, INTER] and dw is [K, INTER, D]."""
    K, T = gw.shape[0], hs_flat.shape[0]
    hsK = hs_flat.unsqueeze(0).expand(K, T, -1)            # [K, T, D]
    hidden = F.silu(torch.bmm(hsK, gw)) * torch.bmm(hsK, uw)
    return torch.bmm(hidden, dw)                           # [K, T, D]
