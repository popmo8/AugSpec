"""Low-level compute kernels shared across adapters / drafts (no model or
strategy knowledge). Currently the batched SwiGLU bmm used by the merged-expert
draft forward."""

from __future__ import annotations

from .bmm import bmm_swiglu, stack_swiglu_weights

__all__ = ["bmm_swiglu", "stack_swiglu_weights"]
