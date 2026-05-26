"""Per-model-family adapter ABC.

An adapter encapsulates everything specific to one MoE family:

  * `iter_moe(model)` — yield `(layer_idx, block)` for every MoE layer
  * `num_experts(block)` — expert count for that block
  * `default_count_top_k(model)` — used by count-based drafts when CLI/YAML
                                   leaves it unset
  * `build_weighted_avg(block, weights)` — `sum_e w_e * expert[e].*` per layer
  * `make_averaged_forward(controller, layer_idx, block)` — forward callable
        that runs the dense averaged expert in draft phase, captures /
        routes normally in target phase
  * `make_masked_forward(controller, layer_idx, block)` — forward callable
        that masks gate logits to a single expert in draft phase, routes
        normally otherwise
  * `post_load(model, tokenizer, args)` — model/tokenizer setup hook
        (e.g. GPT-OSS `reasoning_effort`)
"""

from __future__ import annotations

from typing import Any, Dict, Iterator, List, Tuple

import torch
import torch.nn as nn


class MoEAdapter:
    """Abstract MoE family adapter. See module docstring."""

    name: str = "abstract"

    def iter_moe(self, model: nn.Module) -> Iterator[Tuple[int, nn.Module]]:
        raise NotImplementedError

    def num_experts(self, block: nn.Module) -> int:
        raise NotImplementedError

    def default_count_top_k(self, model: nn.Module) -> int:
        raise NotImplementedError

    def build_weighted_avg(
        self, block: nn.Module, weights: List[float],
        *, cpu_block: Any = None,
    ) -> Dict[str, torch.Tensor]:
        """Build `sum_e w_e * expert[e]` for one MoE layer.

        On HF backend: `cpu_block=None`; read directly from
        `block.experts[e].w*.weight` (real tensors on GPU).

        On offload backend: `cpu_block` is the *corresponding* MoE block
        on a CPU-resident copy of the model. Stream
        `cpu_block.experts[e].w*.weight` CPU→GPU per non-zero-weight
        expert, accumulate into an fp32 buffer on the offloaded block's
        target device. The returned tensors live on GPU (where draft
        forward consumes them).
        """
        raise NotImplementedError

    def make_averaged_forward(self, controller, layer_idx: int, block: nn.Module):
        raise NotImplementedError

    def make_masked_forward(self, controller, layer_idx: int, block: nn.Module):
        raise NotImplementedError

    def post_load(self, model: nn.Module, tokenizer: Any, args: Any) -> None:
        pass
