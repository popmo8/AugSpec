"""Offload-merge engine — the isolated home for offload-backend merge
optimisations, gated by the `merge_offload` config flag.

Why this exists
---------------
The merge-based drafts (count / softmax / topm_count / prefill variants — any
`ScoreBasedAvgDraft`) and the offload backend share a lot of generic code.
Optimisations that only make sense for "merge ON offload" — GPU resident merge,
merge↔PCIe overlap, flushing merged experts after the draft phase to reclaim
workspace — must NOT leak into those shared methods or they would risk the hf
backend and the non-merge drafts.

This engine collects all of that offload-merge-specific *policy* (when to merge,
how to overlap, when to free) in one place. The shared pipeline only ever calls
it through a few guarded hooks, and only when `merge_offload=true` builds one —
so for every other method the engine is `None` and behaviour is unchanged.

Layering (see offload_plan.md M9b):
  * Engine  — offload-merge execution + GPU-memory lifecycle + overlap timing
  * Draft   — merge policy (which experts, how to cluster); calls the engine
  * Adapter — model-specific merge primitive (how qwen3 vs mixtral combine)

Shell status
------------
`build()` currently delegates straight to `adapter.build_weighted_avg` (whose
offload branch already does the GPU resident merge), so wiring the engine in is
a behaviour-preserving refactor. `on_verify_layer` / `on_draft_end` are stubs;
upcoming optimisations grow into them without touching the shared pipeline.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import torch.nn as nn


class OffloadMergeEngine:
    """Per-run owner of the offload-merge optimisations.

    Constructed by the Controller only when `merge_offload` is set on an
    offload backend; `attach` then tags every MoE block with a back-reference
    so the draft's `_build_one` can route through `build` without threading the
    engine through every call signature.
    """

    def __init__(self, adapter, model: nn.Module):
        self.adapter = adapter
        self.model = model
        self.device = getattr(model, "device", None)
        # layer_idx → blocks, filled by attach(); used by the lifecycle hooks.
        self.blocks: List[Tuple[int, nn.Module]] = []

    # ── setup ───────────────────────────────────────────────────────────
    def attach(self, blocks: List[Tuple[int, nn.Module]]) -> None:
        """Tag each MoE block with a back-reference to this engine so the
        shared `_build_one` / verify hooks can reach it off the block (same
        pattern as `_cpu_merge_source`)."""
        self.blocks = list(blocks)
        for _, block in self.blocks:
            block._merge_engine = self

    # ── merge execution ─────────────────────────────────────────────────
    def build(self, block: nn.Module, weights: List[float]) -> Dict[str, Any]:
        """Build one merged dense expert for `block` from per-expert `weights`.

        Shell: delegates to the adapter's offload merge (GPU resident merge via
        the archer dispatcher when available, CPU-source fallback otherwise).
        Future optimisations (resident-aware scheduling, merge↔PCIe overlap,
        flush bookkeeping) move in here.
        """
        return self.adapter.build_weighted_avg(block, weights)

    # ── lifecycle hooks (stubs — grow with each optimisation) ───────────
    def on_verify_layer(self, layer_idx: int, block: nn.Module) -> None:
        """Called from the offload verify routing once layer `layer_idx`'s
        experts are GPU-resident (post-dispatch, pre-evict). Reserved for
        during-verify per-layer merge that overlaps with the next layer's
        PCIe fetch. No-op in the shell."""
        pass

    def on_draft_end(self) -> None:
        """Called at the draft→verify transition. Reserved for flushing the
        merged experts built this cycle so the freed budget can serve verify
        as a larger expert workspace (offload_plan.md §1.4). No-op in the
        shell."""
        pass
