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

    def __init__(self, adapter, model: nn.Module, during_verify: bool = False,
                 flush: bool = False, overlap: bool = False):
        self.adapter = adapter
        self.model = model
        self.device = getattr(model, "device", None)
        # P3: build the merged draft per layer *during* verify (on_verify_layer)
        # instead of after-verify (draft.refresh). 0 re-fetch — experts are still
        # resident from this layer's dispatch. Ablation flag merge_during_verify.
        self.during_verify = during_verify
        # P1: phase-exclusive flush (flush_on_draft_end). At draft start flush the
        # archer expert cache (idle during the merged-dense draft); at draft end
        # flush the merged experts (dead during verify). So merged residency and
        # the verify cache never coexist → peak = max(them), not sum (§1.4).
        self.flush = flush
        # P4: run the per-layer merge on a side CUDA stream and defer that
        # layer's evict by one layer, so the merge overlaps with the next
        # layer's PCIe fetch (instead of a per-layer full device sync). Ablation
        # flag merge_overlap.
        self.overlap = overlap
        self._merge_stream = None                 # lazily created side stream
        self._pending = None                      # (layer_idx, event) deferred evict
        # Back-reference to the Controller (set by it after construction): gives
        # on_verify_layer access to the draft (per-layer merge logic) + draft_cache.
        self.controller = None
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
        experts are GPU-resident (post-dispatch, pre-evict).

        P1: build this layer's merged draft NOW, while its top-M experts (a
        subset of what verify just dispatched) are resident → the merge reads
        them at zero PCIe. Uses the count the draft captured for this layer
        earlier in the same forward (capture runs before _route_offload), so
        it is the exact same weights `refresh` would have used after verify —
        only the timing (and residency) differs, hence acceptance is identical.
        No-op unless `during_verify`.
        """
        if not self.during_verify or self.controller is None:
            return
        draft = self.controller.draft
        score = getattr(draft, "target_score", {}).get(layer_idx)
        if score is None or not hasattr(draft, "_refresh_layer"):
            return
        disp = self._dispatcher()
        import torch

        def _merge():
            draft._refresh_layer(self.adapter, block, layer_idx, score,
                                 self.controller.draft_cache)

        if not self.overlap or disp is None:
            # P2 (synchronous): merge on the default stream, sync its async
            # reads, then evict this layer. Evicting each layer once verify+merge
            # are done keeps the sparse cache from filling → no overload
            # evict-after-use → no concurrent-eviction race (the P3 crash),
            # footprint ~1 layer.
            _merge()
            if disp is not None:
                torch.cuda.synchronize()
                disp.evict_layer(layer_idx, 0)
            return

        # P4: run the merge on a side stream so it overlaps the next layer's PCIe
        # fetch (default stream), and defer this layer's evict by one layer — by
        # the time we evict L, layer L+1's fetch has hidden L's merge, so the
        # event is already complete (no critical-path sync). The merge result in
        # draft_cache is read only in the next draft phase, after on_draft_start
        # syncs the side stream.
        if self._merge_stream is None:
            self._merge_stream = torch.cuda.Stream()
        with torch.cuda.stream(self._merge_stream):
            _merge()
        ev = torch.cuda.Event()
        ev.record(self._merge_stream)
        if self._pending is not None:
            pl, pe = self._pending
            pe.synchronize()           # done by now (overlapped with this layer)
            disp.evict_layer(pl, 0)
        self._pending = (layer_idx, ev)

    def _dispatcher(self):
        """The archer ExpertDispatcher (shared across blocks), or None."""
        if not self.blocks:
            return None
        ex = getattr(self.blocks[0][1], "expert_executor", None)
        return getattr(ex, "expert_dispatcher", None) if ex is not None else None

    def _drain_pending(self) -> None:
        """P4: finish any deferred (side-stream) merge + evict its layer. Called
        at the verify→draft boundary so (a) the last layer's evict isn't left
        hanging and (b) the side stream is synced before the draft reads the
        merged experts off the default stream. No-op unless overlap ran."""
        if self._merge_stream is not None:
            self._merge_stream.synchronize()
        if self._pending is not None:
            disp = self._dispatcher()
            if disp is not None:
                disp.evict_layer(self._pending[0], 0)
            self._pending = None

    def on_draft_start(self) -> None:
        """Called at the verify→draft transition (in_draft_phase set True).
        P4: drain the deferred merge/evict + sync the merge stream (so the draft
        reads valid merged). P1: flush the archer expert cache — the merged-dense
        draft never dispatches, so the cache is idle here; freeing it (host
        copies remain — just drops GPU mirrors) makes that budget available to
        the merged, phase-exclusive (§1.4). flush no-op unless `flush`."""
        self._drain_pending()
        if not self.flush:
            return
        disp = self._dispatcher()
        if disp is not None:
            disp.flush_cache(0)

    def on_draft_end(self) -> None:
        """Called at the draft→verify transition. P1: flush the merged experts —
        they are dead during verify (verify uses real routing), so freeing them
        hands the budget back to the verify expert cache. They are rebuilt next
        cycle (refresh / on_verify_layer). No-op unless `flush`."""
        if not self.flush or self.controller is None:
            return
        self.controller.draft_cache.clear()
        import torch
        torch.cuda.empty_cache()
