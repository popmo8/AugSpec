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
                 flush: bool = False, overlap: bool = False,
                 cache_mode: bool = False, slots_per_layer: int = 0,
                 expert_bytes: int = 0, singleton_pin_budget=None):
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
        # P4: run the per-layer merge on a side CUDA stream so it overlaps with
        # the next layer's PCIe fetch (instead of a per-layer full device sync).
        # Ablation flag merge_overlap.
        self.overlap = overlap
        # C1 (merged_cache_plan.md): the GPU merged-expert cache. Policy lives
        # in MergedCacheIndex (Python); the C0 slot table is the mechanism.
        # None outside cache mode → every hook below no-ops and the legacy
        # per-cycle rebuild is used.
        self.cache_mode = cache_mode
        self._slots_per_layer = slots_per_layer
        self._expert_bytes = expert_bytes
        self._singleton_pin_budget = singleton_pin_budget
        self.merged_cache = None
        self._merge_stream = None                 # lazily created side stream
        # Back-reference to the Controller (set by it after construction): gives
        # on_verify_layer access to the draft (per-layer merge logic) + draft_cache.
        self.controller = None
        # layer_idx → blocks, filled by attach(); used by the lifecycle hooks.
        self.blocks: List[Tuple[int, nn.Module]] = []
        # Whether the C++ per-expert output capture is currently ON. For
        # act_sim_prefill_only methods (hybrid) it is armed per question
        # (on_question_start) and disarmed at the first draft start — the
        # prefill→draft boundary — so decode pays no capture cost.
        self._capture_on = False

    # ── setup ───────────────────────────────────────────────────────────
    def attach(self, blocks: List[Tuple[int, nn.Module]]) -> None:
        """Tag each MoE block with a back-reference to this engine so the
        shared `_build_one` / verify hooks can reach it off the block (same
        pattern as `_cpu_merge_source`)."""
        self.blocks = list(blocks)
        for _, block in self.blocks:
            block._merge_engine = self
        # C1: size the C++ slot table + build the policy index. Behaviour is
        # unchanged until the draft's refresh routes through merged_cache.
        if self.cache_mode and self._slots_per_layer > 0:
            disp = self._dispatcher()
            if disp is not None and hasattr(disp, "init_merged_slots"):
                num_layers = max(li for li, _ in self.blocks) + 1
                disp.init_merged_slots(num_layers, self._slots_per_layer)
                from aug_spec.runtime.merged_cache import MergedCacheIndex
                self.merged_cache = MergedCacheIndex(
                    self._slots_per_layer, self._expert_bytes,
                    singleton_pin_budget=self._singleton_pin_budget)
                print(f"  [merged_cache] enabled: layers={num_layers} "
                      f"S={self._slots_per_layer} "
                      f"expert_bytes={self._expert_bytes}")
            else:
                print("  [merged_cache] NOT enabled "
                      f"(cache_mode={self.cache_mode}, disp={disp is not None})")
        # activation_similarity / hybrid: turn on the C++ engine's per-expert
        # output capture up front (must be on BEFORE any dispatch;
        # on_verify_layer runs post-dispatch). The dispatcher is shared across
        # blocks. Idempotent; no-op for other cluster methods.
        self._set_capture(True)

    def _cluster_method(self):
        """The draft's cluster method (or None) — the object whose class flags
        (`needs_activation_sim`, `act_sim_prefill_only`) gate the capture."""
        draft = getattr(self.controller, "draft", None)
        return getattr(draft, "cluster_method", None)

    def _set_capture(self, on: bool) -> None:
        """Toggle the C++ per-expert output capture (no-op unless the cluster
        method needs it / the dispatcher supports it). Turning OFF also drains
        the capture buffer so stale prefill outputs never leak into later
        accumulation."""
        if not getattr(self._cluster_method(), "needs_activation_sim", False):
            return
        disp = self._dispatcher()
        if disp is None or not hasattr(disp, "set_capture_expert_out"):
            return
        disp.set_capture_expert_out(on)
        if not on and hasattr(disp, "get_captured_expert_outputs"):
            disp.get_captured_expert_outputs()      # swap-clears the buffer
        self._capture_on = on

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
        # activation_similarity: this layer's experts just ran (post-dispatch),
        # so the C++ capture buffer holds their raw outputs — accumulate the
        # pairwise output-cosine BEFORE the merge below reads ctx.pair_sim.
        if getattr(getattr(draft, "cluster_method", None),
                   "needs_activation_sim", False) and \
                hasattr(draft, "accumulate_activation_sim"):
            d = self._dispatcher()
            if d is not None and hasattr(d, "get_captured_expert_outputs"):
                draft.accumulate_activation_sim(
                    layer_idx, d, self.adapter.num_experts(block))
        score = getattr(draft, "target_score", {}).get(layer_idx)
        if score is None or not hasattr(draft, "_refresh_layer"):
            return
        disp = self._dispatcher()
        import torch

        def _merge():
            draft._refresh_layer(self.adapter, block, layer_idx, score,
                                 self.controller.draft_cache)

        if not self.overlap or disp is None:
            # P2 (synchronous): merge on the default stream. C-DEL
            # (merged_cache_plan.md §2.5): NO per-layer eviction any more —
            # verify residents stay in the pool and are reclaimed on demand by
            # the pin-aware LFU (FindExpertEvict); the old rationale (avoid the
            # overload evict-after-use race) died with the overload path. The
            # synchronize stays for now so this change is eviction-only
            # (attribution); C1 removes it with the cache mode.
            _merge()
            if disp is not None:
                torch.cuda.synchronize()
            return

        # P4: run the merge on a side stream so it overlaps the next layer's PCIe
        # fetch (default stream). The merge result in draft_cache is read only
        # in the next draft phase, after on_draft_start syncs the side stream.
        # (The old deferred per-layer evict went with C-DEL — eviction is
        # demand-driven now.)
        if self._merge_stream is None:
            self._merge_stream = torch.cuda.Stream()
        with torch.cuda.stream(self._merge_stream):
            _merge()

    def _dispatcher(self):
        """The archer ExpertDispatcher (shared across blocks), or None."""
        if not self.blocks:
            return None
        ex = getattr(self.blocks[0][1], "expert_executor", None)
        return getattr(ex, "expert_dispatcher", None) if ex is not None else None

    def _drain_pending(self) -> None:
        """P4: sync the side merge stream at the verify→draft boundary so the
        draft reads valid merged tensors off the default stream. No-op unless
        overlap ran. (The old deferred per-layer evict went with C-DEL.)"""
        if self._merge_stream is not None:
            self._merge_stream.synchronize()

    def on_question_start(self) -> None:
        """Called from Controller.reset() at each question start. Re-arms the
        expert-output capture for act_sim_prefill_only methods (hybrid), so the
        upcoming prefill forward is captured again after the previous question
        disarmed it. No-op for always-on methods (already capturing) and for
        methods without activation-sim."""
        if getattr(self._cluster_method(), "act_sim_prefill_only", False) \
                and not self._capture_on:
            self._set_capture(True)
        # C1: question boundary — drop the content index and every pin (no
        # cross-question warm start yet; that is C2).
        if self.merged_cache is not None:
            self.merged_cache.reset(self.blocks, self._dispatcher())

    def on_draft_start(self) -> None:
        """Called at the verify→draft transition (in_draft_phase set True).
        Prefill-only act-sim (hybrid): the FIRST draft start of a question is
        the prefill→draft boundary — disarm the expert-output capture here so
        every decode cycle runs capture-free (the latency point of hybrid);
        on_question_start re-arms it. P4: drain the deferred merge/evict + sync
        the merge stream (so the draft reads valid merged). P1: flush the archer
        expert cache — the merged-dense draft never dispatches, so the cache is
        idle here; freeing it (host copies remain — just drops GPU mirrors)
        makes that budget available to the merged, phase-exclusive (§1.4).
        flush no-op unless `flush`."""
        if self._capture_on and getattr(
                self._cluster_method(), "act_sim_prefill_only", False):
            self._set_capture(False)
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
