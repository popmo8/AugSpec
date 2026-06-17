"""Speculative-decoding phase patching + specbench glue.

When target and draft share weights (shared-model spec decoding), the
controller needs to know which phase it is in so a single block.forward
can branch:

  * `shared_model_phase_patch(controller)` — context manager that
    monkey-patches `AssistedCandidateGenerator.get_candidates` so the
    controller's `in_draft_phase` flag flips to True around the draft
    forward and back to False otherwise.

  * `specbench_callbacks(controller, on_cycle_extra=...)` — returns the
    `on_question_start` / `on_cycle` kwargs to splat into
    `run_specbench(...)`. Resets the controller per question and
    refreshes its draft cache after every verify cycle.

Both helpers depend only on the controller exposing:

    .in_draft_phase  (bool, writable)
    .reset()
    .update_masks()
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Callable, Dict, Optional


@contextmanager
def shared_model_phase_patch(controller: Any):
    """Flip `controller.in_draft_phase` around `AssistedCandidateGenerator.
    get_candidates`. Composes with `_locked_assist_patch` in specbench so
    you can enter this BEFORE `run_specbench(...)` and the T-locking
    patch wraps our phase-toggled version. Net call order:

        locked_get → phase_get → orig_get
        (T lock)    (set True)   (real draft)
                    (set False)
    """
    from transformers.generation.candidate_generator import (
        AssistedCandidateGenerator,
    )

    orig_get = AssistedCandidateGenerator.get_candidates

    def patched_get(self, input_ids):
        controller.in_draft_phase = True
        try:
            return orig_get(self, input_ids)
        finally:
            controller.in_draft_phase = False
            # draft→verify transition: offload-merge engine may flush the
            # merged experts here to reclaim workspace for verify (§1.4).
            # No-op unless merge_offload built an engine.
            engine = getattr(controller, "merge_engine", None)
            if engine is not None:
                engine.on_draft_end()

    AssistedCandidateGenerator.get_candidates = patched_get
    try:
        yield
    finally:
        AssistedCandidateGenerator.get_candidates = orig_get


def specbench_callbacks(
    controller: Any,
    on_cycle_extra: Optional[Callable[..., None]] = None,
) -> Dict[str, Any]:
    """Build the kwargs to splat into `run_specbench(...)`.

    `on_cycle_extra(qres, cs)` (optional) runs *after* the controller's
    refresh — useful for tagging the latest history snapshot, capturing
    `controller.cycle_misses[-1]`, or other per-cycle telemetry.
    """
    def on_cycle(qres, cs):
        controller.update_masks()
        if on_cycle_extra is not None:
            on_cycle_extra(qres, cs)

    return {
        "on_question_start": lambda _q: controller.reset(),
        "on_cycle": on_cycle,
    }
