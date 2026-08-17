"""moe_precache baseline — static, prefill-informed expert pinning.

Non-speculative. During the prefill target forward we tally, per MoE layer,
how many tokens route to each expert (top-k of the router logits). After
prefill we PIN the top ``pin_fraction`` experts per layer resident in the
archer engine (via the same ``set_pinned`` the merged-slot cache uses for its
resident singletons). Every other expert is fetched on-demand each time it is
routed and is NOT cached — the archer pool is sized in ``cli.py`` to hold only
the pinned set plus one MoE layer's decode working set, so nothing beyond the
pins ever stays resident across layers/steps.

Contrast with ``moe_caching`` (the renamed ``none`` baseline): that keeps the
FULL vram budget as a dynamic archer cache with no pinning. ``moe_precache``
replaces the runtime cache with a static, prefill-count-chosen pin and no other
caching — a "what if you just cached the prefill-hot experts and nothing else"
baseline.

Both run paths are supported without touching the speculative machinery:

* B=1 (``specbench.py``, HF ``generate``): ``auto_pin=True``. The per-layer
  gate forward hooks tally routing on the prefill forward(s) (>1 token) and
  pin themselves at the first decode step (1 token). ``reset()`` is wired to
  ``on_question_start`` so pins/counts are per-question.
* B>1 (``batch_spec.py``): ``auto_pin=False``. The gate hooks tally across all
  B per-sequence prefills (pooled); the batch loop calls ``reset()`` at the
  start of each batch and ``pin()`` explicitly after the prefill loop.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional

import torch


class PrecacheManager:
    """Counts prefill routing via gate forward hooks and pins the hot experts.

    The dispatcher is shared across all MoE blocks (``model_offload`` assigns
    the same ``expert_executor`` to every block), so ``set_pinned`` /
    ``clear_pinned`` on any block's dispatcher govern the whole model.
    """

    def __init__(self, model, adapter, *, pin_fraction: float = 0.10,
                 count_top_k: Optional[int] = None, auto_pin: bool = True):
        if not 0.0 < pin_fraction <= 1.0:
            raise ValueError(
                f"pin_fraction must be in (0, 1], got {pin_fraction!r}")
        self.model = model
        self.adapter = adapter
        self.pin_fraction = pin_fraction
        self.auto_pin = auto_pin
        self.count_top_k = (count_top_k if count_top_k is not None
                            else adapter.default_count_top_k(model))
        # (dispatch layer_id, block, num_experts) per MoE layer.
        self._layers: List = []
        self._handles: List = []                     # forward-hook handles
        self._disp = None                            # shared archer dispatcher
        self._counts: Dict[int, torch.Tensor] = {}   # layer_id -> [E] float
        self._counting = False
        self._pinned = False

    # ── setup / teardown ────────────────────────────────────────────────
    def arm(self) -> int:
        """Register a forward hook on every MoE block's ``gate``. Returns the
        number of MoE layers armed. Idempotent guard via ``_handles``."""
        if self._handles:
            return len(self._layers)
        for enum_i, block in self.adapter.iter_moe(self.model):
            # dispatch keys the pin set by ``block.layer_id`` (qwen.py forward),
            # so pin with the SAME index. Fall back to the enum index for
            # backends without an explicit layer_id.
            layer_id = int(getattr(block, "layer_id", enum_i))
            n_experts = self.adapter.num_experts(block)
            if self._disp is None and hasattr(block, "expert_executor"):
                self._disp = block.expert_executor.expert_dispatcher
            self._layers.append((layer_id, block, n_experts))
            self._handles.append(
                block.gate.register_forward_hook(
                    self._make_hook(layer_id, n_experts)))
        self._counting = True
        return len(self._layers)

    def disarm(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()
        if self._disp is not None:
            self._disp.clear_pinned(0)

    # ── per-question / per-batch lifecycle ──────────────────────────────
    def reset(self, *_args, **_kwargs) -> None:
        """Drop all pins + counts and re-open counting for the next prefill.
        Ignores any args so it can be wired straight to ``on_question_start(q)``."""
        self._counts.clear()
        if self._disp is not None:
            self._disp.clear_pinned(0)
        self._counting = True
        self._pinned = False

    def pin(self) -> None:
        """Pin the top-``pin_fraction`` experts per layer from the pooled
        prefill counts, then stop counting. Idempotent per question/batch."""
        if self._pinned:
            return
        self._counting = False
        self._pinned = True
        if self._disp is None:
            return
        for layer_id, _block, n_experts in self._layers:
            counts = self._counts.get(layer_id)
            if counts is None:
                # Layer never observed during prefill — pin nothing rather than
                # pinning arbitrary (zero-count) experts.
                continue
            n_pin = max(1, math.ceil(self.pin_fraction * n_experts))
            n_hot = int((counts > 0).sum().item())
            k = min(n_pin, n_experts) if n_hot == 0 else min(n_pin, n_hot)
            ids = [int(x) for x in torch.topk(counts, k).indices.tolist()]
            self._disp.set_pinned(layer_id, ids, 0)

    # ── the gate hook ───────────────────────────────────────────────────
    def _make_hook(self, layer_id: int, n_experts: int):
        def _hook(_module, _inp, out):
            if not self._counting:
                return
            logits = out[0] if isinstance(out, (tuple, list)) else out
            if not torch.is_tensor(logits) or logits.dim() != 2:
                return
            # auto-pin (B=1): the first single-token forward is the first decode
            # step — pin from the prefill counts gathered so far, then stop.
            if self.auto_pin and logits.shape[0] == 1:
                if self._counts:
                    self.pin()
                return
            k = min(self.count_top_k, logits.shape[-1])
            top = torch.topk(logits.detach(), k, dim=-1).indices   # [tok, k]
            c = torch.bincount(top.reshape(-1),
                               minlength=n_experts).to(torch.float32).cpu()
            prev = self._counts.get(layer_id)
            self._counts[layer_id] = c if prev is None else prev + c
        return _hook


class OndemandFlusher:
    """moe_ondemand: cache-disabled non-speculative baseline. Runs the same path
    and pool as moe_caching, but registers a post-forward hook on the model that
    calls ``expert_dispatcher.flush_cache`` after EVERY forward pass — i.e. after
    prefill and after each decode step. Each step therefore starts with an empty
    expert cache and re-fetches every routed expert on demand: same VRAM as
    moe_caching, but the cache is forcibly cleared, isolating the pure TPS cost
    of "no cache" (moe_ondemand_plan.md §10 Path A).

    A single model-level hook covers both run paths (B=1 specbench.generate and
    B>1 batch_spec both invoke ``model.forward`` per step), so batch_spec needs
    no change.
    """

    def __init__(self, model):
        self.model = model
        self._disp = None
        self._handle = None
        self.n_flushes = 0

    def arm(self) -> bool:
        """Find the shared archer dispatcher from the model and register the
        post-forward flush hook. Returns True if a dispatcher was found."""
        for m in self.model.modules():
            ex = getattr(m, "expert_executor", None)
            d = getattr(ex, "expert_dispatcher", None) if ex else None
            if d is not None:
                self._disp = d
                break
        if self._disp is None or not hasattr(self._disp, "flush_cache"):
            return False
        self._handle = self.model.register_forward_hook(self._hook)
        return True

    def _hook(self, _module, _inp, _out):
        if self._disp is not None:
            self._disp.flush_cache(0)
            self.n_flushes += 1

    def disarm(self) -> None:
        if self._handle is not None:
            self._handle.remove()
            self._handle = None
