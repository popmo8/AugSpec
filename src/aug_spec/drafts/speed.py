"""SPEED early-exit draft (Table 1 "Speed" baseline, hooper2025speed).

SPEED (Hooper et al., NeurIPS-W 2023) predicts future tokens from
EARLY-LAYER hidden states and pipelines their forward passes. The
pipelining / parameter sharing is a systems optimisation that does not
change what token the early layers predict, so as a *draft construction*
SPEED reduces to an early-exit draft: the draft forward runs only the
first `num_layers` decoder layers, skips the rest (identity), and lets
the model's own final norm + lm_head classify the early hidden state.
Training-free adaptation — the original fine-tunes the classifier with a
weighted early-exit loss, which we cannot do to the (frozen) target.

Budget: `num_layers: 6` of Qwen3's 48 → the draft touches 6/48 = 12.5%
of expert memory. This is the DEPTH-axis instantiation of the shared
12.5% draft budget (SpecMoE prunes the width axis, 16/128 per layer).

Mechanism:
  * `cache_kind = "masked"` with no mask ever stored → the first
    `num_layers` MoE blocks run standard full routing in BOTH phases
    (the masked forward falls through when the cache is empty); zero
    adapter changes.
  * `post_install` wraps `model.model.layers[i]` for i >= num_layers so
    the whole decoder layer becomes identity while
    `controller.in_draft_phase`. Skipped layers append no KV during the
    draft, which is safe: HF's cache-length bookkeeping reads layer 0
    (always run) and the assistant-cache crop is a no-op on shorter
    layers. The C-BOOT KV-copy still seeds the early layers with the
    target's true prompt KV.
"""

from __future__ import annotations

from .base import DraftStrategy


class SpeedDraft(DraftStrategy):
    """Early-exit draft: run the first `num_layers` decoder layers only.

    Args:
        num_layers: how many leading decoder layers the draft executes.
            Must be >= 1 and < the model's total decoder-layer count
            (validated against the model in `post_install`).
    """

    cache_kind = "masked"

    def __init__(self, num_layers: int):
        if num_layers < 1:
            raise ValueError(f"num_layers must be >= 1, got {num_layers!r}")
        self.num_layers = num_layers
        self._wrapped = []      # [(layer module, original forward)]

    def post_install(self, controller) -> None:
        if self._wrapped:   # re-install without uninstall: restore first
            self.post_uninstall(controller)
        layers = getattr(getattr(controller.model, "model", None),
                         "layers", None)
        if layers is None:
            raise TypeError(
                "speed draft expects a model with `.model.layers` "
                "(Qwen3-style decoder); got "
                f"{type(controller.model).__name__}")
        if self.num_layers >= len(layers):
            raise ValueError(
                f"num_layers ({self.num_layers}) must be < the model's "
                f"decoder-layer count ({len(layers)})")
        for layer in list(layers)[self.num_layers:]:
            orig = layer.forward
            # The adapter supplies the layer's return convention (plain
            # tensor for HF-native families; deepseek's 4.36-style tuple
            # with cache pass-through).
            layer.forward = (
                lambda hidden_states, *a, _orig=orig, **kw:
                controller.adapter.decoder_skip_output(hidden_states, *a, **kw)
                if controller.in_draft_phase
                else _orig(hidden_states, *a, **kw))
            self._wrapped.append((layer, orig))

    def post_uninstall(self, controller) -> None:
        for layer, orig in self._wrapped:
            layer.forward = orig
        self._wrapped.clear()
