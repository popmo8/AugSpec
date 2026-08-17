"""Draft&Verify self-speculative layer-skip draft (Table 1 "Draft&Verify"
baseline, zhang2024draftverify).

Draft&Verify (Zhang et al., ACL 2024) drafts with the target model itself,
skipping an offline-OPTIMISED set of intermediate attention / MLP sublayers
(their Bayesian-optimisation search, §3.3); verification runs the full
model. Adaptation to the shared Table 1 protocol:

  * Budget: only MoE (MLP) sublayers hold expert memory, so the 12.5%
    draft-side budget binds as "the draft executes `len(mlp_keep)` of the
    model's MoE sublayers" (Qwen3: 6/48 — same accounting as `speed`, the
    depth-prefix instantiation). Attention sublayers are expert-memory-free
    and are retained by default, which only favours this baseline.
  * Selection: `scripts/search_draft_verify.py` runs the paper's Bayesian
    optimisation over which MoE sublayers to KEEP, maximising greedy draft
    /target token agreement on held-out C4 continuations (the acceptance
    analogue of the paper's per-token-latency objective — Table 1 measures
    acceptance only, and the fixed budget already pins the draft cost).
  * The paper's adaptive draft-exiting is superseded by the protocol's
    fixed draft length (T=5, all rows).

Mechanism (sublayer-granular sibling of speed's whole-layer skip):
  * `cache_kind = "masked"` with no mask ever stored → kept MoE sublayers
    run standard full routing in BOTH phases (the masked forward falls
    through when the cache is empty); zero adapter changes.
  * `post_install` wraps `layer.mlp.forward` of skipped-MoE layers and
    `layer.self_attn.forward` of skipped-attention layers so each returns
    a zero contribution while `controller.in_draft_phase` — with the
    pre-norm residual, sublayer output 0 == the sublayer is bypassed
    (exactly the paper's skip semantics). Wrapping happens at the END of
    `Controller.install()`, i.e. on top of the adapter-swapped MoE
    forward, and `post_uninstall` runs first on the way out, so
    install/uninstall compose cleanly.
  * Skipped attention appends no KV during the draft. Layer 0's attention
    must therefore always run (HF's cache-length bookkeeping reads
    layer 0) — validated at install. The C-BOOT KV-copy still seeds every
    layer with the target's true prompt KV.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional, Sequence

import torch

from .base import DraftStrategy


def _validated_indices(name: str, raw: Sequence) -> List[int]:
    ids = [int(i) for i in raw]
    if len(set(ids)) != len(ids):
        raise ValueError(f"draft_verify: duplicate indices in {name}: {ids}")
    return sorted(ids)


class DraftVerifyDraft(DraftStrategy):
    """Skip an offline-selected set of decoder sublayers in the draft phase.

    Args (mutually exclusive sources):
        spec_path: json produced by `scripts/search_draft_verify.py`
            (schema: {"mlp_keep": [...], "attn_skip": [...], ...}).
        mlp_keep:  MoE-layer indices whose sublayer the draft executes;
            every other MoE sublayer is skipped. Dense-MLP layers hold no
            experts and always run. (Keep-set framing matches the
            expert-memory budget: |mlp_keep| / #MoE layers.)
        attn_skip: decoder-layer indices whose attention sublayer the
            draft skips (default: none — attention carries no expert
            memory). Layer 0 must not appear (KV bookkeeping).
    """

    cache_kind = "masked"
    needs_layer_spec = True

    def __init__(self, spec_path: Optional[str] = None,
                 mlp_keep: Optional[Sequence[int]] = None,
                 attn_skip: Optional[Sequence[int]] = None):
        if (spec_path is None) == (mlp_keep is None):
            raise ValueError(
                "draft_verify: pass exactly one of spec_path / mlp_keep")
        if spec_path is not None:
            if attn_skip is not None:
                raise ValueError(
                    "draft_verify: attn_skip comes from the spec when "
                    "spec_path is used")
            spec = json.loads(Path(spec_path).read_text())
            mlp_keep = spec["mlp_keep"]
            attn_skip = spec.get("attn_skip", [])
        self.mlp_keep = _validated_indices("mlp_keep", mlp_keep)
        self.attn_skip = _validated_indices("attn_skip", attn_skip or [])
        if not self.mlp_keep:
            raise ValueError("draft_verify: mlp_keep must not be empty")
        if 0 in self.attn_skip:
            raise ValueError(
                "draft_verify: layer 0's attention must run in the draft "
                "phase (HF cache-length bookkeeping reads layer 0)")
        self._wrapped = []      # [(module, original forward)]

    def post_install(self, controller) -> None:
        if self._wrapped:   # re-install without uninstall: restore first
            self.post_uninstall(controller)
        layers = getattr(getattr(controller.model, "model", None),
                         "layers", None)
        if layers is None:
            raise TypeError(
                "draft_verify expects a model with `.model.layers` "
                "(Qwen3-style decoder); got "
                f"{type(controller.model).__name__}")
        L = len(layers)
        # The budget binds on MoE sublayers only: mlp_keep must name MoE
        # layers, and only MoE sublayers are ever skipped (a dense-MLP
        # layer holds no experts and always runs — matching the search
        # space of scripts/search_draft_verify.py).
        moe_ids = {li for li, _ in controller.blocks}
        bad = [i for i in self.mlp_keep if i not in moe_ids]
        if bad:
            raise ValueError(
                f"draft_verify: mlp_keep indices {bad} are not MoE layers "
                f"(MoE layers: {sorted(moe_ids)})")
        bad = [i for i in self.attn_skip if not 0 <= i < L]
        if bad:
            raise ValueError(
                f"draft_verify: attn_skip indices {bad} out of range for "
                f"a {L}-layer model")
        if len(self.mlp_keep) >= len(moe_ids):
            raise ValueError(
                f"draft_verify: mlp_keep covers all {len(moe_ids)} MoE "
                "layers — nothing is skipped")

        block_by_id = dict(controller.blocks)
        mlp_skip = sorted(moe_ids - set(self.mlp_keep))
        for i in mlp_skip:
            mlp = block_by_id[i]
            orig = mlp.forward
            # Zero contribution + pre-norm residual == sublayer bypassed.
            # The adapter supplies the decoder layer's return convention
            # ((zeros, None) for qwen3/mixtral/gptoss, bare zeros for
            # deepseek's `h = self.mlp(h)`).
            mlp.forward = (
                lambda hidden_states, *a, _orig=orig, **kw:
                controller.adapter.mlp_skip_output(hidden_states)
                if controller.in_draft_phase
                else _orig(hidden_states, *a, **kw))
            self._wrapped.append((mlp, orig))
        for i in self.attn_skip:
            attn = layers[i].self_attn
            orig = attn.forward

            def _skip_attn(*args, _orig=orig, **kw):
                if not controller.in_draft_phase:
                    return _orig(*args, **kw)
                hs = kw["hidden_states"] if "hidden_states" in kw else args[0]
                # (attn_output, attn_weights) — the decoder layer unpacks.
                return torch.zeros_like(hs), None

            attn.forward = _skip_attn
            self._wrapped.append((attn, orig))

    def post_uninstall(self, controller) -> None:
        for module, orig in self._wrapped:
            module.forward = orig
        self._wrapped.clear()
