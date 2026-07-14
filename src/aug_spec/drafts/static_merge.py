"""Static merged-expert draft (Table 1 "HC-SMoE" baseline, merge–static).

Loads an offline-computed grouping spec (`scripts/build_hc_smoe.py` → one
json per model) and, once per run, merges every layer's experts into K
frequency-weighted dense experts — HC-SMoE's clustering + merging, frozen
for the whole run. Routing goes through the shared gate-remap path
(`_route_multi_expert`, `indices` covering all n experts), the same kernel
the dynamic merge drafts use — only the (offline) partition, the frequency
weights, and the freeze differ.

Spec schema (json):
    {"model_id": ..., "K": 16, "count_top_k": 8,
     "layers": {"<layer_idx>": {"groups": [[expert ids]...],
                                "freq":   [n calibration top-k counts]}}}
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

from aug_spec.merging import linear_merge

from .base import DraftStrategy


class StaticMergeDraft(DraftStrategy):
    """K offline-clustered, frequency-weighted merged experts per layer.

    Args:
        spec_path: json produced by `scripts/build_hc_smoe.py`.
        draft_top_k: clusters activated per token in the draft forward.
            None → the model's native top-k (resolved in the adapter).
    """

    cache_kind = "averaged"
    holds_merged_residency = True

    def __init__(self, spec_path: str, draft_top_k: Optional[int] = None):
        if draft_top_k is not None and draft_top_k < 1:
            raise ValueError(
                f"draft_top_k must be >= 1, got {draft_top_k!r}")
        self.spec_path = Path(spec_path)
        self.draft_top_k = draft_top_k
        # layer_idx → frozen "multi" cache dict, built once in prepare().
        self._built: Dict[int, Dict[str, Any]] = {}

    def prepare(self, adapter, blocks) -> None:
        if self._built:
            return
        spec = json.loads(self.spec_path.read_text())
        layers = spec["layers"]
        for li, block in blocks:
            entry = layers.get(str(li))
            if entry is None:
                raise ValueError(
                    f"static_merge: spec {self.spec_path} has no entry for "
                    f"MoE layer {li} — was it built for this model?")
            n = adapter.num_experts(block)
            groups = [sorted(int(i) for i in g) for g in entry["groups"]]
            freq = [float(f) for f in entry["freq"]]
            flat = sorted(i for g in groups for i in g)
            if len(freq) != n or flat != list(range(n)):
                raise ValueError(
                    f"static_merge: layer {li} spec does not partition "
                    f"{n} experts (freq len {len(freq)}, coverage "
                    f"{len(flat)})")

            experts = []
            masses = []
            total_freq = sum(freq)
            for group in groups:
                gf = sum(freq[i] for i in group)
                weights = [0.0] * n
                if gf > 0:
                    # HC-SMoE merging: frequency-weighted within the group.
                    for i in group:
                        weights[i] = freq[i] / gf
                else:
                    for i in group:
                        weights[i] = 1.0 / len(group)
                experts.append(linear_merge(adapter, block, group, weights))
                # Cluster mass = calibration frequency share; only used for
                # the descending order the multi-cache contract expects
                # (routing is per-token gate remap). Size share when the
                # calibration never fired anything (degenerate).
                masses.append(gf / total_freq if total_freq > 0
                              else len(group) / n)
            order = sorted(range(len(groups)), key=lambda j: -masses[j])
            self._built[li] = {
                "kind":    "multi",
                "experts": [experts[j] for j in order],
                "weights": [masses[j] for j in order],
                "indices": [groups[j] for j in order],
            }

    def prepopulate(self, adapter, blocks, draft_cache):
        draft_cache.clear()
        draft_cache.update(self._built)

    def lazy_build(self, layer_idx, block, adapter):
        # The compile-warmup generate runs before the first
        # controller.reset()/prepopulate — serve the frozen cache (same
        # fix as random_merge; job 259444).
        return self._built.get(layer_idx)

    # refresh / capture: inherited no-ops — everything is frozen.
