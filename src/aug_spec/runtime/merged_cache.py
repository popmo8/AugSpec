"""GPU merged-expert cache — the Python POLICY side (merged_cache_plan.md C1).

Owns what lives in the C++ merged-slot table: content addressing
(`(layer, frozenset(members)) → slot`), the two-stage cache-first partition
(adopt cached groups whose members all stay active, greedy over the rest),
slot allocation, and the pin choice (current working set = merged slots +
resident singletons). The dispatcher owns only mechanism (store / merge-into /
pin / discard — C0 APIs).

Active only in "cache mode": offload + merge engine + `within_weight: uniform`
(member-set keys are only valid under uniform coefficients, Eq. weightmerge)
+ `AUG_LEGACY_MERGE` unset. Everything else keeps the legacy per-cycle
rebuild (`ScoreBasedAvgDraft._cluster_and_build`).

C1 scope: within-question reuse only — `reset()` clears the index at每題
boundary (cross-question warm start is C2).
"""

from __future__ import annotations

from collections import OrderedDict, defaultdict
from typing import Any, Dict, List

from aug_spec.clustering import ClusterContext


class MergedCacheIndex:
    """Per-run content index + slot/pin policy over the C0 slot table."""

    def __init__(self, slots_per_layer: int, expert_bytes: int,
                 singleton_pin_budget=None):
        self.S = int(slots_per_layer)
        self.expert_bytes = int(expert_bytes)
        # Resident-singleton archer pins are capped so the verify floor stays
        # unpinnable (cli: pool − floor). None = uncapped (unit tests).
        self.singleton_pin_budget = singleton_pin_budget
        self.pinned_bytes = 0            # current total singleton-pin bytes
        self._layer_pin_bytes: Dict[int, int] = {}
        # layer → OrderedDict[frozenset(member ids) → slot_id]; insertion order
        # doubles as recency (adopt refreshes with move_to_end).
        self.index: Dict[int, "OrderedDict[frozenset, int]"] = defaultdict(
            OrderedDict)
        # telemetry (merged_cache_plan.md §4.3)
        self.hit_n = 0                  # adopted groups served from a slot
        self.miss_n = 0                 # groups merged into a slot this run
        self.elided_bytes = 0           # member reads elided by hits
        self.groups_n = 0               # total groups partitioned
        self.singleton_pinned_n = 0     # singletons pinned for verify hits
        self.singleton_slot_n = 0       # singletons that needed an identity slot
        self.singleton_budget_denied_n = 0   # pin budget full → identity slot
        self.fallback_n = 0             # legacy CPU/GPU merges (slot API failed)

    # ── question boundary ────────────────────────────────────────────────
    def reset(self, blocks, dispatcher) -> None:
        """C1: no cross-question warm start — drop the index and every pin so
        the next question starts clean (slots become discardable history)."""
        self.index.clear()
        self.pinned_bytes = 0
        self._layer_pin_bytes.clear()
        if dispatcher is None:
            return
        for li, _ in blocks:
            dispatcher.set_merged_slot_pinned(li, [], 0)
        if hasattr(dispatcher, "clear_pinned"):
            dispatcher.clear_pinned(0)

    # ── slot allocation ──────────────────────────────────────────────────
    def _alloc_slot(self, li: int) -> int:
        used = set(self.index[li].values())
        for s in range(self.S):
            if s not in used:
                return s
        # Every slot is owned by an index entry → steal the oldest (its
        # content is overwritten by the coming merge; entry dropped).
        oldest_key = next(iter(self.index[li]))
        return self.index[li].pop(oldest_key)

    # ── the per-layer build (replaces _cluster_and_build in cache mode) ──
    def build_layer(self, li: int, block, weights: List[float],
                    draft, adapter) -> Dict[str, Any]:
        disp = block.expert_executor.expert_dispatcher
        active = [i for i, w in enumerate(weights) if w > 0.0]
        active_set = set(active)
        K = draft.K
        # Singleton-pin ledger: this layer's previous contribution is being
        # replaced by this build.
        pinned_other = self.pinned_bytes - self._layer_pin_bytes.get(li, 0)

        experts: List[Dict[str, Any]] = []
        masses: List[float] = []
        indices: List[List[int]] = []
        slot_ids: List[int] = []        # this cycle's working-set slots
        singleton_ids: List[int] = []   # this cycle's resident singleton pins

        def _pin_now():
            # Progressive pinning (2026-07-10 fix): pin at ACQUISITION, not at
            # the end of the layer — a later group's merge_to_slot room loop
            # (or TryDiscardUnpinnedSlot) would otherwise victimise this
            # layer's not-yet-pinned singleton/slot; archer's SetDevice(host)
            # flips the held tensor refs to CPU in place (set_data), which
            # surfaced as a mixed-device stack in the draft (job 258555).
            disp.set_merged_slot_pinned(li, slot_ids, 0)
            disp.set_pinned(li, singleton_ids, 0)

        ref_dev = [None]   # first device seen this layer

        def _emit(group, expert_dict, mass, src=""):
            # Fail fast with full context on a MIXED-device expert list — it
            # would otherwise surface as an opaque torch.stack error inside
            # the draft forward. (Same-device-throughout is the invariant;
            # CPU-only unit tests stay valid.)
            for k, t in expert_dict.items():
                if ref_dev[0] is None:
                    ref_dev[0] = t.device
                elif t.device != ref_dev[0]:
                    raise RuntimeError(
                        f"merged_cache: mixed devices — {k} on {t.device} vs "
                        f"{ref_dev[0]} — layer {li} group {sorted(group)} "
                        f"src={src}")
            experts.append(expert_dict)
            masses.append(mass)
            indices.append(sorted(group))

        def _dict3(tensors):
            return {"gate_proj": tensors[0], "up_proj": tensors[1],
                    "down_proj": tensors[2]}

        # ── stage 1: adopt(§2.3 cache-first)— cached groups whose members
        # all remain active are taken as-is: 0 fetch, 0 merge.
        used: set = set()
        adopted_groups: List[frozenset] = []
        for members in list(reversed(self.index[li])):   # most recent first
            if len(adopted_groups) >= K:
                break
            if not members.issubset(active_set) or (members & used):
                continue
            slot = self.index[li][members]
            tensors = disp.get_merged_slot(li, slot, 0)
            if len(tensors) != 3:              # discarded under fetch pressure
                del self.index[li][members]
                continue
            self.index[li].move_to_end(members)
            used |= members
            adopted_groups.append(members)
            slot_ids.append(slot)
            _pin_now()
            self.hit_n += 1
            self.elided_bytes += len(members) * self.expert_bytes
            _emit(members, _dict3(tensors),
                  sum(weights[m] for m in members), src="adopt")

        # ── stage 2: greedy over the remainder ──
        remaining = [i for i in active if i not in used]
        k_rem = max(0, K - len(adopted_groups))
        groups: List[List[int]] = []
        if remaining:
            if k_rem == 0:
                groups = [[i] for i in remaining]    # degenerate: keep them
            else:
                ctx = ClusterContext(
                    active=remaining, weights=weights, layer_idx=li,
                    cooccur=draft.cooccur.get(li),
                    pair_sim=draft._pair_sim_table(li))
                groups = draft.cluster_method.assign(ctx, k_rem)

        for g in groups:
            mass = sum(weights[m] for m in g)
            if len(g) == 1:
                # Singleton: the DRAFT is always served from an identity slot
                # (our own torch buffers) — holding refs to the resident
                # original is unsafe: archer\'s prefetcher relocates nodes
                # regardless of pinned_ and SetDevice(host) flips held refs to
                # CPU in place (jobs 258555/258658). The original is STILL
                # pinned (budget permitting), purely for the dual-identity
                # verify hit — losing that pin costs a re-fetch, never
                # correctness.
                i = g[0]
                pin_ok = (self.singleton_pin_budget is None
                          or pinned_other
                          + (len(singleton_ids) + 1) * self.expert_bytes
                          <= self.singleton_pin_budget)
                if pin_ok:
                    self.singleton_pinned_n += 1
                    singleton_ids.append(i)
                    _pin_now()
                else:
                    self.singleton_budget_denied_n += 1
                key = frozenset(g)
                slot = self._alloc_slot(li)
                if disp.merge_experts_to_slot(li, slot, [i], [1.0], 0):
                    tensors = disp.get_merged_slot(li, slot, 0)
                    self.index[li][key] = slot
                    slot_ids.append(slot)
                    _pin_now()
                    self.singleton_slot_n += 1
                    self.miss_n += 1
                    _emit(g, _dict3(tensors), mass, src="singleton-slot")
                else:
                    self.fallback_n += 1
                    _emit(g, self._legacy_one(draft, adapter, block,
                                              weights, g), mass,
                          src="legacy-fallback-single")
                continue
            # Pair / group: uniform coefficients (cache-mode precondition) →
            # the merged content is determined by the member set alone.
            key = frozenset(g)
            slot = self._alloc_slot(li)
            uw = [1.0 / len(g)] * len(g)
            if disp.merge_experts_to_slot(li, slot, list(g), uw, 0):
                tensors = disp.get_merged_slot(li, slot, 0)
                self.index[li][key] = slot
                slot_ids.append(slot)
                _pin_now()
                self.miss_n += 1
                _emit(g, _dict3(tensors), mass, src="pair-slot")
            else:
                self.fallback_n += 1
                _emit(g, self._legacy_one(draft, adapter, block, weights, g),
                      mass, src="legacy-fallback-pair")

        self.groups_n += len(experts)

        # ── final working-set pins(冪等;跌出上輪集合者在首次 _pin_now
        # 時即被替換掉 → 自動 unpin)──
        _pin_now()
        self._layer_pin_bytes[li] = len(singleton_ids) * self.expert_bytes
        self.pinned_bytes = pinned_other + self._layer_pin_bytes[li]

        total = sum(masses) or 1.0
        order = sorted(range(len(experts)), key=lambda k: -masses[k])
        return {
            "kind":    "multi",
            "experts": [experts[k] for k in order],
            "weights": [masses[k] / total for k in order],
            "indices": [indices[k] for k in order],
        }

    @staticmethod
    def _legacy_one(draft, adapter, block, weights, group):
        """Slot API failed (e.g. no evictable room) — fall back to the legacy
        per-cycle merge for THIS group only (uniform within-group weights,
        matching the cache-mode coefficients)."""
        cw = [0.0] * len(weights)
        for i in group:
            cw[i] = 1.0 / len(group)
        return draft._build_one(adapter, block, cw)
