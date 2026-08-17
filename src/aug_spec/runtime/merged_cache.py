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
boundary(跨題 warm start 已於 2026-07-10 定案不做)。
C2 (2026-07-10): retention-aware slot stealing — the ONLY reclaimer of slots
under the carve model is `_alloc_slot`, so the paper's retention rule lives
there: never steal this build's entries; steal unprotected (members no longer
all-active) before protected; singletons (cheap identity rebuild) before
pairs; low co-occur pairs first; LRU as the tiebreak.
C2.1 (2026-07-11): PAIR-adopt-first — stage 1 adopts multi-member groups
only. Adopting a cached singleton spends 1 of the K group quota while
covering 1 expert, which breaks the M<=2K "2 coverage per group" accounting
and is what forced >K groups (slot overflow). Singletons are now decided by
greedy and served afterwards by a content probe (same zero-cost hit, no
quota distortion) — group count is structurally <=K whenever |active|<=2K.
The probe path also runs the archer-pin step, so a re-used singleton's
original keeps its verify-hit pin (the old stage-1 singleton adopt skipped
it → retained experts were re-fetched every verify).
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
        self.singleton_hit_n = 0        # singletons re-served from a cached slot
        self.singleton_budget_denied_n = 0   # pin budget full → identity slot
        self.sgl_feas_denied_n = 0      # singleton adopt denied: quota
                                        # infeasible (B variant guard)
        self.steal_n = 0                # slot steals (history evictions)
        self.steal_protected_n = 0      # had to steal a protected entry
        # per-layer build counter — entries stamped with it; this build's
        # entries are never steal victims.
        self._build_no: Dict[int, int] = defaultdict(int)

    # ── question boundary ────────────────────────────────────────────────
    def reset(self, blocks, dispatcher) -> None:
        """C1: no cross-question warm start — drop the index and every pin so
        the next question starts clean (slots become discardable history)."""
        self.index.clear()
        self.pinned_bytes = 0
        self._layer_pin_bytes.clear()
        self._build_no.clear()
        if dispatcher is None:
            return
        for li, _ in blocks:
            dispatcher.set_merged_slot_pinned(li, [], 0)
        if hasattr(dispatcher, "clear_pinned"):
            dispatcher.clear_pinned(0)

    # ── slot allocation(C2:retention-aware steal)─────────────────────
    def _alloc_slot(self, li: int, active_set=frozenset(),
                    cooccur=None, cur_stamp: int = -1) -> int:
        used = {v[0] for v in self.index[li].values()}
        for s in range(self.S):
            if s not in used:
                return s
        # Every slot is owned by an index entry → steal by the retention rule
        # (this is the ONLY slot reclaimer under the carve model): never this
        # build's entries; unprotected before protected (members all-active =
        # likely re-adopted next cycle); singletons before pairs (identity
        # rebuild is a µs copy, a pair costs a re-merge); low co-occur pairs
        # first; LRU (oldest first, stable min) as the tiebreak.
        def badness(item):
            members, (_slot, stamp) = item
            co = 0.0
            if len(members) > 1 and cooccur is not None:
                a, b = sorted(members)[:2]
                co = float(cooccur[a][b])
            return (stamp == cur_stamp,            # this build → last resort
                    members <= active_set,         # protected → late
                    len(members) > 1,              # pairs → later than singles
                    co)                            # cold pairs first
        # O(1) common case: entries iterate oldest-first, so the first old
        # unprotected singleton is provably optimal — break instead of a full
        # O(S) min() scan (C1's steal was next(iter()); 778k steals/run make
        # the scan a wall-clock line item, job 258941).
        best = best_item = None
        for item in self.index[li].items():
            b = badness(item)
            if best is None or b < best:
                best, best_item = b, item
                if b == (False, False, False, 0.0):
                    break
        victim, (slot, stamp) = best_item
        if stamp == cur_stamp:
            # Every slot is owned by THIS build ⇒ the partition produced more
            # than S=K' groups. With adopt-first accounting that is provably
            # impossible while |active| <= cap*K (cap = ceil(M/K), classic
            # pairing cap=2), so reaching here means an unsupported
            # configuration (a draft without a top-M <= cap*K cutoff in cache
            # mode). Stealing would overwrite content already emitted
            # this cycle — silently wrong draft weights — so fail fast instead
            # (2026-07-11: fallback path removed on request; no degraded mode).
            raise RuntimeError(
                f"merged_cache: slot demand exceeded S={self.S} at layer {li} "
                f"— partition produced >K groups, |active|>cap*K? Use a "
                f"draft with top-M <= cap*K (e.g. topm_count) in cache mode.")
        self.steal_n += 1
        if victim <= active_set:
            self.steal_protected_n += 1
        del self.index[li][victim]
        return slot

    # ── the per-layer build (replaces _cluster_and_build in cache mode) ──
    def build_layer(self, li: int, block, weights: List[float],
                    draft, routed=None) -> Dict[str, Any]:
        """routed=None → P2(同步 merge,dispatch 後呼叫)。
        routed=list  → C3 pipeline(dispatch 前呼叫):miss 組收集成 MergeJob
        一次 submit,成員到齊即在 C++ merge 線程執行;emit 的 slot handle 因
        D0 預配而在內容寫入前即有效,draft 在 WaitMergesDone 之後才讀。"""
        disp = block.expert_executor.expert_dispatcher
        cap = getattr(draft, "group_cap", 2)   # ceil(M/K); 2 = classic pairing
        pipeline = routed is not None
        job_slots: List[int] = []
        job_members: List[List[int]] = []
        job_weights: List[List[float]] = []

        def _write_slot(slot, ids, uw):
            if pipeline:
                job_slots.append(slot)
                job_members.append(list(ids))
                job_weights.append(list(uw))
                return True
            return disp.merge_experts_to_slot(li, slot, list(ids),
                                              list(uw), 0)
        active = [i for i, w in enumerate(weights) if w > 0.0]
        active_set = set(active)
        K = draft.K
        # Singleton-pin ledger: this layer's previous contribution is being
        # replaced by this build.
        pinned_other = self.pinned_bytes - self._layer_pin_bytes.get(li, 0)
        # C2: stamp this build — its entries are never steal victims.
        self._build_no[li] += 1
        cur = self._build_no[li]
        li_cooccur = draft.cooccur.get(li)

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

        def _pin_singleton(i):
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

        # ── stage 1: adopt(§2.3 cache-first)— cached groups whose members
        # all remain active are taken as-is: 0 fetch, 0 merge. Pairs adopt
        # unconditionally(每組覆蓋 2,不動額度 slack);singletons adopt only
        # while quota-feasible(B variant guard below)——denied 者仍可能被
        # greedy 判回 singleton 而走 probe(pass 1)。
        used: set = set()
        adopted_groups: List[frozenset] = []
        for members in list(reversed(self.index[li])):   # most recent first
            if len(adopted_groups) >= K:
                break
            if not members.issubset(active_set) or (members & used):
                continue
            if len(members) == 1:
                # B variant (2026-07-11, jobs 259050/259089): singleton 續留
                # 特權——adopt 讓熱門 expert 以精確權重續任 singleton,不被
                # greedy 配對稀釋(兩次 C2.1 純語意跑 AccR 都落在 0.627,低於
                # C1/C2 的 0.70 帶)。特權排在額度記帳之後:adopt 後剩餘的
                # active 必須仍塞得進剩餘額度(每組覆蓋 cap=ceil(M/K)),否則跳過、
                # 交給 greedy——組數因此永遠 ≤ K,溢位維持構造性不可能。
                r_after = len(active_set) - len(used) - 1
                if r_after > cap * (K - len(adopted_groups) - 1):
                    self.sgl_feas_denied_n += 1
                    continue
            slot, _ = self.index[li][members]
            # D0:slot buffer 開機預配且無 discard 路徑 → handle 恆有效。
            tensors = disp.get_merged_slot(li, slot, 0)
            self.index[li][members] = (slot, cur)   # refresh stamp
            self.index[li].move_to_end(members)
            used |= members
            adopted_groups.append(members)
            slot_ids.append(slot)
            _pin_now()
            self.hit_n += 1
            if len(members) == 1:
                self.singleton_hit_n += 1
                _pin_singleton(next(iter(members)))   # fetch-once verify pin
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
                    pair_sim=draft._pair_sim_table(li),
                    group_cap=cap)
                groups = draft.cluster_method.assign(ctx, k_rem)
                if len(groups) > k_rem:
                    # No-data fallback(hybrid/cooccur 的表在題首尚未累積時
                    # greedy_pair 回全 singleton)會超出剩餘額度——legacy
                    # 模式無所謂,cache mode 的組數受 slot 上限硬約束
                    # (f15 259303/259305:S=16 fail-fast)。壓回額度:
                    # 低權重 singleton 兩兩合併、高權重保持純 singleton
                    # (與 freq_slice 的鄰接配對精神一致)。只影響表格
                    # 空白的最初幾個 build。
                    singles = sorted((g for g in groups if len(g) == 1),
                                     key=lambda g: weights[g[0]])
                    multis = [g for g in groups if len(g) > 1]
                    need = len(groups) - k_rem
                    while need > 0 and len(singles) >= 2:
                        # 低權重 singleton 打包,一包最多 cap 顆(cap=2 時
                        # 位元級同舊行為 = 兩兩配對);每包減少組數 take-1。
                        take = min(cap, need + 1, len(singles))
                        pack: List[int] = []
                        for _ in range(take):
                            pack += singles.pop(0)
                        multis.append(pack)
                        need -= take - 1
                    groups = multis + singles

        # Singleton note: the DRAFT is always served from an identity slot
        # (our own torch buffers) — holding refs to the resident original is
        # unsafe: archer's prefetcher relocates nodes regardless of pinned_
        # and SetDevice(host) flips held refs to CPU in place (jobs
        # 258555/258658). The original is STILL pinned (budget permitting),
        # purely for the dual-identity verify hit — losing that pin costs a
        # re-fetch, never correctness.
        #
        # Processing ORDER (C2.1 fix, job 259050): singleton probes run
        # BEFORE pair allocs. Pair allocs steal slots and singletons are the
        # preferred steal victims, so probing at the end of the loop let the
        # same build steal entries that were about to be probe hits (adopt
        # 0.51→0.38, protected steals 0.1%→19%). Probing first refreshes
        # their stamps (this-build ⇒ unstealable). Emit order is free — the
        # final list is re-sorted by mass.
        single_miss: List[List[int]] = []
        for g in groups:                      # pass 1: singleton probes
            if len(g) != 1:
                continue
            _pin_singleton(g[0])
            key = frozenset(g)
            cached = self.index[li].get(key)
            if cached is not None:
                tensors = disp.get_merged_slot(li, cached[0], 0)
                self.index[li][key] = (cached[0], cur)
                self.index[li].move_to_end(key)
                slot_ids.append(cached[0])
                _pin_now()
                self.hit_n += 1
                self.singleton_hit_n += 1
                self.elided_bytes += self.expert_bytes
                _emit(g, _dict3(tensors),
                      sum(weights[m] for m in g), src="singleton-adopt")
                continue
            single_miss.append(g)

        for g in groups:                      # pass 2: pair merges (steal ok)
            if len(g) == 1:
                continue
            mass = sum(weights[m] for m in g)
            # Pair / group: uniform coefficients (cache-mode precondition) →
            # the merged content is determined by the member set alone.
            key = frozenset(g)
            slot = self._alloc_slot(li, active_set, li_cooccur, cur)
            uw = [1.0 / len(g)] * len(g)
            if not _write_slot(slot, list(g), uw):
                raise RuntimeError(
                    f"merged_cache: merge into slot {slot} failed — layer "
                    f"{li} group {sorted(g)}")
            tensors = disp.get_merged_slot(li, slot, 0)
            self.index[li][key] = (slot, cur)
            slot_ids.append(slot)
            _pin_now()
            self.miss_n += 1
            _emit(g, _dict3(tensors), mass, src="pair-slot")

        for g in single_miss:                 # pass 3: identity copies
            i = g[0]
            key = frozenset(g)
            slot = self._alloc_slot(li, active_set, li_cooccur, cur)
            if not disp.merge_experts_to_slot(li, slot, [i], [1.0], 0):
                raise RuntimeError(
                    f"merged_cache: identity merge into slot {slot} failed "
                    f"— layer {li} expert {i}")
            tensors = disp.get_merged_slot(li, slot, 0)
            self.index[li][key] = (slot, cur)
            slot_ids.append(slot)
            _pin_now()
            self.singleton_slot_n += 1
            self.miss_n += 1
            _emit(g, _dict3(tensors),
                  sum(weights[m] for m in g), src="singleton-slot")

        if pipeline and job_slots:
            if not disp.submit_merge_jobs(li, job_slots, job_members,
                                          job_weights, list(routed)):
                raise RuntimeError(
                    f"merged_cache: submit_merge_jobs failed — layer {li} "
                    f"slots {job_slots}")

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
