"""MergedCacheIndex(merged_cache_plan.md C1)的純 CPU 單元測試。

FakeDisp 鴨子型別替代 C++ dispatcher:驗 adopt-first、slot 配置/竊取、
singleton resident/identity 分流、pin 集合、probe-miss 對帳、fail-fast、
reset。不碰 GPU / moe_infinity。
"""

from types import SimpleNamespace

import pytest
import torch

from aug_spec.clustering import get_cluster_method
from aug_spec.runtime.merged_cache import MergedCacheIndex


class FakeDisp:
    def __init__(self, resident=()):
        self.submitted = []              # C3: (li, slots, members, routed)
        self.drains = 0
        self.slots = {}                  # (li, slot) -> [3 tensors]
        self.resident = set(resident)    # (li, expert_id) resident originals
        self.slot_pins = {}
        self.pins = {}
        self.merge_calls = []
        self.fail_merge = False

    # D0 契約:buffer 開機預配、穩定;merge 用 copy_ 就地覆寫,handle 恆
    # 有效(內容未寫入前為零)。
    def _slot_buf(self, li, slot):
        key = (li, slot)
        if key not in self.slots:
            self.slots[key] = [torch.zeros(2, 2) for _ in range(3)]
        return self.slots[key]

    def merge_experts_to_slot(self, li, slot, ids, ws, gpu):
        if self.fail_merge:
            return False
        self.merge_calls.append((li, slot, tuple(sorted(ids))))
        t = torch.full((2, 2), float(sum(ids) * 10 + len(ids)))
        for buf, src in zip(self._slot_buf(li, slot), (t, t, t)):
            buf.copy_(src)
        return True

    def get_merged_slot(self, li, slot, gpu):
        return self._slot_buf(li, slot)

    def get_resident_expert_weights(self, li, e, gpu):
        if (li, e) in self.resident:
            t = torch.full((2, 2), float(e))
            return [t, t.clone(), t.clone()]
        return []

    def set_merged_slot_pinned(self, li, ids, gpu):
        self.slot_pins[li] = list(ids)

    # C3 pipeline:同步執行 job(單元測試裡「到齊即 merge」立即發生),
    # 記錄 submit 供斷言。
    def submit_merge_jobs(self, li, slots, members, weights, routed):
        self.submitted.append((li, list(slots),
                               [list(m) for m in members], list(routed)))
        for sl, m, w in zip(slots, members, weights):
            if not self.merge_experts_to_slot(li, sl, m, w, 0):
                return False
        return True

    def wait_merges_done(self, timeout_s):
        self.drains += 1

    def set_pinned(self, li, ids, gpu):
        self.pins[li] = list(ids)

    def clear_pinned(self, gpu):
        self.pins = {}


class FakeDraft:
    def __init__(self, K):
        self.K = K
        self.cooccur = {}
        self.cluster_method = get_cluster_method("freq_slice")

    def _pair_sim_table(self, li):
        return None


def _block(disp):
    return SimpleNamespace(
        expert_executor=SimpleNamespace(expert_dispatcher=disp))


def _weights(n, hot):
    w = [0.0] * n
    for i, v in hot.items():
        w[i] = v
    s = sum(w)
    return [x / s for x in w]


def test_miss_then_adopt_and_output_shape():
    disp, draft = FakeDisp(), FakeDraft(K=2)
    mc = MergedCacheIndex(slots_per_layer=4, expert_bytes=100)
    w = _weights(8, {0: 4, 1: 3, 2: 2, 3: 1})   # 4 actives, K=2 → 2 pairs

    out = mc.build_layer(0, _block(disp), w, draft)
    assert out["kind"] == "multi" and len(out["experts"]) == 2
    assert abs(sum(out["weights"]) - 1.0) < 1e-6
    assert out["weights"][0] >= out["weights"][1]        # mass-desc order
    assert mc.miss_n == 2 and mc.hit_n == 0
    n_merges = len(disp.merge_calls)

    out2 = mc.build_layer(0, _block(disp), w, draft)
    assert mc.hit_n == 2                                  # adopted both
    assert len(disp.merge_calls) == n_merges              # no re-merge
    assert mc.elided_bytes == 2 * 2 * 100
    assert len(out2["experts"]) == 2


def test_singleton_slot_served_and_pinned():
    disp = FakeDisp()
    draft = FakeDraft(K=2)
    mc = MergedCacheIndex(slots_per_layer=4, expert_bytes=100)
    w = _weights(8, {0: 2, 5: 1})                         # 2 actives, K=2 → 2 singles
    mc.build_layer(0, _block(disp), w, draft)
    assert mc.singleton_slot_n == 2                       # BOTH served from slots
    assert mc.singleton_pinned_n == 2                     # both pinned (verify hits)
    assert sorted(disp.pins[0]) == [0, 5]
    assert len(disp.slot_pins[0]) == 2                    # identity slots pinned


def test_pins_replace_on_partition_change():
    disp, draft = FakeDisp(), FakeDraft(K=2)
    mc = MergedCacheIndex(slots_per_layer=4, expert_bytes=100)
    mc.build_layer(0, _block(disp), _weights(8, {0: 4, 1: 3, 2: 2, 3: 1}), draft)
    first_slots = set(disp.slot_pins[0])
    mc.build_layer(0, _block(disp), _weights(8, {4: 4, 5: 3, 6: 2, 7: 1}), draft)
    assert set(disp.slot_pins[0]).isdisjoint(first_slots) or \
        len(disp.slot_pins[0]) == 2                       # new working set pinned


def test_slot_steal_when_exhausted():
    disp, draft = FakeDisp(), FakeDraft(K=1)
    mc = MergedCacheIndex(slots_per_layer=1, expert_bytes=100)
    # K=1, 2 actives → freq_slice gives one group of 2 → one slot
    mc.build_layer(0, _block(disp), _weights(8, {0: 2, 1: 1}), draft)
    assert len(mc.index[0]) == 1
    # different pair → must steal the single slot
    mc.build_layer(0, _block(disp), _weights(8, {2: 2, 3: 1}), draft)
    assert len(mc.index[0]) == 1
    assert frozenset({2, 3}) in mc.index[0]


def test_merge_failure_raises():
    """C2.1:fallback 已移除 — slot merge 失敗即 fail-fast。"""
    disp, draft = FakeDisp(), FakeDraft(K=1)
    disp.fail_merge = True
    mc = MergedCacheIndex(slots_per_layer=2, expert_bytes=100)
    with pytest.raises(RuntimeError, match="merge into slot"):
        mc.build_layer(0, _block(disp), _weights(8, {0: 2, 1: 1}), draft)


def test_reset_clears_index_and_pins():
    disp, draft = FakeDisp(), FakeDraft(K=2)
    mc = MergedCacheIndex(slots_per_layer=4, expert_bytes=100)
    mc.build_layer(0, _block(disp), _weights(8, {0: 4, 1: 3, 2: 2, 3: 1}), draft)
    mc.reset([(0, _block(disp))], disp)
    assert not mc.index
    assert disp.slot_pins[0] == [] and disp.pins == {}


def test_singleton_pin_budget_denied():
    disp = FakeDisp()
    draft = FakeDraft(K=2)
    mc = MergedCacheIndex(slots_per_layer=4, expert_bytes=100,
                          singleton_pin_budget=100)   # room for ONE pin
    w = _weights(8, {0: 2, 5: 1})                     # two singletons
    mc.build_layer(0, _block(disp), w, draft)
    assert mc.singleton_slot_n == 2                   # slot-served regardless
    assert mc.singleton_pinned_n == 1                 # only first fits budget
    assert mc.singleton_budget_denied_n == 1
    assert disp.pins[0] == [0]                        # pin = verify-hit only


def test_steal_retention_order():
    """C2:偷 slot 順序 = 非保護 singleton → 冷 pair → 熱 pair → 保護級。"""
    mc = MergedCacheIndex(slots_per_layer=4, expert_bytes=100)
    idx = mc.index[0]
    idx[frozenset({0, 1})] = (0, 1)   # protected pair(成員都 active)
    idx[frozenset({9})] = (1, 1)      # unprotected singleton
    idx[frozenset({7, 8})] = (2, 1)   # unprotected 冷 pair
    idx[frozenset({5, 6})] = (3, 1)   # unprotected 熱 pair
    co = torch.zeros(12, 12)
    co[5][6] = 9.0
    co[7][8] = 1.0
    active = {0, 1, 2, 3}
    order = []
    for newk in ({20}, {21}, {22}):
        before = set(idx.keys())
        s = mc._alloc_slot(0, active, co, cur_stamp=2)
        order.append(next(iter(before - set(idx.keys()))))
        idx[frozenset(newk)] = (s, 2)
    assert order == [frozenset({9}), frozenset({7, 8}), frozenset({5, 6})]
    before = set(idx.keys())
    mc._alloc_slot(0, active, co, cur_stamp=2)
    assert frozenset({0, 1}) in before - set(idx.keys())   # 最後才偷保護級
    assert mc.steal_protected_n == 1 and mc.steal_n == 4


def test_steal_never_takes_this_build():
    mc = MergedCacheIndex(slots_per_layer=2, expert_bytes=100)
    idx = mc.index[0]
    idx[frozenset({0, 1})] = (0, 5)   # 本 build 的條目
    idx[frozenset({2, 3})] = (1, 4)   # 舊條目(同樣是保護級)
    s = mc._alloc_slot(0, {0, 1, 2, 3}, None, cur_stamp=5)
    assert s == 1 and frozenset({0, 1}) in idx   # 偷舊的,絕不偷本 build


def test_steal_refuses_this_build_only_pool():
    """全部 slot 都是本 build(分群 >K 組,只可能在 |active|>2K 的
    非 top-M draft 出現)→ fail-fast,絕不覆寫本 cycle 已 emit 的 slot。"""
    mc = MergedCacheIndex(slots_per_layer=2, expert_bytes=100)
    idx = mc.index[0]
    idx[frozenset({0, 1})] = (0, 3)
    idx[frozenset({2, 3})] = (1, 3)
    with pytest.raises(RuntimeError, match="slot demand exceeded"):
        mc._alloc_slot(0, {0, 1, 2, 3}, None, cur_stamp=3)
    assert mc.steal_n == 0 and len(idx) == 2   # 沒有條目被刪


def test_c21_singleton_probe_keeps_verify_pin():
    """C2.1:singleton 不在 stage-1 adopt,改由 greedy 後 probe 命中;
    命中路徑保留 archer pin(舊 adopt 路徑會丟 pin → 留用 expert 每輪重 fetch)。"""
    disp, draft = FakeDisp(), FakeDraft(K=2)
    mc = MergedCacheIndex(slots_per_layer=4, expert_bytes=100)
    w = _weights(8, {0: 2, 5: 1})                 # 2 actives, K=2 → 2 singles
    mc.build_layer(0, _block(disp), w, draft)
    n_merges = len(disp.merge_calls)
    mc.build_layer(0, _block(disp), w, draft)
    assert mc.singleton_hit_n == 2                # probe 命中,零重建
    assert len(disp.merge_calls) == n_merges      # 沒有 re-merge
    assert sorted(disp.pins[0]) == [0, 5]         # 原件 pin 仍在(fetch once)
    assert mc.singleton_pinned_n == 4             # 兩輪都走了 pin 記帳


def test_c21_no_overflow_when_m_le_2k():
    """C2.1:快取裡的 singleton 不再吃 stage-1 額度 → |active| ≤ 2K 時
    組數恆 ≤ K,不觸發 exhausted/fallback(舊行為會溢位)。"""
    disp, draft = FakeDisp(), FakeDraft(K=2)
    mc = MergedCacheIndex(slots_per_layer=2, expert_bytes=100)
    # 先養出兩個 singleton 條目
    mc.build_layer(0, _block(disp), _weights(8, {0: 2, 1: 1}), draft)
    assert mc.singleton_slot_n == 2
    # 4 actives(=2K)且兩個快取 singleton 都仍 active:
    # 舊行為 adopt {0},{1} 佔滿 K → 剩 {2},{3} 退化溢位;
    # 新行為 greedy 直接配成 2 對 → 剛好 2 slot。
    out = mc.build_layer(0, _block(disp), _weights(8, {0: 4, 1: 3, 2: 2, 3: 1}),
                         draft)
    assert len(out["experts"]) == 2               # 組數 = K,不觸發 fail-fast
    assert mc.sgl_feas_denied_n == 2              # B guard:slack=0 全數拒絕


def test_c21_probe_runs_before_pair_allocs():
    """順序不變量(job 259050 修復):同 build 內 singleton probe 必須先於
    pair merge——否則 pair 的 steal 會偷走即將命中的 singleton 條目。"""
    class RecordingDisp(FakeDisp):
        def __init__(self):
            super().__init__()
            self.ops = []

        def get_merged_slot(self, li, slot, gpu):
            self.ops.append(("get", slot))
            return super().get_merged_slot(li, slot, gpu)

        def merge_experts_to_slot(self, li, slot, ids, ws, gpu):
            self.ops.append(("merge", tuple(sorted(ids))))
            return super().merge_experts_to_slot(li, slot, ids, ws, gpu)

    disp, draft = RecordingDisp(), FakeDraft(K=2)
    mc = MergedCacheIndex(slots_per_layer=4, expert_bytes=100)
    mc.build_layer(0, _block(disp), _weights(16, {9: 1}), draft)   # 養出 {9}
    slot9 = mc.index[0][frozenset({9})][0]
    disp.ops.clear()
    mc.build_layer(0, _block(disp), _weights(16, {9: 4, 0: 2, 1: 1}), draft)
    assert mc.singleton_hit_n == 1
    probe_at = disp.ops.index(("get", slot9))
    pair_at = next(k for k, op in enumerate(disp.ops) if op[0] == "merge")
    assert probe_at < pair_at          # probe 佔 stamp 在任何 steal 之前


def test_b_singleton_adopt_feasibility_partial():
    """B variant:額度夠就 adopt(續留特權),不夠就拒絕交給 greedy。"""
    disp, draft = FakeDisp(), FakeDraft(K=2)
    mc = MergedCacheIndex(slots_per_layer=4, expert_bytes=100)
    mc.build_layer(0, _block(disp), _weights(8, {0: 2, 1: 1}), draft)  # 養 {0},{1}
    out = mc.build_layer(0, _block(disp), _weights(8, {0: 3, 1: 2, 2: 1}),
                         draft)
    # 最近優先:{1} adopt(r_after=2 ≤ 2);{0} 拒絕(r_after=1 > 0)→
    # greedy 把 [0,2] 配成 pair。
    assert mc.singleton_hit_n == 1 and mc.sgl_feas_denied_n == 1
    assert sorted(map(sorted, out["indices"])) == [[0, 2], [1]]


def test_c3_pipeline_mode_submits_jobs():
    """C3:routed 非 None → miss 組收集成 MergeJob 一次 submit;emit 的
    handle 與內容(FakeDisp 同步執行)與 P2 模式一致;adopt/probe 命中不產生
    job。"""
    disp, draft = FakeDisp(), FakeDraft(K=2)
    mc = MergedCacheIndex(slots_per_layer=4, expert_bytes=100)
    w = _weights(8, {0: 4, 1: 3, 2: 2, 3: 1})
    out = mc.build_layer(0, _block(disp), w, draft, routed=[0, 1, 2, 3])
    assert len(disp.submitted) == 1
    li, slots, members, routed = disp.submitted[0]
    assert li == 0 and len(slots) == 2 and routed == [0, 1, 2, 3]
    assert sorted(map(sorted, members)) == sorted(map(sorted, out["indices"]))
    # 第二輪全 adopt → 不 submit
    n = len(disp.submitted)
    mc.build_layer(0, _block(disp), w, draft, routed=[0, 1, 2, 3])
    assert len(disp.submitted) == n and mc.hit_n == 2


def test_c3_pipeline_content_matches_sync():
    """pipeline 與 P2 兩模式對同一輸入產出相同的 slot 內容。"""
    dp, d1 = FakeDisp(), FakeDraft(K=2)
    ds, d2 = FakeDisp(), FakeDraft(K=2)
    m1 = MergedCacheIndex(slots_per_layer=4, expert_bytes=100)
    m2 = MergedCacheIndex(slots_per_layer=4, expert_bytes=100)
    w = _weights(16, {0: 4, 1: 3, 9: 2})
    o1 = m1.build_layer(0, _block(dp), w, d1, routed=[0, 1, 9])
    o2 = m2.build_layer(0, _block(ds), w, d2)
    assert o1["indices"] == o2["indices"] and o1["weights"] == o2["weights"]
    for e1, e2 in zip(o1["experts"], o2["experts"]):
        for k in e1:
            assert torch.equal(e1[k], e2[k])
