"""MergedCacheIndex(merged_cache_plan.md C1)的純 CPU 單元測試。

FakeDisp 鴨子型別替代 C++ dispatcher:驗 adopt-first、slot 配置/竊取、
singleton resident/identity 分流、pin 集合、probe-miss 對帳、fallback、
reset。不碰 GPU / moe_infinity。
"""

from types import SimpleNamespace

import torch

from aug_spec.clustering import get_cluster_method
from aug_spec.runtime.merged_cache import MergedCacheIndex


class FakeDisp:
    def __init__(self, resident=()):
        self.slots = {}                  # (li, slot) -> [3 tensors]
        self.resident = set(resident)    # (li, expert_id) resident originals
        self.slot_pins = {}
        self.pins = {}
        self.merge_calls = []
        self.fail_merge = False

    def merge_experts_to_slot(self, li, slot, ids, ws, gpu):
        if self.fail_merge:
            return False
        self.merge_calls.append((li, slot, tuple(sorted(ids))))
        t = torch.full((2, 2), float(sum(ids) * 10 + len(ids)))
        self.slots[(li, slot)] = [t, t.clone(), t.clone()]
        return True

    def get_merged_slot(self, li, slot, gpu):
        return self.slots.get((li, slot), [])

    def get_resident_expert_weights(self, li, e, gpu):
        if (li, e) in self.resident:
            t = torch.full((2, 2), float(e))
            return [t, t.clone(), t.clone()]
        return []

    def set_merged_slot_pinned(self, li, ids, gpu):
        self.slot_pins[li] = list(ids)

    def set_pinned(self, li, ids, gpu):
        self.pins[li] = list(ids)

    def clear_pinned(self, gpu):
        self.pins = {}


class FakeDraft:
    def __init__(self, K):
        self.K = K
        self.cooccur = {}
        self.cluster_method = get_cluster_method("freq_slice")
        self.built = []

    def _pair_sim_table(self, li):
        return None

    def _build_one(self, adapter, block, weights):
        self.built.append(tuple(round(w, 3) for w in weights))
        return {"gate_proj": torch.zeros(2, 2),
                "up_proj": torch.zeros(2, 2),
                "down_proj": torch.zeros(2, 2)}


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

    out = mc.build_layer(0, _block(disp), w, draft, adapter=None)
    assert out["kind"] == "multi" and len(out["experts"]) == 2
    assert abs(sum(out["weights"]) - 1.0) < 1e-6
    assert out["weights"][0] >= out["weights"][1]        # mass-desc order
    assert mc.miss_n == 2 and mc.hit_n == 0
    n_merges = len(disp.merge_calls)

    out2 = mc.build_layer(0, _block(disp), w, draft, adapter=None)
    assert mc.hit_n == 2                                  # adopted both
    assert len(disp.merge_calls) == n_merges              # no re-merge
    assert mc.elided_bytes == 2 * 2 * 100
    assert len(out2["experts"]) == 2


def test_probe_miss_drops_entry_and_remerges():
    disp, draft = FakeDisp(), FakeDraft(K=2)
    mc = MergedCacheIndex(slots_per_layer=4, expert_bytes=100)
    w = _weights(8, {0: 4, 1: 3, 2: 2, 3: 1})
    mc.build_layer(0, _block(disp), w, draft, adapter=None)
    # simulate fetch-pressure discard of every slot
    disp.slots.clear()
    mc.build_layer(0, _block(disp), w, draft, adapter=None)
    assert mc.hit_n == 0 and mc.miss_n == 4               # all re-merged


def test_singleton_slot_served_and_pinned():
    disp = FakeDisp()
    draft = FakeDraft(K=2)
    mc = MergedCacheIndex(slots_per_layer=4, expert_bytes=100)
    w = _weights(8, {0: 2, 5: 1})                         # 2 actives, K=2 → 2 singles
    mc.build_layer(0, _block(disp), w, draft, adapter=None)
    assert mc.singleton_slot_n == 2                       # BOTH served from slots
    assert mc.singleton_pinned_n == 2                     # both pinned (verify hits)
    assert sorted(disp.pins[0]) == [0, 5]
    assert len(disp.slot_pins[0]) == 2                    # identity slots pinned


def test_pins_replace_on_partition_change():
    disp, draft = FakeDisp(), FakeDraft(K=2)
    mc = MergedCacheIndex(slots_per_layer=4, expert_bytes=100)
    mc.build_layer(0, _block(disp), _weights(8, {0: 4, 1: 3, 2: 2, 3: 1}),
                   draft, adapter=None)
    first_slots = set(disp.slot_pins[0])
    mc.build_layer(0, _block(disp), _weights(8, {4: 4, 5: 3, 6: 2, 7: 1}),
                   draft, adapter=None)
    assert set(disp.slot_pins[0]).isdisjoint(first_slots) or \
        len(disp.slot_pins[0]) == 2                       # new working set pinned


def test_slot_steal_when_exhausted():
    disp, draft = FakeDisp(), FakeDraft(K=1)
    mc = MergedCacheIndex(slots_per_layer=1, expert_bytes=100)
    # K=1, 2 actives → freq_slice gives one group of 2 → one slot
    mc.build_layer(0, _block(disp), _weights(8, {0: 2, 1: 1}), draft, None)
    assert len(mc.index[0]) == 1
    # different pair → must steal the single slot
    mc.build_layer(0, _block(disp), _weights(8, {2: 2, 3: 1}), draft, None)
    assert len(mc.index[0]) == 1
    assert frozenset({2, 3}) in mc.index[0]


def test_fallback_when_merge_fails():
    disp, draft = FakeDisp(), FakeDraft(K=1)
    disp.fail_merge = True
    mc = MergedCacheIndex(slots_per_layer=2, expert_bytes=100)
    mc.build_layer(0, _block(disp), _weights(8, {0: 2, 1: 1}), draft, None)
    assert mc.fallback_n == 1 and draft.built              # legacy path used


def test_reset_clears_index_and_pins():
    disp, draft = FakeDisp(), FakeDraft(K=2)
    mc = MergedCacheIndex(slots_per_layer=4, expert_bytes=100)
    mc.build_layer(0, _block(disp), _weights(8, {0: 4, 1: 3, 2: 2, 3: 1}),
                   draft, adapter=None)
    mc.reset([(0, _block(disp))], disp)
    assert not mc.index
    assert disp.slot_pins[0] == [] and disp.pins == {}


def test_singleton_pin_budget_denied():
    disp = FakeDisp()
    draft = FakeDraft(K=2)
    mc = MergedCacheIndex(slots_per_layer=4, expert_bytes=100,
                          singleton_pin_budget=100)   # room for ONE pin
    w = _weights(8, {0: 2, 5: 1})                     # two singletons
    mc.build_layer(0, _block(disp), w, draft, None)
    assert mc.singleton_slot_n == 2                   # slot-served regardless
    assert mc.singleton_pinned_n == 1                 # only first fits budget
    assert mc.singleton_budget_denied_n == 1
    assert disp.pins[0] == [0]                        # pin = verify-hit only
