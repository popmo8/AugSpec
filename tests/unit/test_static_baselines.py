"""Unit tests for the static baselines (baseline_tables_plan.md WS-C):
average-linkage agglomerative clustering (C1) and the static_merge draft
(C2). Pure CPU — login-node safe."""

from __future__ import annotations

import json

import pytest
import torch

from aug_spec.clustering.agglomerative import average_linkage_groups
from aug_spec.drafts import get_draft
from aug_spec.drafts.static_merge import StaticMergeDraft

from test_baseline_drafts import FakeAdapter, FakeBlock


# ── C1: average-linkage clustering ──────────────────────────────────────

def _dist(points):
    x = torch.tensor(points, dtype=torch.float32).unsqueeze(1)
    return torch.cdist(x, x)


class TestAverageLinkage:
    def test_obvious_clusters(self):
        D = _dist([0.0, 0.1, 10.0, 10.1, 20.0, 20.1])
        assert average_linkage_groups(D, 3) == [[0, 1], [2, 3], [4, 5]]

    def test_partition_properties(self):
        g = torch.Generator()
        g.manual_seed(0)
        o = torch.randn(16, 8, generator=g)
        D = torch.cdist(o, o)
        for K in (1, 4, 16):
            groups = average_linkage_groups(D, K)
            assert len(groups) == K
            flat = sorted(i for grp in groups for i in grp)
            assert flat == list(range(16))

    def test_deterministic(self):
        g = torch.Generator()
        g.manual_seed(1)
        o = torch.randn(12, 4, generator=g)
        D = torch.cdist(o, o)
        assert (average_linkage_groups(D, 3)
                == average_linkage_groups(D.clone(), 3))

    def test_average_linkage_beats_single_link_chain(self):
        # A chain 0-1-2-3 (unit gaps) + far pair {10, 10.4}: average linkage
        # at K=2 must cut the chain from the far pair, not chain everything.
        D = _dist([0.0, 1.0, 2.0, 3.0, 10.0, 10.4])
        assert average_linkage_groups(D, 2) == [[0, 1, 2, 3], [4, 5]]

    def test_validation(self):
        D = _dist([0.0, 1.0])
        with pytest.raises(ValueError):
            average_linkage_groups(D, 0)
        with pytest.raises(ValueError):
            average_linkage_groups(D, 3)
        with pytest.raises(ValueError):
            average_linkage_groups(torch.zeros(2, 3), 1)


# ── C2: static_merge draft ──────────────────────────────────────────────

def _write_spec(tmp_path, layers, model_id="dummy/model", K=2):
    spec = {"model_id": model_id, "K": K, "count_top_k": 2,
            "layers": layers}
    p = tmp_path / "spec.json"
    p.write_text(json.dumps(spec))
    return p


class TestStaticMerge:
    def test_frequency_weighted_merge(self, tmp_path):
        # n=6, groups {0,1,2} (freq 3,1,0) and {3,4,5} (freq 0,0,0).
        freq = [3, 1, 0, 0, 0, 0]
        path = _write_spec(tmp_path, {"0": {
            "groups": [[0, 1, 2], [3, 4, 5]], "freq": freq}})
        d = StaticMergeDraft(spec_path=str(path), draft_top_k=1)
        adapter = FakeAdapter()
        block = FakeBlock(6)
        d.prepare(adapter, [(0, block)])
        cache = d._built[0]
        assert cache["kind"] == "multi"
        # Ordered by mass desc: group {0,1,2} carries all the freq.
        assert cache["indices"] == [[0, 1, 2], [3, 4, 5]]
        assert cache["weights"][0] == pytest.approx(1.0)
        # Group 1: freq-weighted 0.75·w0 + 0.25·w1 (expert 2 has freq 0).
        expect0 = 0.75 * block.weights[0] + 0.25 * block.weights[1]
        assert torch.allclose(cache["experts"][0]["w"], expect0, atol=1e-6)
        # Group 2: zero total freq → uniform 1/3 each.
        expect1 = torch.stack([block.weights[i] for i in (3, 4, 5)]).mean(0)
        assert torch.allclose(cache["experts"][1]["w"], expect1, atol=1e-6)

    def test_frozen_and_lazy_build(self, tmp_path):
        path = _write_spec(tmp_path, {"0": {
            "groups": [[0, 1], [2, 3]], "freq": [1, 1, 1, 1]}})
        d = StaticMergeDraft(spec_path=str(path))
        adapter = FakeAdapter()
        blocks = [(0, FakeBlock(4))]
        d.prepare(adapter, blocks)
        c1, c2 = {}, {}
        d.prepopulate(adapter, blocks, c1)
        d.refresh(adapter, blocks, c1)              # no-op
        d.prepopulate(adapter, blocks, c2)
        assert c1[0] is c2[0]
        # Warmup runs before any prepopulate → lazy_build must serve the
        # frozen cache (job 259444 regression class).
        assert d.lazy_build(0, blocks[0][1], adapter) is d._built[0]

    def test_missing_layer_raises(self, tmp_path):
        path = _write_spec(tmp_path, {"0": {
            "groups": [[0, 1]], "freq": [1, 1]}})
        d = StaticMergeDraft(spec_path=str(path))
        with pytest.raises(ValueError, match="no entry for MoE layer 5"):
            d.prepare(FakeAdapter(), [(5, FakeBlock(2))])

    def test_bad_coverage_raises(self, tmp_path):
        path = _write_spec(tmp_path, {"0": {
            "groups": [[0, 1], [1, 2]], "freq": [1, 1, 1]}})  # 1 twice, 3 short
        d = StaticMergeDraft(spec_path=str(path))
        with pytest.raises(ValueError, match="does not partition"):
            d.prepare(FakeAdapter(), [(0, FakeBlock(4))])

    def test_registry(self, tmp_path):
        path = _write_spec(tmp_path, {"0": {
            "groups": [[0], [1]], "freq": [1, 1]}})
        assert isinstance(get_draft("static_merge", spec_path=str(path)),
                          StaticMergeDraft)


# ── C1→C2 mini integration: cluster → spec → draft ─────────────────────

def test_cluster_to_draft_roundtrip(tmp_path):
    g = torch.Generator()
    g.manual_seed(0)
    # 8 experts whose "outputs" form 4 tight pairs.
    centers = torch.randn(4, 16, generator=g) * 10
    o = torch.cat([centers + 0.01, centers + 0.02]).float()
    order = [0, 4, 1, 5, 2, 6, 3, 7]                # interleave the pairs
    o = o[torch.tensor(order).argsort()]            # experts 2i, 2i+1 pair up
    groups = average_linkage_groups(torch.cdist(o, o), 4)
    assert all(len(gr) == 2 for gr in groups)
    path = _write_spec(tmp_path, {"0": {
        "groups": groups, "freq": [1] * 8}}, K=4)
    d = StaticMergeDraft(spec_path=str(path), draft_top_k=2)
    d.prepare(FakeAdapter(), [(0, FakeBlock(8))])
    cache = d._built[0]
    assert len(cache["experts"]) == 4
    assert sorted(i for gr in cache["indices"] for i in gr) == list(range(8))
