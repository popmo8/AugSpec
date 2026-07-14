"""Unit tests for the Table-1 baseline infrastructure
(baseline_tables_plan.md WS-A): the run-fixed random drafts (A1/A2), the
`draft: none` / `run.humaneval` config fields (A3/A4), and the humaneval
subtask wiring. Pure CPU — login-node safe."""

from __future__ import annotations

import pytest
import torch

from aug_spec.drafts import get_draft
from aug_spec.drafts.random_mask import RandomMaskDraft
from aug_spec.drafts.random_merge import RandomMergeDraft


class FakeBlock:
    def __init__(self, n: int, dim: int = 4, seed: int = 0):
        g = torch.Generator()
        g.manual_seed(seed)
        self.weights = [torch.randn(dim, generator=g) for _ in range(n)]


class FakeAdapter:
    def num_experts(self, block):
        return len(block.weights)

    def build_weighted_avg(self, block, weights):
        out = torch.zeros_like(block.weights[0])
        for w, t in zip(weights, block.weights):
            if w:
                out = out + w * t
        return {"w": out}


# ── A1: random_mask — run-fixed num_keep mask ──────────────────────────

class TestRandomMask:
    def test_static_fixed_across_cycles_and_questions(self):
        d = RandomMaskDraft(num_experts=16, seed=0, num_keep=4)
        blocks = [(0, None), (1, None)]
        d.prepare(None, blocks)
        cache = {}
        d.prepopulate(None, blocks, cache)
        masks0 = {li: m.clone() for li, m in cache.items()}
        for m in masks0.values():
            assert m.dtype == torch.bool and int(m.sum()) == 4
        d.refresh(None, blocks, cache)          # cycle: static → unchanged
        d.prepopulate(None, blocks, cache)      # next question: same masks
        for li in cache:
            assert torch.equal(cache[li], masks0[li])

    def test_seed_reproducible(self):
        a = RandomMaskDraft(num_experts=32, seed=7, num_keep=8)
        b = RandomMaskDraft(num_experts=32, seed=7, num_keep=8)
        blocks = [(0, None), (1, None)]
        a.prepare(None, blocks)
        b.prepare(None, blocks)
        for li in (0, 1):
            assert torch.equal(a._masks[li], b._masks[li])

    def test_per_cycle_legacy_redraws(self):
        d = RandomMaskDraft(num_experts=128, seed=0, num_keep=16,
                            per_cycle=True)
        blocks = [(0, None)]
        d.prepare(None, blocks)
        cache = {}
        d.prepopulate(None, blocks, cache)
        m0 = cache[0].clone()
        d.refresh(None, blocks, cache)
        assert int(cache[0].sum()) == 16
        assert not torch.equal(cache[0], m0)    # C(128,16) — collision ≈ 0

    def test_num_keep_validation(self):
        with pytest.raises(ValueError):
            RandomMaskDraft(num_experts=8, seed=0, num_keep=9)
        with pytest.raises(ValueError):
            RandomMaskDraft(num_experts=8, seed=0, num_keep=0)


# ── A2: random_merge — run-fixed random balanced partition ─────────────

class TestRandomMerge:
    def test_partition_properties_and_numerics(self):
        d = RandomMergeDraft(K=4, draft_top_k=2, seed=0)
        adapter = FakeAdapter()
        blocks = [(0, FakeBlock(8, seed=0)), (2, FakeBlock(8, seed=1))]
        d.prepare(adapter, blocks)
        for li, block in blocks:
            cache = d._built[li]
            assert cache["kind"] == "multi"
            groups = cache["indices"]
            assert len(groups) == 4
            flat = sorted(i for g in groups for i in g)
            assert flat == list(range(8))              # covers all, disjoint
            assert all(len(g) == 2 for g in groups)    # balanced 8/4
            assert abs(sum(cache["weights"]) - 1.0) < 1e-9
            for g, e in zip(groups, cache["experts"]):
                expect = torch.stack([block.weights[i] for i in g]).mean(0)
                assert torch.allclose(e["w"], expect, atol=1e-6)

    def test_frozen_across_questions(self):
        d = RandomMergeDraft(K=2, seed=3)
        adapter = FakeAdapter()
        blocks = [(0, FakeBlock(6))]
        d.prepare(adapter, blocks)
        c1, c2 = {}, {}
        d.prepopulate(adapter, blocks, c1)
        d.refresh(adapter, blocks, c1)              # no-op
        d.prepopulate(adapter, blocks, c2)
        assert c1[0] is c2[0]                       # same frozen dict
        d.prepare(adapter, blocks)                  # idempotent
        assert d._built[0] is c1[0]

    def test_lazy_build_serves_frozen_cache(self):
        # The compile warmup runs before any prepopulate — the averaged
        # forward falls back to lazy_build, which must serve the frozen
        # cache built in prepare() (regression: job 259444).
        d = RandomMergeDraft(K=2, seed=0)
        adapter = FakeAdapter()
        block = FakeBlock(6)
        d.prepare(adapter, [(0, block)])
        assert d.lazy_build(0, block, adapter) is d._built[0]
        assert d.lazy_build(99, block, adapter) is None

    def test_seed_changes_partition(self):
        adapter = FakeAdapter()
        parts = []
        for seed in (0, 1):
            d = RandomMergeDraft(K=4, seed=seed)
            d.prepare(adapter, [(0, FakeBlock(16))])
            parts.append(d._built[0]["indices"])
        assert parts[0] != parts[1]

    def test_registry(self):
        assert isinstance(get_draft("random_merge", K=4, seed=0),
                          RandomMergeDraft)
        assert isinstance(
            get_draft("random_mask", num_experts=8, seed=0, num_keep=2),
            RandomMaskDraft)


# ── A4: humaneval subtask wiring ────────────────────────────────────────

class TestHumanEvalWiring:
    def test_subtask_constants_and_matching(self):
        from aug_spec.runtime.specbench import (
            SPEC_BENCH_MT_BENCH_CATS, SPEC_BENCH_SUBTASKS, _category_matches)
        assert "humaneval" in SPEC_BENCH_SUBTASKS
        assert SPEC_BENCH_SUBTASKS[-1] == "overall"
        assert "humaneval" not in SPEC_BENCH_MT_BENCH_CATS
        # "coding" is an mt_bench SUB-CATEGORY — it must stay in mt_bench's
        # aggregation and never leak into the humaneval subtask (and vice
        # versa). This is why the category name is NOT "coding".
        assert _category_matches("coding", "mt_bench")
        assert not _category_matches("coding", "humaneval")
        assert _category_matches("humaneval", "humaneval")
        assert not _category_matches("humaneval", "mt_bench")
        assert _category_matches("humaneval", "overall")


class TestMtBenchPooledSampling:
    def _fake_questions(self):
        from aug_spec.runtime.specbench import SPEC_BENCH_MT_BENCH_CATS
        qs = []
        for cat in sorted(SPEC_BENCH_MT_BENCH_CATS):
            qs += [{"question_id": f"{cat}-{i}", "category": cat,
                    "turns": ["x"]} for i in range(10)]
        qs += [{"question_id": f"qa-{i}", "category": "qa", "turns": ["x"]}
               for i in range(80)]
        return qs

    def test_pooled_caps_mt_bench_total(self):
        from aug_spec.runtime.specbench import (
            SPEC_BENCH_MT_BENCH_CATS, _sample_questions)
        qs, _ = _sample_questions(self._fake_questions(), 15, seed=0,
                                  skip_categories=None, mt_bench_pooled=True)
        n_mt = sum(1 for q in qs
                   if q["category"] in SPEC_BENCH_MT_BENCH_CATS)
        n_qa = sum(1 for q in qs if q["category"] == "qa")
        assert n_mt == 15          # ONE pool of 80 → qpc total
        assert n_qa == 15

    def test_legacy_is_per_subcategory(self):
        from aug_spec.runtime.specbench import (
            SPEC_BENCH_MT_BENCH_CATS, _sample_questions)
        qs, _ = _sample_questions(self._fake_questions(), 15, seed=0,
                                  skip_categories=None, mt_bench_pooled=False)
        n_mt = sum(1 for q in qs
                   if q["category"] in SPEC_BENCH_MT_BENCH_CATS)
        assert n_mt == 80          # min(15, 10) per subcat × 8

    def test_pooled_deterministic_and_skip_compatible(self):
        from aug_spec.runtime.specbench import _sample_questions
        a, _ = _sample_questions(self._fake_questions(), 15, 3, None, True)
        b, _ = _sample_questions(self._fake_questions(), 15, 3, None, True)
        assert [q["question_id"] for q in a] == [q["question_id"] for q in b]
        c, skip = _sample_questions(self._fake_questions(), 15, 0,
                                    ["mt_bench"], True)
        assert all(q["category"] == "qa" for q in c) and len(c) == 15


# ── A3/A4: RunConfig fields ─────────────────────────────────────────────

class TestRunConfig:
    def test_draft_none_and_humaneval_flag(self, tmp_path):
        from aug_spec.cli import RunConfig
        f = tmp_path / "x.yaml"
        f.write_text("model:\n  id: dummy/model\n"
                     "draft:\n  name: none\n"
                     "run:\n  humaneval: true\n")
        cfg = RunConfig.from_yaml(f)
        assert cfg.draft_name == "none"
        assert cfg.humaneval is True

    def test_humaneval_defaults_off(self, tmp_path):
        from aug_spec.cli import RunConfig
        f = tmp_path / "y.yaml"
        f.write_text("model:\n  id: dummy/model\n"
                     "draft:\n  name: topm_count\n")
        cfg = RunConfig.from_yaml(f)
        assert cfg.humaneval is False
        assert cfg.batch_loop is False and cfg.batch_size == 1

    def test_batch_loop_fields(self, tmp_path):
        from aug_spec.cli import RunConfig
        f = tmp_path / "z.yaml"
        f.write_text("model:\n  id: dummy/model\n"
                     "draft:\n  name: topm_count\n"
                     "run:\n  batch_loop: true\n  batch_size: 16\n")
        cfg = RunConfig.from_yaml(f)
        assert cfg.batch_loop is True and cfg.batch_size == 16
