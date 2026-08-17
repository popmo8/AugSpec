"""Unit tests for the MC-SMoE baseline: offline grouping
(clustering/mc_smoe.py), permutation-aligned merging and the mc_smoe
draft (drafts/mc_smoe.py). Pure CPU — login-node safe."""

from __future__ import annotations

import json

import pytest
import torch
import torch.nn.functional as F

from aug_spec.clustering.mc_smoe import (group_by_similarity,
                                         router_logits_similarity,
                                         select_dominant_adaptive)
from aug_spec.drafts import get_draft
from aug_spec.drafts.mc_smoe import (MCSMoEDraft, match_permutation,
                                     permuted_linear_merge)


# ── SwiGLU fakes (qwen3-shaped: gate_proj/up_proj/down_proj modules) ────

class _W:
    def __init__(self, t):
        self.weight = t


class SwigluExpert:
    def __init__(self, gate, up, down):
        self.gate_proj = _W(gate)
        self.up_proj = _W(up)
        self.down_proj = _W(down)

    def tensors(self):
        return (self.gate_proj.weight, self.up_proj.weight,
                self.down_proj.weight)

    def forward(self, x):
        gate = F.linear(x, self.gate_proj.weight)
        up = F.linear(x, self.up_proj.weight)
        return F.linear(F.silu(gate) * up, self.down_proj.weight)


def _rand_expert(g, dim=4, inter=6):
    return SwigluExpert(torch.randn(inter, dim, generator=g),
                        torch.randn(inter, dim, generator=g),
                        torch.randn(dim, inter, generator=g))


def _permuted_clone(e: SwigluExpert, perm: torch.Tensor) -> SwigluExpert:
    """Function-identical copy of `e` with its intermediate neurons
    shuffled by `perm` — the alignment problem's ground truth."""
    return SwigluExpert(e.gate_proj.weight[perm], e.up_proj.weight[perm],
                        e.down_proj.weight[:, perm])


class SwigluBlock:
    def __init__(self, experts):
        self.experts = experts


class SwigluAdapter:
    name = "qwen3_moe"

    def num_experts(self, block):
        return len(block.experts)

    def build_weighted_avg(self, block, weights):
        keys = ("gate_proj", "up_proj", "down_proj")
        out = {k: torch.zeros_like(getattr(block.experts[0], k).weight)
               for k in keys}
        for w, e in zip(weights, block.experts):
            for k in keys:
                out[k] = out[k] + w * getattr(e, k).weight
        return out


# ── dominant-expert selection (adaptive layer-wise ratio) ───────────────

class TestSelectDominant:
    def test_adaptive_split(self):
        # Layer 0 uniform (all normalise to 1.0), layer 1 heavily skewed:
        # the shared budget 2*2=4 goes 3:1.
        freq = torch.tensor([[10, 10, 10, 10], [100, 1, 1, 1]])
        assert select_dominant_adaptive(freq, 2) == [[0, 1, 2], [0]]

    def test_budget_and_coverage(self):
        g = torch.Generator()
        g.manual_seed(0)
        freq = torch.randint(0, 100, (6, 16), generator=g)
        for k in (1, 4, 16):
            dom = select_dominant_adaptive(freq, k)
            assert sum(len(d) for d in dom) == 6 * k
            assert all(len(d) >= 1 for d in dom)
            assert all(d == sorted(set(d)) for d in dom)

    def test_zero_layer_still_seeded(self):
        freq = torch.tensor([[0, 0, 0], [5, 1, 1]])
        assert select_dominant_adaptive(freq, 1) == [[0], [0]]

    def test_validation(self):
        with pytest.raises(ValueError):
            select_dominant_adaptive(torch.zeros(3), 1)
        with pytest.raises(ValueError):
            select_dominant_adaptive(torch.zeros(2, 4), 0)
        with pytest.raises(ValueError):
            select_dominant_adaptive(torch.zeros(2, 4), 5)


# ── router-logits similarity + grouping ─────────────────────────────────

class TestGrouping:
    def test_router_logits_similarity(self):
        # Experts 0 and 1 see identical logits, expert 2 orthogonal ones.
        logits = torch.tensor([[1.0, 1.0, 0.0], [2.0, 2.0, 0.0],
                               [0.0, 0.0, 3.0], [0.0, 0.0, -1.0]])
        sim = router_logits_similarity(logits)
        assert sim.shape == (3, 3)
        assert sim[0, 1] == pytest.approx(1.0)
        assert sim[0, 2] == pytest.approx(0.0)
        assert torch.allclose(sim, sim.t())

    def test_assignment(self):
        sim = torch.tensor([[1.0, 0.9, 0.1, 0.2],
                            [0.9, 1.0, 0.0, 0.3],
                            [0.1, 0.0, 1.0, 0.8],
                            [0.2, 0.3, 0.8, 1.0]])
        assert group_by_similarity(sim, [0, 2]) == [[0, 1], [2, 3]]

    def test_tie_prefers_smaller_dominant(self):
        sim = torch.full((3, 3), 0.5)
        assert group_by_similarity(sim, [1, 2]) == [[0, 1], [2]]

    def test_validation(self):
        sim = torch.eye(3)
        for bad in ([], [0, 0], [3], [-1]):
            with pytest.raises(ValueError):
                group_by_similarity(sim, bad)
        with pytest.raises(ValueError):
            group_by_similarity(torch.zeros(2, 3), [0])


# ── permutation alignment + merge ───────────────────────────────────────

class TestPermutedMerge:
    def test_match_recovers_planted_permutation(self):
        g = torch.Generator()
        g.manual_seed(0)
        a = _rand_expert(g)
        perm = torch.tensor([3, 0, 5, 1, 4, 2])
        b = _permuted_clone(a, perm)
        q = match_permutation(a.tensors(), b.tensors())
        for ra, rb in ((a.gate_proj, b.gate_proj), (a.up_proj, b.up_proj)):
            assert torch.equal(rb.weight[q], ra.weight)
        assert torch.equal(b.down_proj.weight[:, q], a.down_proj.weight)

    def test_merge_of_permuted_clone_is_identity(self):
        # {A, P·A} merged with any weights must give back exactly A —
        # the whole point of the alignment step.
        g = torch.Generator()
        g.manual_seed(1)
        a = _rand_expert(g)
        b = _permuted_clone(a, torch.tensor([1, 2, 3, 4, 5, 0]))
        block = SwigluBlock([a, b])
        merged = permuted_linear_merge(SwigluAdapter(), block, [0, 1],
                                       [0.75, 0.25], ref_id=0)
        for k, t in zip(("gate_proj", "up_proj", "down_proj"), a.tensors()):
            assert torch.allclose(merged[k], t, atol=1e-6)
        # Sanity: the unaligned average would NOT reproduce A.
        naive = SwigluAdapter().build_weighted_avg(block, [0.75, 0.25])
        assert not torch.allclose(naive["gate_proj"], a.gate_proj.weight,
                                  atol=1e-3)

    def test_merged_expert_matches_reference_function(self):
        g = torch.Generator()
        g.manual_seed(2)
        a = _rand_expert(g)
        b = _permuted_clone(a, torch.randperm(6, generator=g))
        merged = permuted_linear_merge(SwigluAdapter(), SwigluBlock([a, b]),
                                       [0, 1], [0.5, 0.5], ref_id=0)
        x = torch.randn(5, 4, generator=g)
        out = SwigluExpert(merged["gate_proj"], merged["up_proj"],
                           merged["down_proj"]).forward(x)
        assert torch.allclose(out, a.forward(x), atol=1e-5)

    def test_unsupported_adapter_raises(self):
        class GptOssAdapter:
            name = "gptoss"
        with pytest.raises(NotImplementedError, match="gptoss"):
            permuted_linear_merge(GptOssAdapter(), SwigluBlock([]), [0],
                                  [1.0], ref_id=0)


# ── mc_smoe draft (spec → frozen multi cache) ───────────────────────────

def _write_spec(tmp_path, layers, with_dominant=True):
    spec = {"model_id": "dummy/model", "method": "mc_smoe", "K": 2,
            "count_top_k": 2, "layers": layers}
    if not with_dominant:
        for entry in spec["layers"].values():
            entry.pop("dominant", None)
    p = tmp_path / "spec.json"
    p.write_text(json.dumps(spec))
    return p


class TestMCSMoEDraft:
    def test_prepare_aligned_merge(self, tmp_path):
        g = torch.Generator()
        g.manual_seed(3)
        a = _rand_expert(g)
        b = _permuted_clone(a, torch.tensor([2, 0, 1, 5, 3, 4]))
        c, d = _rand_expert(g), _rand_expert(g)
        block = SwigluBlock([a, b, c, d])
        path = _write_spec(tmp_path, {"0": {
            "groups": [[0, 1], [2, 3]], "dominant": [0, 2],
            "freq": [3, 1, 2, 1]}})
        draft = MCSMoEDraft(spec_path=str(path), draft_top_k=1)
        draft.prepare(SwigluAdapter(), [(0, block)])
        cache = draft._built[0]
        assert cache["kind"] == "multi"
        # Mass order: group {0,1} (4/7) before {2,3} (3/7).
        assert cache["indices"] == [[0, 1], [2, 3]]
        # Group {A, P·A}: aligned frequency-weighted merge == A exactly.
        for k, t in zip(("gate_proj", "up_proj", "down_proj"), a.tensors()):
            assert torch.allclose(cache["experts"][0][k], t, atol=1e-6)
        assert cache["experts"][1]["gate_proj"].shape == (6, 4)

    def test_singleton_group_uses_plain_merge(self, tmp_path):
        g = torch.Generator()
        g.manual_seed(4)
        block = SwigluBlock([_rand_expert(g), _rand_expert(g),
                             _rand_expert(g)])
        path = _write_spec(tmp_path, {"0": {
            "groups": [[0], [1, 2]], "dominant": [0, 1],
            "freq": [5, 1, 1]}})
        draft = MCSMoEDraft(spec_path=str(path))
        draft.prepare(SwigluAdapter(), [(0, block)])
        assert torch.allclose(draft._built[0]["experts"][0]["gate_proj"],
                              block.experts[0].gate_proj.weight)

    def test_spec_without_dominant_raises(self, tmp_path):
        path = _write_spec(tmp_path, {"0": {
            "groups": [[0, 1]], "freq": [1, 1]}}, with_dominant=False)
        draft = MCSMoEDraft(spec_path=str(path))
        with pytest.raises(ValueError, match="dominant"):
            draft.prepare(SwigluAdapter(),
                          [(0, SwigluBlock([_rand_expert(torch.Generator()),
                                            _rand_expert(torch.Generator())]))])

    def test_dominant_not_in_group_raises(self, tmp_path):
        path = _write_spec(tmp_path, {"0": {
            "groups": [[0, 1]], "dominant": [3], "freq": [1, 1]}})
        draft = MCSMoEDraft(spec_path=str(path))
        g = torch.Generator()
        g.manual_seed(5)
        with pytest.raises(ValueError, match="not a member"):
            draft.prepare(SwigluAdapter(),
                          [(0, SwigluBlock([_rand_expert(g),
                                            _rand_expert(g)]))])

    def test_registry(self, tmp_path):
        path = _write_spec(tmp_path, {"0": {
            "groups": [[0], [1]], "dominant": [0, 1], "freq": [1, 1]}})
        assert isinstance(get_draft("mc_smoe", spec_path=str(path)),
                          MCSMoEDraft)
