"""Unit tests for the NAEE baseline (baseline_tables_plan.md WS-D):
`naee_losses` candidate evaluation and the static_mask draft. Pure CPU."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

from search_naee import make_candidates, naee_losses  # noqa: E402

from aug_spec.drafts import get_draft
from aug_spec.drafts.static_mask import StaticMaskDraft


# ── D1: candidate evaluation ────────────────────────────────────────────

def _masked_route_ref(E, logits, kept, top_k, norm=True):
    """Independent re-derivation of the deployment masked-forward output
    for ONE kept set (loop form, no chunking) — the test oracle."""
    n, T, D = E.shape
    keep = torch.zeros(n, dtype=torch.bool)
    keep[torch.tensor(kept)] = True
    out = torch.zeros(T, D)
    for t in range(T):
        lg = logits[t].clone().float()
        lg[~keep] = float("-inf")
        p = lg.softmax(dim=-1)
        k = min(top_k, len(kept))
        w, idx = p.topk(k)
        if norm:
            w = w / w.sum()
        out[t] = sum(w[j] * E[idx[j], t].float() for j in range(k))
    return out


class TestNaeeLosses:
    def _setup(self, n=6, T=10, D=4, seed=0):
        g = torch.Generator()
        g.manual_seed(seed)
        E = torch.randn(n, T, D, generator=g)
        logits = torch.randn(T, n, generator=g)
        return E, logits

    def test_full_set_reproduces_reference(self):
        # ref built with ALL experts kept → the all-expert candidate must
        # score ~0 while any strict subset scores > 0.
        E, logits = self._setup()
        ref = _masked_route_ref(E, logits, list(range(6)), top_k=2)
        cands = torch.arange(6).unsqueeze(0)
        losses = naee_losses(E, logits, cands, top_k=2, ref=ref)
        assert losses[0] == pytest.approx(0.0, abs=1e-4)

    def test_matches_loop_oracle_and_ranks(self):
        E, logits = self._setup()
        ref = _masked_route_ref(E, logits, list(range(6)), top_k=2)
        cands, exact = make_candidates(6, 3, budget=100, seed=0)
        assert exact and cands.shape[0] == 20
        losses = naee_losses(E, logits, cands, top_k=2, ref=ref,
                             cand_chunk=7, token_chunk=3)   # exercise chunking
        for c in range(cands.shape[0]):
            oracle = (_masked_route_ref(E, logits, cands[c].tolist(), 2)
                      - ref).pow(2).sum().sqrt()
            assert losses[c] == pytest.approx(float(oracle), rel=1e-4)

    def test_r_smaller_than_topk(self):
        # Mixtral 8→1 case: r=1 < top_k=2 — zero-prob padding winners must
        # not break the math (weight renorm gives the kept expert w=1).
        E, logits = self._setup(n=4)
        cands = torch.tensor([[0], [1], [2], [3]])
        ref = _masked_route_ref(E, logits, [2], top_k=2)
        losses = naee_losses(E, logits, cands, top_k=2, ref=ref)
        assert int(losses.argmin()) == 2
        assert losses[2] == pytest.approx(0.0, abs=1e-4)

    def test_sampled_candidates_deterministic(self):
        a, ea = make_candidates(128, 16, budget=50, seed=7)
        b, eb = make_candidates(128, 16, budget=50, seed=7)
        assert not ea and not eb
        assert torch.equal(a, b)
        assert a.shape == (50, 16)
        # sorted ids, no duplicates within a candidate
        assert all(len(set(row.tolist())) == 16 for row in a)


# ── D1 gptoss semantics (softmax-after-topk, plan item G4) ──────────────

def _masked_route_ref_gptoss(E, logits, kept, top_k):
    """gpt-oss oracle: mask → topk the LOGITS → softmax over those k values
    (mirrors gptoss.make_masked_forward). Loop form, no chunking."""
    n, T, D = E.shape
    keep = torch.zeros(n, dtype=torch.bool)
    keep[torch.tensor(kept)] = True
    out = torch.zeros(T, D)
    for t in range(T):
        lg = logits[t].clone().float()
        lg[~keep] = float("-inf")
        k = min(top_k, len(kept))
        vals, idx = lg.topk(k)
        w = vals.softmax(dim=-1)
        out[t] = sum(w[j] * E[idx[j], t].float() for j in range(k))
    return out


class TestNaeeGptOssSemantics:
    def test_matches_gptoss_oracle(self):
        E, logits = TestNaeeLosses()._setup()
        ref = _masked_route_ref_gptoss(E, logits, list(range(6)), top_k=2)
        cands, _ = make_candidates(6, 3, budget=100, seed=0)
        losses = naee_losses(E, logits, cands, top_k=2, ref=ref,
                             softmax_after_topk=True,
                             cand_chunk=7, token_chunk=3)
        for c in range(cands.shape[0]):
            oracle = (_masked_route_ref_gptoss(E, logits, cands[c].tolist(), 2)
                      - ref).pow(2).sum().sqrt()
            assert losses[c] == pytest.approx(float(oracle), rel=1e-4)

    def test_equivalent_to_renormed_softmax_topk(self):
        # A renormalised restriction of a softmax to its own top-k IS the
        # softmax over those k logits — so with norm_topk_prob=True the two
        # semantics must agree numerically. Cross-validates both branches.
        E, logits = TestNaeeLosses()._setup()
        ref = _masked_route_ref(E, logits, list(range(6)), top_k=2)
        cands, _ = make_candidates(6, 3, budget=100, seed=0)
        a = naee_losses(E, logits, cands, top_k=2, ref=ref,
                        norm_topk_prob=True)
        b = naee_losses(E, logits, cands, top_k=2, ref=ref,
                        softmax_after_topk=True)
        assert torch.allclose(a, b, rtol=1e-4)


# ── D2: static_mask draft ───────────────────────────────────────────────

def _write_spec(tmp_path, layers, r=2):
    p = tmp_path / "naee.json"
    p.write_text(json.dumps(
        {"model_id": "dummy/model", "r": r, "layers": layers}))
    return p


class _Adapter:
    def num_experts(self, block):
        return block


class TestStaticMask:
    def test_masks_frozen(self, tmp_path):
        path = _write_spec(tmp_path, {"0": {"kept": [1, 3], "loss": 0.5},
                                      "2": {"kept": [0, 2], "loss": 0.4}})
        d = StaticMaskDraft(spec_path=str(path))
        blocks = [(0, 4), (2, 4)]           # block = num_experts (see _Adapter)
        d.prepare(_Adapter(), blocks)
        c1, c2 = {}, {}
        d.prepopulate(_Adapter(), blocks, c1)
        d.refresh(_Adapter(), blocks, c1)   # no-op
        d.prepopulate(_Adapter(), blocks, c2)
        assert torch.equal(c1[0], torch.tensor([False, True, False, True]))
        assert torch.equal(c1[2], torch.tensor([True, False, True, False]))
        assert c1[0] is c2[0]

    def test_validation(self, tmp_path):
        d = StaticMaskDraft(spec_path=str(_write_spec(
            tmp_path, {"0": {"kept": [1, 9], "loss": 0}})))
        with pytest.raises(ValueError, match="kept set invalid"):
            d.prepare(_Adapter(), [(0, 4)])
        d2 = StaticMaskDraft(spec_path=str(_write_spec(
            tmp_path, {"0": {"kept": [1], "loss": 0}})))
        with pytest.raises(ValueError, match="no entry for MoE layer 3"):
            d2.prepare(_Adapter(), [(3, 4)])

    def test_registry(self, tmp_path):
        path = _write_spec(tmp_path, {"0": {"kept": [0], "loss": 0}})
        assert isinstance(get_draft("static_mask", spec_path=str(path)),
                          StaticMaskDraft)
