"""Unit tests for the Draft&Verify sublayer-skip draft
(drafts/draft_verify.py) and the BO search helpers
(runtime/dv_search.py). Pure CPU — login-node safe."""

from __future__ import annotations

import json

import pytest
import torch
import torch.nn as nn

from aug_spec.runtime.dv_search import (
    count_agreement, default_spec_path, heuristic_keeps, keep_from_params,
    params_for_keep, resolve_num_keep)

from aug_spec.controller import Controller
from aug_spec.drafts import get_draft
from aug_spec.drafts.draft_verify import DraftVerifyDraft


# ── tiny decoder mimicking the pre-norm sublayer structure ─────────────

class TinyAttn(nn.Module):
    """Contribution +1 per layer; returns (out, weights) like HF attention."""

    def forward(self, hidden_states, **kwargs):
        return torch.ones_like(hidden_states), None


class TinyMoE(nn.Module):
    """Forward is swapped by the controller (FakeAdapter → +10)."""

    def forward(self, hidden_states):
        return torch.full_like(hidden_states, 10.0)


class TinyLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = TinyAttn()
        self.mlp = TinyMoE()

    def forward(self, hidden_states, *args, **kwargs):
        attn_out, _ = self.self_attn(hidden_states=hidden_states)
        hidden_states = hidden_states + attn_out
        mlp_out = self.mlp(hidden_states)
        if isinstance(mlp_out, tuple):
            mlp_out = mlp_out[0]
        return hidden_states + mlp_out


class TinyModel(nn.Module):
    def __init__(self, n_layers: int):
        super().__init__()
        inner = nn.Module()
        inner.layers = nn.ModuleList(TinyLayer() for _ in range(n_layers))
        self.model = inner

    def forward(self, x):
        for layer in self.model.layers:
            x = layer(x)
        return x


class FakeAdapter:
    """Just enough surface for Controller: iter_moe + the masked factory."""

    name = "fake"

    def iter_moe(self, model):
        for i, layer in enumerate(model.model.layers):
            yield i, layer.mlp

    def make_masked_forward(self, controller, layer_idx, block):
        return lambda block, hidden_states: torch.full_like(
            hidden_states, 10.0)

    # MoEAdapter's default skip-return conventions (see adapters/base.py).
    def mlp_skip_output(self, hidden_states):
        return torch.zeros_like(hidden_states), None

    def decoder_skip_output(self, hidden_states, *args, **kwargs):
        return hidden_states


def make_installed(n_layers=4, **draft_args):
    model = TinyModel(n_layers)
    draft = DraftVerifyDraft(**draft_args)
    controller = Controller(model, FakeAdapter(), draft)
    controller.install()
    return model, draft, controller


# ── draft mechanism ─────────────────────────────────────────────────────

class TestDraftVerifyDraft:
    def test_registry(self):
        d = get_draft("draft_verify", mlp_keep=[0, 2])
        assert isinstance(d, DraftVerifyDraft)
        assert d.cache_kind == "masked"
        assert not d.holds_merged_residency

    def test_draft_phase_skips_selected_sublayers(self):
        model, draft, controller = make_installed(
            n_layers=4, mlp_keep=[0], attn_skip=[2])
        x = torch.zeros(1, 3)
        # verify phase: every layer contributes 1 (attn) + 10 (mlp)
        controller.in_draft_phase = False
        assert torch.equal(model(x), torch.full_like(x, 44.0))
        # draft phase: attn at layers {0,1,3}, mlp at layer 0 only
        controller.in_draft_phase = True
        assert torch.equal(model(x), torch.full_like(x, 13.0))
        controller.in_draft_phase = False
        assert torch.equal(model(x), torch.full_like(x, 44.0))

    def test_uninstall_restores(self):
        model, draft, controller = make_installed(n_layers=4, mlp_keep=[1])
        controller.uninstall()
        assert draft._wrapped == []
        controller.in_draft_phase = True
        x = torch.zeros(1, 3)
        # pristine TinyMoE forward (+10) is back on every layer
        assert torch.equal(model(x), torch.full_like(x, 44.0))
        controller.uninstall()          # second uninstall is a no-op

    def test_spec_path_loading(self, tmp_path):
        spec = tmp_path / "dv.json"
        spec.write_text(json.dumps(
            {"mlp_keep": [0, 3], "attn_skip": [1]}))
        model, draft, controller = make_installed(
            n_layers=4, spec_path=str(spec))
        assert draft.mlp_keep == [0, 3]
        assert draft.attn_skip == [1]
        x = torch.zeros(1, 3)
        controller.in_draft_phase = True
        # attn at {0,2,3} = 3, mlp at {0,3} = 20
        assert torch.equal(model(x), torch.full_like(x, 23.0))

    def test_arg_validation(self, tmp_path):
        with pytest.raises(ValueError):        # neither source
            DraftVerifyDraft()
        with pytest.raises(ValueError):        # both sources
            DraftVerifyDraft(spec_path="x.json", mlp_keep=[0])
        with pytest.raises(ValueError):        # empty keep set
            DraftVerifyDraft(mlp_keep=[])
        with pytest.raises(ValueError):        # duplicate indices
            DraftVerifyDraft(mlp_keep=[1, 1])
        with pytest.raises(ValueError):        # layer 0 attention must run
            DraftVerifyDraft(mlp_keep=[0], attn_skip=[0])
        spec = tmp_path / "dv.json"
        spec.write_text(json.dumps({"mlp_keep": [0]}))
        with pytest.raises(ValueError):        # attn_skip comes from spec
            DraftVerifyDraft(spec_path=str(spec), attn_skip=[1])

    def test_install_validation(self):
        model = TinyModel(4)
        for bad_args in ({"mlp_keep": [0, 7]},          # out of range
                         {"mlp_keep": [0], "attn_skip": [9]},
                         {"mlp_keep": [0, 1, 2, 3]}):   # nothing skipped
            controller = Controller(model, FakeAdapter(),
                                    DraftVerifyDraft(**bad_args))
            with pytest.raises(ValueError):
                controller.install()
            controller.uninstall()

    def test_requires_model_dot_layers(self):
        draft = DraftVerifyDraft(mlp_keep=[0])

        class NoLayers:
            model = object()
            in_draft_phase = False

        with pytest.raises(TypeError):
            draft.post_install(NoLayers())


# ── DV1: search helpers ─────────────────────────────────────────────────

class TestSearchHelpers:
    def test_keep_from_params_top_k_with_tie_break(self):
        moe_ids = [0, 1, 2, 3]
        params = {"l0": 0.2, "l1": 0.9, "l2": 0.2, "l3": 0.5}
        # top-3: l1, l3, then the 0.2 tie breaks toward the earlier layer
        assert keep_from_params(params, moe_ids, 3) == (0, 1, 3)

    def test_params_for_keep_round_trips(self):
        moe_ids = list(range(8))
        keep = (1, 4, 6)
        params = params_for_keep(keep, moe_ids)
        assert keep_from_params(params, moe_ids, 3) == keep

    def test_heuristic_keeps(self):
        h = heuristic_keeps(list(range(48)), 6)
        assert h["first_k"] == (0, 1, 2, 3, 4, 5)
        assert h["last_k"] == (42, 43, 44, 45, 46, 47)
        assert h["even_k"] == (0, 8, 16, 24, 32, 40)

    def test_resolve_num_keep(self):
        assert resolve_num_keep(48) == 6           # 12.5% budget
        assert resolve_num_keep(24) == 3           # gpt-oss depth
        assert resolve_num_keep(32) == 4           # mixtral depth
        assert resolve_num_keep(4) == 1            # floor at 1
        assert resolve_num_keep(48, num_keep=8) == 8
        assert resolve_num_keep(48, keep_frac=0.25) == 12

    def test_default_spec_path(self):
        p = default_spec_path("Qwen/Qwen3-30B-A3B-Base", 6)
        assert str(p) == "output/draft_verify/Qwen3-30B-A3B-Base_L6.json"

    def test_count_agreement(self):
        # B=2, prompt_len=2; ids row0 gen [5, 6, 7], row1 gen [8] (padded)
        ids = torch.tensor([[1, 2, 5, 6, 7],
                            [3, 4, 8, 0, 0]])
        # pred[b, t] predicts position t+1
        pred = torch.zeros_like(ids)
        pred[0, 1], pred[0, 2], pred[0, 3] = 5, 6, 9   # 2 of 3 match
        pred[1, 1] = 8                                  # 1 of 1 match
        m, t = count_agreement(pred, ids, prompt_len=2, gen_lens=[3, 1])
        assert (m, t) == (3, 4)
        m, t = count_agreement(pred, ids, prompt_len=2, gen_lens=[3, 0])
        assert (m, t) == (2, 3)
