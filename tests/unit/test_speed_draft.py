"""Unit tests for the SPEED early-exit draft (drafts/speed.py): the
decoder-layer skip wrapping via the Controller post_install/post_uninstall
hooks, phase gating, and restore. Pure CPU — login-node safe."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from aug_spec.controller import Controller
from aug_spec.drafts import get_draft
from aug_spec.drafts.speed import SpeedDraft


class TinyBlock(nn.Module):
    """Stands in for a MoE block; forward is swapped by the controller."""

    def forward(self, hidden_states):
        return hidden_states


class TinyLayer(nn.Module):
    """Stands in for a decoder layer: adds 1 so skips are countable."""

    def __init__(self):
        super().__init__()
        self.mlp = TinyBlock()

    def forward(self, hidden_states, *args, **kwargs):
        return hidden_states + 1


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
        return lambda block, hidden_states: hidden_states

    # MoEAdapter's default skip-return conventions (see adapters/base.py).
    def mlp_skip_output(self, hidden_states):
        return torch.zeros_like(hidden_states), None

    def decoder_skip_output(self, hidden_states, *args, **kwargs):
        return hidden_states


def make_installed(n_layers=8, num_layers=2):
    model = TinyModel(n_layers)
    draft = SpeedDraft(num_layers=num_layers)
    controller = Controller(model, FakeAdapter(), draft)
    controller.install()
    return model, draft, controller


class TestSpeedDraft:
    def test_registry(self):
        d = get_draft("speed", num_layers=6)
        assert isinstance(d, SpeedDraft)
        assert d.cache_kind == "masked"
        assert not d.holds_merged_residency

    def test_draft_phase_skips_trailing_layers(self):
        model, draft, controller = make_installed(n_layers=8, num_layers=2)
        x = torch.zeros(1, 3)
        controller.in_draft_phase = False
        assert torch.equal(model(x), torch.full_like(x, 8.0))
        controller.in_draft_phase = True
        assert torch.equal(model(x), torch.full_like(x, 2.0))
        # phase flips back → all layers run again
        controller.in_draft_phase = False
        assert torch.equal(model(x), torch.full_like(x, 8.0))

    def test_wrap_range_and_uninstall_restores(self):
        model, draft, controller = make_installed(n_layers=8, num_layers=2)
        wrapped = {id(layer) for layer, _ in draft._wrapped}
        for i, layer in enumerate(model.model.layers):
            assert (id(layer) in wrapped) == (i >= 2)
        controller.uninstall()
        assert draft._wrapped == []
        controller.in_draft_phase = True
        x = torch.zeros(1, 3)
        assert torch.equal(model(x), torch.full_like(x, 8.0))
        controller.uninstall()          # second uninstall is a no-op

    def test_validation(self):
        with pytest.raises(ValueError):
            SpeedDraft(num_layers=0)
        model = TinyModel(4)
        draft = SpeedDraft(num_layers=4)    # == layer count → nothing to skip
        controller = Controller(model, FakeAdapter(), draft)
        with pytest.raises(ValueError):
            controller.install()

    def test_requires_model_dot_layers(self):
        draft = SpeedDraft(num_layers=2)

        class NoLayers:
            model = object()
            in_draft_phase = False

        with pytest.raises(TypeError):
            draft.post_install(NoLayers())
