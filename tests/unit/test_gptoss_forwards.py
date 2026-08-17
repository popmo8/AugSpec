"""GPT-OSS masked / substitute forwards (gptoss_acceptance_plan.md Steps 1-2).

Routing semantics under test: gpt-oss is softmax-AFTER-topk (winners = top-k
logits, weights = softmax over exactly those logits) — the opposite order to
qwen3. Fakes record what reaches `mlp.experts`; on the real GPU inference
path only the dense routing_weights matters (router_indices is ignored), so
the assertions target the scattered scores.
"""

import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from aug_spec.adapters.gptoss import GptOssAdapter
from aug_spec.drafts.specmoe import SpecMoeDraft, pairwise_l2

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

from collect_calibration import expert_output_fn, gptoss_expert_outputs  # noqa: E402


class FakeRouter:
    def __init__(self, n, top_k):
        self.weight = torch.eye(n)          # logits == hidden features
        self.bias = torch.zeros(n)
        self.top_k = top_k


class FakeExperts:
    """Records the routing that reaches it; output = input (identity)."""

    def __init__(self):
        self.calls = []

    def __call__(self, hidden_states, router_indices=None,
                 routing_weights=None):
        self.calls.append((router_indices, routing_weights))
        return hidden_states


class FakeMlp:
    def __init__(self, n=4, top_k=2):
        self.router = FakeRouter(n, top_k)
        self.experts = FakeExperts()


class StubDraft:
    route_top_k = 2

    def __init__(self):
        self.captured = []

    def capture(self, layer_idx, softmax):
        self.captured.append((layer_idx, softmax))


class StubController:
    def __init__(self, in_draft):
        self.draft = StubDraft()
        self.in_draft_phase = in_draft
        self.draft_cache = {}


HIDDEN = torch.tensor([[[10.0, 9.0, 1.0, 0.0]]])   # logits = [10, 9, 1, 0]


def run_masked(in_draft, mask=None):
    controller = StubController(in_draft)
    if mask is not None:
        controller.draft_cache[0] = mask
    mlp = FakeMlp()
    out, scores = GptOssAdapter().make_masked_forward(
        controller, 0, mlp)(mlp, HIDDEN)
    assert torch.equal(out, HIDDEN)                 # identity experts
    return mlp.experts.calls[-1], scores


def test_masked_draft_routes_within_kept():
    mask = torch.tensor([True, False, True, False])
    (indices, weights), _ = run_masked(True, mask)
    assert set(indices.flatten().tolist()) == {0, 2}   # natural top-2 was {0, 1}
    expected = F.softmax(torch.tensor([10.0, 1.0]), dim=0)
    assert weights[0, 0] == pytest.approx(expected[0].item(), rel=1e-5)
    assert weights[0, 2] == pytest.approx(expected[1].item(), rel=1e-5)
    assert weights[0, [1, 3]].sum() == 0


def test_masked_draft_without_mask_and_target_are_natural():
    for in_draft in (True, False):
        (indices, weights), _ = run_masked(in_draft, mask=None)
        assert set(indices.flatten().tolist()) == {0, 1}
        expected = F.softmax(torch.tensor([10.0, 9.0]), dim=0)
        assert weights[0, 0] == pytest.approx(expected[0].item(), rel=1e-5)
        assert weights[0, 1] == pytest.approx(expected[1].item(), rel=1e-5)


def run_substitute(in_draft, table=None):
    controller = StubController(in_draft)
    if table is not None:
        controller.draft_cache[0] = table
    mlp = FakeMlp()
    hidden = torch.tensor([[[1.0, 2.0, 3.0, 4.0]]])    # natural top-2 = {3, 2}
    out, scores = GptOssAdapter().make_substitute_forward(
        controller, 0, mlp)(mlp, hidden)
    assert torch.equal(out, hidden)
    return controller, mlp.experts.calls[-1], scores


def test_substitute_draft_remaps_and_accumulates_collisions():
    table = torch.zeros(4, dtype=torch.long)           # everything → expert 0
    _, (indices, weights), _ = run_substitute(True, table)
    assert indices.flatten().tolist() == [0, 0]
    # Both winners' softmax-after-topk weights pile onto expert 0.
    assert weights[0, 0] == pytest.approx(1.0, rel=1e-5)
    assert weights[0, 1:].sum() == 0


def test_substitute_target_captures_full_softmax():
    controller, (indices, weights), _ = run_substitute(False)
    assert set(indices.flatten().tolist()) == {3, 2}   # no remap in target
    expected = F.softmax(torch.tensor([4.0, 3.0]), dim=0)
    assert weights[0, 3] == pytest.approx(expected[0].item(), rel=1e-5)
    (li, sm), = controller.draft.captured
    assert li == 0 and sm.dtype == torch.float32
    assert torch.allclose(
        sm, F.softmax(torch.tensor([[1.0, 2.0, 3.0, 4.0]]), dim=1))


class FakeFusedExperts:
    """Fused tensors sized so expert 1 is a near-copy of expert 0 and expert 2
    is far from both (drives the L2 substitute table)."""

    def __init__(self, n=3, d=2, inner=2):
        self.gate_up_proj = torch.zeros(n, d, 2 * inner)
        self.gate_up_proj_bias = torch.zeros(n, 2 * inner)
        self.down_proj = torch.zeros(n, inner, d)
        self.down_proj_bias = torch.zeros(n, d)
        self.gate_up_proj[1] += 0.1
        self.gate_up_proj[2] += 100.0


class FakeBlock:
    def __init__(self):
        self.experts = FakeFusedExperts()


def test_expert_flat_weights_and_substitute_table():
    adapter = GptOssAdapter()
    block = FakeBlock()
    flats = adapter.expert_flat_weights(block)
    e = block.experts
    per_expert = (e.gate_up_proj[0].numel() + e.gate_up_proj_bias[0].numel()
                  + e.down_proj[0].numel() + e.down_proj_bias[0].numel())
    assert len(flats) == 3 and all(f.numel() == per_expert for f in flats)
    D = pairwise_l2(flats)
    assert torch.equal(D, D.t()) and D.diagonal().sum() == 0

    # prepare → capture → refresh roundtrip: kept {0, 2}, so the near-copy
    # expert 1 substitutes to 0 and the kept experts map to themselves.
    draft = SpecMoeDraft(N=2, route_top_k=2)
    draft.prepare(adapter, [(0, block)])
    assert draft.num_experts == 3
    softmax = torch.tensor([[0.5, 0.1, 0.4], [0.5, 0.1, 0.4]])
    draft.capture(0, softmax)                          # votes: {0: 2, 2: 2}
    cache = {}
    draft.refresh(adapter, [(0, block)], cache)
    assert cache[0].tolist() == [0, 0, 2]


class RandnFusedExperts:
    """Randomised fused tensors for the calibration-kernel tests (G3)."""

    alpha = 1.702
    limit = 7.0

    def __init__(self, n=3, d=4, inner=2, seed=1):
        g = torch.Generator().manual_seed(seed)
        self.gate_up_proj = torch.randn(n, d, 2 * inner, generator=g)
        self.gate_up_proj_bias = torch.randn(n, 2 * inner, generator=g)
        self.down_proj = torch.randn(n, inner, d, generator=g)
        self.down_proj_bias = torch.randn(n, d, generator=g)


def test_calibration_expert_outputs_match_adapter_dense():
    # The fused calibration kernel must agree with _run_dense_expert on
    # per-expert slices — the path the merged draft already exercises.
    experts = RandnFusedExperts()
    adapter = GptOssAdapter()
    adapter._alpha, adapter._limit = experts.alpha, experts.limit
    x = torch.randn(5, 4)
    outs = gptoss_expert_outputs(experts, x, 0, 3)
    assert outs.shape == (3, 5, 4)
    for e in range(3):
        slices = {
            "gate_up_proj": experts.gate_up_proj[e],
            "gate_up_proj_bias": experts.gate_up_proj_bias[e],
            "down_proj": experts.down_proj[e],
            "down_proj_bias": experts.down_proj_bias[e],
        }
        assert torch.allclose(outs[e], adapter._run_dense_expert(slices, x),
                              atol=1e-5)


def test_expert_output_fn_gptoss_chunking():
    experts = RandnFusedExperts()
    block = type("B", (), {"experts": experts})()
    fn, n = expert_output_fn(GptOssAdapter(), block)
    assert n == 3
    x = torch.randn(4, 4)
    assert torch.allclose(fn(x, 0, 3),
                          torch.cat([fn(x, 0, 2), fn(x, 2, 3)]), atol=1e-6)
