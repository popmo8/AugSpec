"""hf-backend act-sim capture (gptoss_acceptance_plan.md Step 0.5).

Covers:
  (a) the extracted _accumulate_act_sim numerics against a hand computation
      (l2 and cosine), including the unvisited-pair sentinel;
  (b) the wants_prefill_act_sim gate: only prefill-only act-sim methods
      qualify, a layer freezes after capture, reset() re-arms it;
  (c) refactor equivalence: the offload entry (fake dispatcher) produces the
      same tables as the hf entry and never touches the hf freeze set;
  (d) GptOssAdapter._fired_expert_outputs: per-expert token grouping, raw
      (pre-routing-weight) outputs, CPU placement, engine tuple format.
"""

import pytest
import torch

from aug_spec.adapters.gptoss import GptOssAdapter
from aug_spec.clustering import get_cluster_method
from aug_spec.clustering.activation_sim import ActivationSimCluster
from aug_spec.clustering.hybrid import HybridRelationCluster
from aug_spec.drafts.topm_count import TopMCountDraft


def make_draft(method=None):
    draft = TopMCountDraft(count_top_k=4, M=8, K=4, draft_top_k=4)
    if method is not None:
        draft.cluster_method = method
    return draft


def hybrid_l2():
    return HybridRelationCluster(alpha=0.75, metric="l2", norm="rank",
                                 cooccur_norm="raw", cooccur_scope="decode")


# Two tokens, three experts: token0 co-fires {0, 1}, token1 co-fires {0, 2}.
# L2(e0, e1 | token0) = |(0,0)-(3,4)| = 5, L2(e0, e2 | token1) = |(1,0)-(1,2)| = 2.
CAPTURED = [
    (0, 0, torch.tensor([0, 1]), torch.tensor([[0.0, 0.0], [1.0, 0.0]])),
    (0, 1, torch.tensor([0]), torch.tensor([[3.0, 4.0]])),
    (0, 2, torch.tensor([1]), torch.tensor([[1.0, 2.0]])),
]


def test_accumulate_numeric_l2():
    draft = make_draft(hybrid_l2())
    draft.accumulate_prefill_act_sim(0, CAPTURED, n=3)
    sim = draft._pair_sim_table(0)
    assert sim[0, 1] == pytest.approx(-5.0)
    assert sim[0, 2] == pytest.approx(-2.0)
    assert sim[1, 0] == pytest.approx(-5.0)          # symmetric
    assert sim[1, 2] == float("-inf")                # never co-fired
    assert draft.act_sim_cnt[0][0, 1] == 1.0


def test_accumulate_numeric_cosine():
    # Orthogonal outputs on the one co-firing token → mean cosine 0.
    draft = make_draft(ActivationSimCluster(metric="cosine"))
    captured = [
        (0, 0, torch.tensor([0]), torch.tensor([[1.0, 0.0]])),
        (0, 1, torch.tensor([0]), torch.tensor([[0.0, 1.0]])),
    ]
    draft._accumulate_act_sim(0, captured, n=2)
    sim = draft._pair_sim_table(0)
    assert sim[0, 1] == pytest.approx(0.0)


def test_wants_gate_and_freeze():
    draft = make_draft(hybrid_l2())
    assert draft.wants_prefill_act_sim(0)
    draft.accumulate_prefill_act_sim(0, CAPTURED, n=3)
    assert not draft.wants_prefill_act_sim(0)        # frozen for the question
    assert draft.wants_prefill_act_sim(1)            # other layers unaffected
    draft.reset()
    assert draft.wants_prefill_act_sim(0)            # re-armed per question

    # Non-act-sim and non-prefill-only methods never qualify.
    assert not make_draft(get_cluster_method("freq_slice")
                          ).wants_prefill_act_sim(0)
    assert not make_draft(ActivationSimCluster()).wants_prefill_act_sim(0)


def test_offload_entry_matches_and_keeps_gate_untouched():
    class FakeDispatcher:
        def get_captured_expert_outputs(self):
            return CAPTURED

    via_engine = make_draft(hybrid_l2())
    via_engine.accumulate_activation_sim(0, FakeDispatcher(), n=3)
    via_hf = make_draft(hybrid_l2())
    via_hf.accumulate_prefill_act_sim(0, CAPTURED, n=3)

    assert torch.equal(via_engine.act_sim_num[0], via_hf.act_sim_num[0])
    assert torch.equal(via_engine.act_sim_cnt[0], via_hf.act_sim_cnt[0])
    # The engine path must not consume the hf prefill gate.
    assert via_engine.wants_prefill_act_sim(0)


class FakeExperts:
    """Fused gpt-oss expert tensors: gate_up [n, D, 2I], down [n, I, D]."""

    def __init__(self, n=3, d=4, inner=2, seed=0):
        g = torch.Generator().manual_seed(seed)
        self.gate_up_proj = torch.randn(n, d, 2 * inner, generator=g)
        self.gate_up_proj_bias = torch.randn(n, 2 * inner, generator=g)
        self.down_proj = torch.randn(n, inner, d, generator=g)
        self.down_proj_bias = torch.randn(n, d, generator=g)


def ref_expert_out(x, experts, e, alpha, limit):
    gu = x @ experts.gate_up_proj[e] + experts.gate_up_proj_bias[e]
    gate = gu[..., ::2].clamp(max=limit)
    up = gu[..., 1::2].clamp(min=-limit, max=limit)
    glu = gate * torch.sigmoid(gate * alpha)
    return ((up + 1) * glu) @ experts.down_proj[e] + experts.down_proj_bias[e]


def test_gptoss_fired_expert_outputs():
    adapter = GptOssAdapter()
    adapter._alpha, adapter._limit = 1.702, 7.0
    experts = FakeExperts()
    flat = torch.randn(3, 4)
    router_indices = torch.tensor([[0, 1], [0, 2], [1, 2]])

    captured = {e: (tok, out) for _, e, tok, out
                in adapter._fired_expert_outputs(5, experts, flat,
                                                 router_indices)}
    assert set(captured) == {0, 1, 2}
    assert captured[0][0].tolist() == [0, 1]         # expert 0 ← tokens 0, 1
    assert captured[1][0].tolist() == [0, 2]
    assert captured[2][0].tolist() == [1, 2]
    for e, (tok, out) in captured.items():
        assert out.device.type == "cpu" and tok.device.type == "cpu"
        ref = ref_expert_out(flat[tok], experts, e, 1.702, 7.0)
        assert torch.allclose(out, ref, atol=1e-6)
    # Tuple format carries the caller's layer index.
    assert all(li == 5 for li, _, _, _
               in adapter._fired_expert_outputs(5, experts, flat,
                                                router_indices))
