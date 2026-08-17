"""Unit tests for clustering/hybrid.py (pure CPU, no model / GPU).

Covers the plan's §3.8 checklist:
  (a) alpha=0 + cooccur_norm=raw ranks identically to CooccurPairCluster;
  (b) alpha=1 ranks identically to ActivationSimCluster (same pair_sim);
  (c) a designed A-prefers-(0,1) / C-prefers-(0,2) table flips the chosen
      pair exactly once as alpha sweeps 0 → 1;
  (d) sentinel / missing-table / degenerate edge cases stay valid partitions;
plus direct norm01 properties, for both norm=minmax and norm=rank.
"""

import math

import pytest
import torch

from aug_spec.clustering import (
    ActivationSimCluster, CooccurPairCluster, HybridRelationCluster,
    get_cluster_method)
from aug_spec.clustering.base import ClusterContext
from aug_spec.clustering.hybrid import norm01

NEG_INF = float("-inf")
NORMS = ("minmax", "rank")


def make_ctx(n=4, active=None, pair_sim=None, cooccur=None):
    active = list(range(n)) if active is None else active
    weights = [0.0] * n
    for i in active:
        weights[i] = 1.0 / len(active)
    return ClusterContext(active=active, weights=weights, layer_idx=0,
                          cooccur=cooccur, pair_sim=pair_sim)


def l2_pair_sim(n, dists):
    """pair_sim as the draft builds it for metric=l2: NEGATIVE mean distance,
    -inf where the pair never co-fired. `dists` = {(i, j): distance}."""
    t = torch.full((n, n), NEG_INF)
    for (i, j), d in dists.items():
        t[i, j] = t[j, i] = -float(d)
    return t


def cooccur_table(n, counts, diag=None):
    """Symmetric co-occurrence counts; diagonal = per-expert totals (defaults
    to each expert's max pair count so cosine denominators are valid)."""
    t = torch.zeros(n, n)
    for (i, j), c in counts.items():
        t[i, j] = t[j, i] = float(c)
    for i in range(n):
        t[i, i] = float(diag[i]) if diag is not None else max(
            float(t[i].max()), 1.0)
    return t


def canon(groups):
    return sorted(tuple(sorted(g)) for g in groups)


def assert_valid_partition(groups, active, K):
    flat = [e for g in groups for e in g]
    assert sorted(flat) == sorted(active)          # cover all, no overlap
    assert all(1 <= len(g) <= 2 for g in groups)   # pairs or singletons
    if len(active) > K:
        assert len(groups) == K


# ── endpoint equivalence ─────────────────────────────────────────────────

@pytest.mark.parametrize("norm", NORMS)
def test_alpha0_raw_matches_cooccur_pair(norm):
    C = cooccur_table(6, {(0, 1): 9, (2, 3): 7, (0, 4): 3, (1, 5): 1},
                      diag=[20, 15, 9, 8, 5, 4])
    ctx = make_ctx(6, cooccur=C, pair_sim=l2_pair_sim(6, {(0, 5): 0.2}))
    hybrid = HybridRelationCluster(alpha=0.0, cooccur_norm="raw", norm=norm)
    for K in (2, 3, 4, 5):
        assert canon(hybrid.assign(ctx, K)) == \
            canon(CooccurPairCluster().assign(ctx, K))


@pytest.mark.parametrize("norm", NORMS)
def test_alpha1_matches_activation_sim(norm):
    A = l2_pair_sim(6, {(0, 1): 0.1, (2, 3): 0.5, (0, 4): 1.5, (1, 5): 2.0})
    ctx = make_ctx(6, pair_sim=A,
                   cooccur=cooccur_table(6, {(0, 5): 100, (1, 4): 90}))
    hybrid = HybridRelationCluster(alpha=1.0, metric="l2", norm=norm)
    for K in (2, 3, 4, 5):
        assert canon(hybrid.assign(ctx, K)) == \
            canon(ActivationSimCluster(metric="l2").assign(ctx, K))


# ── alpha sweep flips the winning pair exactly once ──────────────────────

@pytest.mark.parametrize("norm", NORMS)
def test_alpha_sweep_monotonic_flip(norm):
    # A prefers (0,1): closest outputs. C prefers (0,2): dominant co-count.
    A = l2_pair_sim(4, {(0, 1): 0.1, (0, 2): 2.0, (1, 2): 2.5, (2, 3): 3.0})
    C = cooccur_table(4, {(0, 2): 50, (0, 1): 1, (1, 3): 2, (2, 3): 1},
                      diag=[60, 10, 55, 8])
    ctx = make_ctx(4, pair_sim=A, cooccur=C)
    picks = []
    for alpha in [i / 20 for i in range(21)]:
        groups = HybridRelationCluster(alpha=alpha, norm=norm).assign(ctx, 3)
        assert_valid_partition(groups, ctx.active, 3)
        pair = next(tuple(sorted(g)) for g in groups if len(g) == 2)
        picks.append(pair)
    assert picks[0] == (0, 2)                      # alpha=0 → co-occur's pick
    assert picks[-1] == (0, 1)                     # alpha=1 → act-sim's pick
    flips = sum(1 for a, b in zip(picks, picks[1:]) if a != b)
    assert flips == 1                              # single crossover, no zigzag


# ── missing data / sentinels / degenerate shapes ─────────────────────────

@pytest.mark.parametrize("norm", NORMS)
@pytest.mark.parametrize("alpha", [0.0, 0.5, 1.0])
def test_edges_stay_valid_partitions(norm, alpha):
    m = HybridRelationCluster(alpha=alpha, norm=norm)
    # both tables missing → singleton fallback (may exceed K, like greedy_pair)
    groups = m.assign(make_ctx(4), K=2)
    assert canon(groups) == [(0,), (1,), (2,), (3,)]
    # active <= K → singletons
    ctx = make_ctx(4, active=[1, 3], pair_sim=l2_pair_sim(4, {(1, 3): 0.1}))
    assert canon(m.assign(ctx, K=2)) == [(1,), (3,)]
    # one table missing, the other partial: unobserved pairs only fill up to K
    A = l2_pair_sim(5, {(0, 1): 0.3})
    groups = m.assign(make_ctx(5, pair_sim=A), K=3)
    assert_valid_partition(groups, list(range(5)), 3)
    if alpha > 0.0:
        assert [0, 1] in [sorted(g) for g in groups]  # only observed pair wins
    # all observed values equal (min == max) → still a valid partition
    C = cooccur_table(4, {(0, 1): 5, (2, 3): 5, (0, 2): 5})
    groups = m.assign(make_ctx(4, cooccur=C), K=2)
    assert_valid_partition(groups, list(range(4)), 2)


def test_cosine_cooccur_removes_frequency_bias():
    # Raw counts favour the hot pair (0,1); cosine favours the exclusive pair
    # (2,3) whose members co-fire almost only with each other.
    C = cooccur_table(4, {(0, 1): 30, (2, 3): 8}, diag=[100, 90, 9, 9])
    ctx = make_ctx(4, cooccur=C)
    raw = HybridRelationCluster(alpha=0.0, cooccur_norm="raw")
    cos = HybridRelationCluster(alpha=0.0, cooccur_norm="cosine")
    assert (0, 1) in canon(raw.assign(ctx, 3))
    assert (2, 3) in canon(cos.assign(ctx, 3))


# ── norm01 unit behaviour ────────────────────────────────────────────────

def test_norm01_minmax_and_rank():
    v = torch.tensor([3.0, 1.0, 2.0, 99.0])
    mask = torch.tensor([True, True, True, False])
    mm = norm01(v, mask, "minmax")
    assert torch.allclose(mm, torch.tensor([1.0, 0.0, 0.5, -1.0]))
    rk = norm01(v, mask, "rank")
    assert torch.allclose(rk, torch.tensor([1.0, 0.0, 0.5, -1.0]))
    # rank is outlier-robust where minmax is not
    v2 = torch.tensor([1.0, 2.0, 1000.0])
    m2 = torch.ones(3, dtype=torch.bool)
    assert norm01(v2, m2, "minmax")[1] < 0.01
    assert math.isclose(float(norm01(v2, m2, "rank")[1]), 0.5)
    # ties share a value (dense rank), order preserved
    v3 = torch.tensor([5.0, 5.0, 7.0])
    rk3 = norm01(v3, torch.ones(3, dtype=torch.bool), "rank")
    assert rk3[0] == rk3[1] < rk3[2]
    # all-equal → 0.5; empty mask → all sentinel
    assert torch.allclose(
        norm01(torch.tensor([4.0, 4.0]), torch.ones(2, dtype=torch.bool),
               "minmax"), torch.tensor([0.5, 0.5]))
    assert torch.all(
        norm01(torch.tensor([1.0]), torch.zeros(1, dtype=torch.bool),
               "rank") == -1.0)


# ── registry / constructor validation ────────────────────────────────────

def test_registry_and_validation():
    m = get_cluster_method("hybrid", alpha=0.25, metric="l2", norm="rank",
                           cooccur_norm="raw", cooccur_scope="all")
    assert isinstance(m, HybridRelationCluster)
    assert m.needs_cooccur and m.needs_activation_sim
    assert m.act_sim_prefill_only
    d = HybridRelationCluster()
    assert (d.alpha, d.metric, d.norm, d.cooccur_norm, d.cooccur_scope) == \
        (0.5, "l2", "rank", "cosine", "decode")
    for bad in (dict(alpha=1.5), dict(metric="dot"), dict(norm="zscore"),
                dict(cooccur_norm="lift"), dict(cooccur_scope="prefill")):
        with pytest.raises(ValueError):
            HybridRelationCluster(**bad)
