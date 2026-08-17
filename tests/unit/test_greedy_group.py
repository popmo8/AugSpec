"""greedy_group (capacity-generalised pairing, M>2K support, 2026-07-25)."""
import random

import pytest

from aug_spec.clustering.base import greedy_group, greedy_pair


def _rand_table(n, seed):
    rng = random.Random(seed)
    t = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            t[i][j] = t[j][i] = rng.random()
    return t


@pytest.mark.parametrize("seed", range(5))
def test_cap2_identical_to_greedy_pair(seed):
    n = 32
    t = _rand_table(n, seed)
    active = list(range(n))
    assert greedy_group(t, active, 16, cap=2) == greedy_pair(t, active, 16)


@pytest.mark.parametrize("m,k", [(48, 16), (64, 16), (80, 16), (96, 16)])
def test_cap_partition_properties(m, k):
    cap = -(-m // k)
    t = _rand_table(m, seed=m)
    active = list(range(m))
    groups = greedy_group(t, active, k, cap=cap)
    assert len(groups) <= k
    assert all(len(g) <= cap for g in groups)
    flat = sorted(x for g in groups for x in g)
    assert flat == active                      # 不重不漏


def test_stall_then_repair():
    # cap=3, K=2, active=6;表設計成先配成 3 對(2+2>3 全被 cap 擋)→ 修復路徑
    t = [[0.0] * 6 for _ in range(6)]
    for a, b in ((0, 1), (2, 3), (4, 5)):
        t[a][b] = t[b][a] = 1.0
    groups = greedy_group(t, list(range(6)), 2, cap=3)
    assert len(groups) == 2
    assert sorted(len(g) for g in groups) == [3, 3]


def test_few_active_all_singletons():
    t = _rand_table(8, 0)
    assert greedy_group(t, [1, 5], 16, cap=4) == [[1], [5]]
