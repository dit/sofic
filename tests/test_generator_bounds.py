"""Support-type polytope vertices: exact redundancy pruning keeps the vertex set."""

from __future__ import annotations

import itertools
import time
from collections.abc import Iterator
from fractions import Fraction

import numpy as np
import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from sofic.examples.epsilon_machines import from_symbol_matrices
from sofic.generators import generator_bounds as gb
from sofic.generators.coupled_generators import _GeneratorProblem, _pair_joint
from sofic.generators.generative_search import _persistent_criteria


def _type_inequalities(joint: dict) -> tuple[list, int]:
    p = {c: Fraction(v).limit_denominator(10**6) for c, v in joint.items() if v > 0}
    total = sum(p.values())
    p = {c: v / total for c, v in p.items()}
    types = gb.maximal_rectangles(set(p))
    n = len(types)
    contains = [{c for c in p if c[0] in U and c[1] in V} for U, V in types]
    ineq = [([-Fraction(int(j == i)) for j in range(n)], Fraction(0)) for i in range(n)]
    for size in range(1, n + 1):
        for fam in itertools.combinations(range(n), size):
            union = set().union(*(contains[i] for i in fam))
            only = {c for c in union if all(i in fam for i in range(n) if c in contains[i])}
            row = [Fraction(int(j in fam)) for j in range(n)]
            ineq.append((row, sum((p[c] for c in union), Fraction(0))))
            ineq.append(([-x for x in row], -sum((p[c] for c in only), Fraction(0))))
    return ineq, n


def _brute_force_vertices(ineq: list, n: int) -> set:
    out = set()
    for subset in itertools.combinations(range(len(ineq)), n - 1):
        M = [[Fraction(1)] * n] + [ineq[i][0] for i in subset]
        r = [Fraction(1)] + [ineq[i][1] for i in subset]
        x = gb._solve(M, r)
        if x is not None and all(sum(a * xi for a, xi in zip(row, x, strict=True)) <= v for row, v in ineq):
            out.add(tuple(x))
    return out


@st.composite
def _sparse_joints(draw):
    cells = [(a, b) for a in range(3) for b in range(3)]
    mask = draw(st.lists(st.booleans(), min_size=9, max_size=9))
    live = [c for c, keep in zip(cells, mask, strict=True) if keep]
    if len(live) < 2:
        live = cells[:2]
    weights = [draw(st.integers(1, 9)) for _ in live]
    return {c: w / sum(weights) for c, w in zip(live, weights, strict=True)}


@settings(max_examples=40, deadline=None)
@given(_sparse_joints())
def test_pruned_vertices_equal_brute_force(joint) -> None:
    ineq, n = _type_inequalities(joint)
    if n > 4:
        return
    assert gb._polytope_vertices(ineq, n) == _brute_force_vertices(ineq, n)


def test_support_bound_is_fast_with_many_rectangles() -> None:
    # 4 x 4 staircase support with 6 maximal rectangles; brute force needs ~3e8 linear solves
    cells = [(a, b) for a in range(4) for b in range(4) if b >= a - 1 and b <= a + 1]
    rng = np.random.default_rng(0)
    joint = dict(zip(cells, rng.dirichlet(np.ones(len(cells))), strict=True))
    start = time.monotonic()
    result = gb.support_type_bound_result(joint)
    assert time.monotonic() - start < 60
    assert result.certified and 0 < result.value <= np.log2(len(cells))


def _set_partitions(items: list) -> Iterator[list[list]]:
    if not items:
        yield []
        return
    first, rest = items[0], items[1:]
    for part in _set_partitions(rest):
        for i in range(len(part)):
            yield [*part[:i], [first, *part[i]], *part[i + 1 :]]
        yield [[first], *part]


@st.composite
def _one_wide_joints(draw):
    cells = draw(st.sets(st.tuples(st.integers(0, 3), st.integers(0, 3)), min_size=1, max_size=7))
    support = np.zeros((4, 4), dtype=int)
    for c in cells:
        support[c] = 1
    overlap = support @ support.T
    np.fill_diagonal(overlap, 0)
    assume((overlap < 2).all())
    P = np.zeros((4, 4))
    for c in sorted(cells):
        P[c] = draw(st.integers(1, 9))
    return P / P.sum()


@settings(max_examples=60, deadline=None)
@given(_one_wide_joints())
def test_one_wide_partition_matches_brute_force(P) -> None:
    cells = [tuple(map(int, c)) for c in np.argwhere(P > 0)]
    exact: dict[int, float] = {}
    for part in _set_partitions(cells):
        if all(len({a for a, _ in block}) == 1 or len({b for _, b in block}) == 1 for block in part):
            masses = np.array([sum(P[c] for c in block) for block in part])
            exact[len(part)] = min(exact.get(len(part), np.inf), float(-(masses * np.log2(masses)).sum()))
    result = gb.one_wide_exact_common_information(P)
    assert result.value == pytest.approx(min(exact.values()), abs=1e-12)
    for k, value in result.by_terms.items():
        assert value == pytest.approx(min(v for j, v in exact.items() if j <= k), abs=1e-12)
    assert sorted(c for block in result.blocks for c in block) == sorted(cells)


def test_one_wide_partition_rejects_rectangles() -> None:
    with pytest.raises(ValueError, match="2 x 2 rectangle"):
        gb.one_wide_exact_common_information(np.full((2, 2), 0.25))


def _skip_one_chain():
    """Markov chain on Z_6 moving i -> i + 1 or i + 2; the cells (i, i + 1), (i, i + 2) in cycle order repeat
    heavy, heavy, light."""
    cells = [(j // 2, (j // 2 + 1 + j % 2) % 6) for j in range(12)]
    P = np.zeros((6, 6))
    for j, c in enumerate(cells):
        P[c] = 1e-3 if j % 3 == 2 else (1 - 4e-3) / 8
    T = P / P.sum(axis=1, keepdims=True)
    mats = {x: np.where(np.arange(6)[None, :] == x, T, 0.0) for x in range(6)}
    return from_symbol_matrices(tuple(range(6)), tuple(range(6)), mats)


def test_skip_one_chain_needs_more_than_min_plus_one_states() -> None:
    # Both persistent crypticities vanish, so C_g = G[S+ : S-], and the support (a 12-cycle) has no 2 x 2 rectangle.
    problem = _GeneratorProblem(_skip_one_chain().to_bidirectional())
    assert (problem.n_plus, problem.n_minus) == (6, 6)
    assert max(_persistent_criteria(problem)) < 1e-10
    result = gb.one_wide_exact_common_information(_pair_joint(problem))
    assert result.value == pytest.approx(2.0376224, abs=1e-6)
    assert len(result.blocks) == 8
    assert result.by_terms[7] == pytest.approx(2.2698054, abs=1e-6)
