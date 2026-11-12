"""Support-type polytope vertices: exact redundancy pruning keeps the vertex set."""

from __future__ import annotations

import itertools
import time
from fractions import Fraction

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

from sofic.generators import generator_bounds as gb


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
