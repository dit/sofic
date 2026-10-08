"""State splitting, amalgamation, and conjugacy invariants (Lind & Marcus §2.4, §7.4)."""

from __future__ import annotations

import math

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from sofic.shifts import (
    TopologicalMarkovChain,
    bowen_franks_group,
    in_amalgamate,
    in_edges,
    in_split,
    in_split_matrices,
    jordan_form_away_from_zero,
    out_amalgamate,
    out_edges,
    out_split,
    out_split_matrices,
)
from sofic.shifts.algorithms import integer_adjacency_matrix

SETTINGS = settings(max_examples=40, deadline=None, suppress_health_check=[HealthCheck.too_slow])


def adjacency(tmc):
    return np.array(integer_adjacency_matrix(tmc)[0], dtype=int)


def chain(rows):
    return TopologicalMarkovChain.from_adjacency(np.array(rows, dtype=int))


@st.composite
def split_cases(draw, *, outgoing=True):
    """A small TMC plus a random partition of the out- (or in-) edges of some states."""
    n = draw(st.integers(1, 3))
    rows = draw(st.lists(st.lists(st.integers(0, 2), min_size=n, max_size=n), min_size=n, max_size=n))
    tmc = chain(rows)
    edges_of = out_edges if outgoing else in_edges
    partitions = {}
    for state in tmc.states():
        edges = edges_of(tmc, state)
        if not edges or not draw(st.booleans()):
            continue
        labels = draw(st.lists(st.integers(0, len(edges) - 1), min_size=len(edges), max_size=len(edges)))
        parts = {}
        for edge, label in zip(edges, labels, strict=True):
            parts.setdefault(label, []).append(edge)
        partitions[state] = list(parts.values())
    return tmc, partitions


def invariants(tmc, *, symbolic):
    values = [tmc.periodic_points(n) for n in range(1, 7)]
    values += [round(tmc.topological_entropy(), 9), bowen_franks_group(tmc)]
    if symbolic:
        values += [tmc.zeta_function(), jordan_form_away_from_zero(tmc)]
    return values


# --- Known values ----------------------------------------------------------


def example_2_4_6():
    """Lind & Marcus Figure 2.4.5(a): a, b: I -> J; c: I -> K; d: J -> I; e: K -> J; f: K -> I."""
    tmc = chain([[0, 2, 1], [1, 0, 0], [1, 1, 0]])
    i, j, k = tmc.states()
    a, b = [e for e in out_edges(tmc, i) if e[1] == j]
    (c,) = [e for e in out_edges(tmc, i) if e[1] == k]
    e, f = sorted(out_edges(tmc, k), key=lambda edge: edge[1] != j)
    return tmc, {i: [[a], [b, c]], k: [[e], [f]]}


def test_division_and_edge_matrices_example_2_4_6():
    tmc, partitions = example_2_4_6()
    d, e = out_split_matrices(tmc, partitions)
    assert d.tolist() == [[1, 1, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 1, 1]]
    assert e.tolist() == [[0, 1, 0], [0, 1, 1], [1, 0, 0], [0, 1, 0], [1, 0, 0]]
    split = out_split(tmc, partitions)
    assert (d @ e == adjacency(tmc)).all()
    assert (e @ d == adjacency(split)).all()
    assert adjacency(split).tolist() == [
        [0, 0, 1, 0, 0],
        [0, 0, 1, 1, 1],
        [1, 1, 0, 0, 0],
        [0, 0, 1, 0, 0],
        [1, 1, 0, 0, 0],
    ]


def test_complete_out_splitting_of_full_two_shift_example_2_4_5():
    tmc = chain([[2]])
    (state,) = tmc.states()
    split = out_split(tmc, {state: [[edge] for edge in out_edges(tmc, state)]})
    assert adjacency(split).tolist() == [[1, 1], [1, 1]]


def test_split_copies_symbols():
    tmc = TopologicalMarkovChain(symbol_alphabet=frozenset("ab"))
    tmc.add_transition("I", "I", "a")
    tmc.add_transition("I", "I", "b")
    split = out_split(tmc, {"I": [[e] for e in out_edges(tmc, "I")]})
    symbols = sorted((t.source, t.target, t.data["symbol"]) for t in split.transitions())
    assert symbols == sorted((("I", s), ("I", d), sym) for s, sym in enumerate("ab") for d in range(2))


@pytest.mark.parametrize(
    ("rows", "factors"),
    [
        ([[2]], ()),
        ([[1, 1], [1, 0]], ()),
        ([[4, 1], [1, 0]], (4,)),
        ([[3, 2], [2, 1]], (2, 2)),
        ([[5, 3], [3, 5]], (7,)),
        ([[6, 8], [1, 4]], (7,)),
        ([[1]], (0,)),
        ([[1, 0], [0, 1]], (0, 0)),
    ],
)
def test_bowen_franks_known_values(rows, factors):
    # Lind & Marcus Example 7.4.16 and Exercise 7.4.12.
    assert bowen_franks_group(np.array(rows)).invariant_factors == factors


def test_bowen_franks_sign_and_inputs():
    assert bowen_franks_group(np.array([[2]])) == ((), -1)
    assert bowen_franks_group(np.array([[1]])).det_sign == 0
    assert bowen_franks_group(np.array([[0, 1], [1, 0]])) == ((0,), 0)
    assert bowen_franks_group(chain([[4, 1], [1, 0]])) == bowen_franks_group(np.array([[4, 1], [1, 0]]))
    with pytest.raises(ValueError):
        bowen_franks_group(np.array([[0.5]]))


def test_jordan_form_away_from_zero_known_values():
    sp = pytest.importorskip("sympy")
    # Lind & Marcus Examples 7.4.7 and 7.4.8.
    assert jordan_form_away_from_zero(np.array([[1, 1], [1, 1]])) == ((2, 1),)
    assert jordan_form_away_from_zero(np.array([[3, 1, 1], [2, 2, 1], [1, 2, 2]])) == ((1, 2), (5, 1))
    assert jordan_form_away_from_zero(np.array([[3, 1, 1], [2, 2, 1], [2, 1, 2]])) == ((1, 1), (1, 1), (5, 1))
    assert jordan_form_away_from_zero(np.array([[0, 1], [0, 0]])) == ()
    golden = jordan_form_away_from_zero(np.array([[1, 1], [1, 0]]))
    assert {root for root, _ in golden} == {(1 + sp.sqrt(5)) / 2, (1 - sp.sqrt(5)) / 2}


def test_invalid_partitions_and_amalgamations():
    tmc = chain([[1, 1], [1, 0]])
    i, j = tmc.states()
    edges = out_edges(tmc, i)
    with pytest.raises(ValueError, match="cover"):
        out_split(tmc, {i: [edges[:1]]})
    with pytest.raises(ValueError, match="empty"):
        out_split(tmc, {i: [edges, []]})
    with pytest.raises(ValueError, match="disjoint"):
        out_split(tmc, {i: [edges, edges[:1]]})
    with pytest.raises(ValueError, match="columns"):
        out_amalgamate(tmc, {"merged": [i, j]})
    with pytest.raises(ValueError, match="rows"):
        in_amalgamate(tmc, {"merged": [i, j]})


# --- Properties -------------------------------------------------------------


@SETTINGS
@given(split_cases(outgoing=True))
def test_out_split_matrices_and_round_trip(case):
    tmc, partitions = case
    split = out_split(tmc, partitions)
    d, e = out_split_matrices(tmc, partitions)
    assert ((d == 0) | (d == 1)).all() and (d.sum(axis=0) == 1).all()
    assert (d @ e == adjacency(tmc)).all()
    assert (e @ d == adjacency(split)).all()
    classes = {state: [(state, i) for i in range(len(parts))] for state, parts in partitions.items()}
    assert (adjacency(out_amalgamate(split, classes)) == adjacency(tmc)).all()


@SETTINGS
@given(split_cases(outgoing=False))
def test_in_split_matrices_and_round_trip(case):
    tmc, partitions = case
    split = in_split(tmc, partitions)
    d, e = in_split_matrices(tmc, partitions)
    assert (d @ e == adjacency(tmc)).all()
    assert (e @ d == adjacency(split)).all()
    classes = {state: [(state, i) for i in range(len(parts))] for state, parts in partitions.items()}
    assert (adjacency(in_amalgamate(split, classes)) == adjacency(tmc)).all()


@SETTINGS
@given(split_cases(outgoing=True))
def test_out_splitting_preserves_conjugacy_invariants(case):
    pytest.importorskip("sympy")
    tmc, partitions = case
    assert invariants(out_split(tmc, partitions), symbolic=True) == invariants(tmc, symbolic=True)


@SETTINGS
@given(split_cases(outgoing=False))
def test_in_splitting_preserves_conjugacy_invariants(case):
    pytest.importorskip("sympy")
    tmc, partitions = case
    assert invariants(in_split(tmc, partitions), symbolic=True) == invariants(tmc, symbolic=True)


@SETTINGS
@given(
    st.integers(1, 4).flatmap(
        lambda n: st.lists(st.lists(st.integers(-3, 3), min_size=n, max_size=n), min_size=n, max_size=n)
    )
)
def test_bowen_franks_matches_sympy_smith_form(rows):
    sp = pytest.importorskip("sympy")
    from sympy.matrices.normalforms import smith_normal_form

    difference = sp.eye(len(rows)) - sp.Matrix(rows)
    diagonal = [
        abs(int(difference_entry)) for difference_entry in smith_normal_form(difference, domain=sp.ZZ).diagonal()
    ]
    group = bowen_franks_group(np.array(rows))
    assert sorted(group.invariant_factors) == sorted(d for d in diagonal if d != 1)
    nonzero = [d for d in group.invariant_factors if d]
    assert all(b % a == 0 for a, b in zip(nonzero, nonzero[1:], strict=False))
    det = int(difference.det())
    assert group.det_sign == (det > 0) - (det < 0)
    if det:
        assert math.prod(group.invariant_factors) == abs(det)
