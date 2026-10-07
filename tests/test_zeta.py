"""Periodic points and zeta functions (Lind & Marcus §6.4)."""

from __future__ import annotations

import math
from itertools import product

import numpy as np
import pytest
from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st

from sofic.graph import ATTR_SYMBOL, TransitionGraph
from sofic.shifts import ShiftOfFiniteType, SoficShift, TopologicalMarkovChain, higher_block_presentation
from sofic.shifts.algorithms import reciprocal_characteristic_polynomial, signed_subset_matrices
from sofic.testing.strategies import sfts, sofic_shifts

sp = pytest.importorskip("sympy")
t = sp.Symbol("t")

SETTINGS = settings(max_examples=40, deadline=None, suppress_health_check=[HealthCheck.too_slow])


def brute_force_periodic_points(shift, n):
    """Count words ``w`` of length ``n`` with ``w^infinity`` in the shift.

    ``w^infinity`` is a bi-infinite path label iff some state returns to itself
    reading ``w^k`` for some ``1 <= k <= |V|`` (pigeonhole on the states at
    multiples of ``n``), which is weaker than a cycle labeled ``w`` itself.
    """
    outgoing = {state: [] for state in shift.states()}
    for transition in shift.transitions():
        outgoing[transition.source].append((transition.data[ATTR_SYMBOL], transition.target))
    states = list(outgoing)

    def read(current, word):
        for symbol in word:
            current = {target for state in current for label, target in outgoing[state] if label == symbol}
        return current

    count = 0
    for word in product(sorted(shift.symbol_alphabet), repeat=n):
        if any(_returns_within(read, state, word, len(states)) for state in states):
            count += 1
    return count


def _returns_within(read, state, word, max_repeats):
    """Whether reading ``word^k`` from ``state`` can return to it for some ``1 <= k <= max_repeats``."""
    current = {state}
    for _ in range(max_repeats):
        current = read(current, word)
        if state in current:
            return True
    return False


def series_from_zeta(zeta, order):
    """``p_n = n [t^n] log zeta`` for ``n = 1..order``."""
    series = sp.series(sp.log(zeta), t, 0, order + 1).removeO()
    return [int(n * series.coeff(t, n)) for n in range(1, order + 1)]


def block_recoding(shift, order):
    """The higher block shift ``X^[order+1]``, conjugate to ``shift``."""
    refined = higher_block_presentation(shift, order)
    graph = TransitionGraph()
    for state in refined.states():
        graph.add_state(state)
    labels = set()
    for transition in refined.transitions():
        label = transition.source[0] + (transition.data[ATTR_SYMBOL],)
        labels.add(label)
        graph.add_transition(transition.source, transition.target, **{ATTR_SYMBOL: label})
    return SoficShift(graph=graph, symbol_alphabet=frozenset(labels))


def as_sofic(shift):
    return SoficShift(graph=shift.graph.copy(), symbol_alphabet=shift.symbol_alphabet)


def even_shift():
    shift = SoficShift(symbol_alphabet=frozenset("ab"))
    shift.add_transition(0, 0, "a")
    shift.add_transition(0, 1, "b")
    shift.add_transition(1, 0, "b")
    return shift


def same_rational(f, g):
    return sp.cancel(f - g) == 0


tmc_matrices = st.integers(1, 3).flatmap(
    lambda n: st.lists(st.lists(st.integers(0, 2), min_size=n, max_size=n), min_size=n, max_size=n)
)


# --- Known values ----------------------------------------------------------


def test_full_two_shift():
    shift = TopologicalMarkovChain.from_adjacency(np.array([[2]]))
    assert [shift.periodic_points(n) for n in range(1, 8)] == [2**n for n in range(1, 8)]
    assert same_rational(shift.zeta_function(t), 1 / (1 - 2 * t))


def test_golden_mean_lucas_numbers():
    shift = TopologicalMarkovChain.from_adjacency(np.array([[1, 1], [1, 0]]))
    assert [shift.periodic_points(n) for n in range(1, 8)] == [1, 3, 4, 7, 11, 18, 29]
    assert same_rational(shift.zeta_function(t), 1 / (1 - t - t**2))


def test_even_shift_example_6_4_10():
    shift = even_shift()
    lucas = [1, 3, 4, 7, 11, 18, 29]
    assert [shift.periodic_points(n) for n in range(1, 8)] == [lucas[n - 1] - (-1) ** n for n in range(1, 8)]
    assert same_rational(shift.zeta_function(t), (1 + t) / (1 - t - t**2))
    matrices = signed_subset_matrices(shift)
    assert [(m.tolist(), sign) for m, sign in matrices] == [([[1, 1], [1, 0]], 1), ([[-1]], -1)]


def test_golden_mean_sft_from_forbidden_words():
    shift = ShiftOfFiniteType.from_forbidden_words({(1, 1)}, frozenset({0, 1}))
    assert [shift.periodic_points(n) for n in range(1, 8)] == [1, 3, 4, 7, 11, 18, 29]
    assert same_rational(shift.zeta_function(), 1 / (1 - t - t**2))


@pytest.mark.parametrize(
    "rows",
    [
        [["", "a", "b"], ["a", "", "b"], ["a", "b", ""]],
        [["a", "b", ""], ["", "", "b"], ["b", "", "a"]],
    ],
)
def test_exercise_6_4_2_matches_brute_force(rows):
    shift = SoficShift(symbol_alphabet=frozenset("ab"))
    for i, row in enumerate(rows):
        for j, label in enumerate(row):
            if label:
                shift.add_transition(i, j, label)
    zeta = shift.zeta_function(t)
    expected = [brute_force_periodic_points(shift, n) for n in range(1, 8)]
    assert [shift.periodic_points(n) for n in range(1, 8)] == expected
    assert series_from_zeta(zeta, 7) == expected


def test_nondeterministic_cycle_counts_once():
    # 0^infinity is presented only by the 2-cycle 0 -> 1 -> 0 and the self-loop at 2.
    shift = SoficShift(symbol_alphabet=frozenset("0"))
    shift.add_transition(0, 1, "0")
    shift.add_transition(1, 0, "0")
    shift.add_transition(2, 2, "0")
    assert [shift.periodic_points(n) for n in range(1, 5)] == [1, 1, 1, 1]
    assert same_rational(shift.zeta_function(t), 1 / (1 - t))


def test_empty_shift_and_bad_period():
    shift = SoficShift(symbol_alphabet=frozenset("a"))
    shift.add_transition(0, 1, "a")
    assert shift.periodic_points(3) == 0
    assert shift.zeta_function(t) == 1
    with pytest.raises(ValueError):
        shift.periodic_points(0)
    empty = ShiftOfFiniteType.from_forbidden_words({("a",)}, frozenset("a"))
    assert empty.periodic_points(2) == 0


def test_exact_for_large_periods():
    shift = TopologicalMarkovChain.from_adjacency(np.array([[3]]))
    assert shift.periodic_points(90) == 3**90


def test_reciprocal_characteristic_polynomial():
    a = np.array([[3, 1, 1], [2, 2, 1], [1, 2, 2]], dtype=object)
    # chi_A = (t - 5)(t - 1)^2  (Lind & Marcus Example 7.4.8)
    assert reciprocal_characteristic_polynomial(a) == [1, -7, 11, -5]


# --- Properties -------------------------------------------------------------


@SETTINGS
@given(sofic_shifts(max_states=3))
def test_sofic_matches_brute_force(shift):
    for n in range(1, 9):
        assert shift.periodic_points(n) == brute_force_periodic_points(shift, n)


@SETTINGS
@given(sofic_shifts(max_states=3))
def test_zeta_series_is_periodic_points(shift):
    zeta = shift.zeta_function(t)
    assert series_from_zeta(zeta, 6) == [shift.periodic_points(n) for n in range(1, 7)]


@SETTINGS
@given(sofic_shifts(max_states=3), st.integers(1, 2))
def test_higher_block_recoding_is_invariant(shift, order):
    recoded = block_recoding(shift, order)
    assert [recoded.periodic_points(n) for n in range(1, 7)] == [shift.periodic_points(n) for n in range(1, 7)]
    assert same_rational(recoded.zeta_function(t), shift.zeta_function(t))


@SETTINGS
@given(sfts(max_states=12))
def test_sft_block_matrix_agrees_with_manning(sft):
    presentation = as_sofic(sft)
    for n in range(1, 7):
        assert sft.periodic_points(n) == presentation.periodic_points(n)
    assert same_rational(sft.zeta_function(t), presentation.zeta_function(t))


@SETTINGS
@given(tmc_matrices)
def test_tmc_trace_and_zeta(rows):
    matrix = np.array(rows, dtype=int)
    shift = TopologicalMarkovChain.from_adjacency(matrix)
    for n in range(1, 6):
        assert shift.periodic_points(n) == int(np.trace(np.linalg.matrix_power(matrix, n)))
    assert same_rational(shift.zeta_function(t), 1 / sp.Matrix(sp.eye(len(rows)) - t * sp.Matrix(rows)).det())


@SETTINGS
@given(sofic_shifts(max_states=3))
def test_smallest_pole_gives_entropy(shift):
    _, denominator = sp.fraction(shift.zeta_function(t))
    assume(sp.degree(denominator, t) > 0)
    radius = min(abs(complex(root)) for root in sp.Poly(denominator, t).nroots())
    assert math.isclose(math.log2(1 / radius), shift.topological_entropy(), abs_tol=1e-6)
