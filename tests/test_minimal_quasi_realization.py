"""Tests for Schützenberger/Fliess minimal quasi-realizations and the process rank."""

from functools import cache
from typing import Any

import numpy as np
import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from sofic.examples import even_process, golden_mean, iid
from sofic.exceptions import MixedStateExplosionError
from sofic.generators.epsilon_machine import EpsilonMachine
from sofic.generators.mealy import MealyHMM
from sofic.generators.minimal_quasi_realization import minimal_quasi_realization, process_rank
from sofic.generators.quasi_realization import QuasiRealization
from sofic.graph import ATTR_EMISSION, ATTR_PROB
from sofic.testing.strategies import epsilon_machines, mealy_hmms
from tests import oracles

ALPHABET = (0, 1)


def _duplicate_state(hmm: MealyHMM, state: int, weight: float) -> MealyHMM:
    """Split ``state`` into itself and a clone, sending ``weight`` of its inflow to the original."""
    clone = max(hmm.states()) + 1
    out = MealyHMM(observation_alphabet=hmm.observation_alphabet)
    for s in [*hmm.states(), clone]:
        out.graph.add_state(s)
    for transition in hmm.transitions():
        symbol, prob = transition.data[ATTR_EMISSION], transition.data[ATTR_PROB]
        sources = [transition.source, clone] if transition.source == state else [transition.source]
        for source in sources:
            if transition.target == state:
                out.add_transition(source, state, symbol, weight * prob)
                out.add_transition(source, clone, symbol, (1 - weight) * prob)
            else:
                out.add_transition(source, transition.target, symbol, prob)
    initial = dict(hmm.initial_distribution)
    mass = initial.pop(state, 0.0)
    out.initial_distribution = {**initial, state: weight * mass, clone: (1 - weight) * mass}
    return out


def _words(max_length: int) -> list[tuple[int, ...]]:
    return oracles.words(ALPHABET, max_length)


def _hankel_rank(hmm: MealyHMM, length: int) -> int:
    probability = cache(lambda word: oracles.word_probability(hmm, word))
    prefixes = _words(length)
    hankel = np.array([[probability(u + v) for v in prefixes] for u in prefixes])
    singular = np.linalg.svd(hankel, compute_uv=False)
    return int(np.sum(singular > 1e-9 * singular[0]))


@settings(max_examples=60, deadline=None)
@given(mealy_hmms(alphabet=ALPHABET, max_states=3))
def test_reproduces_every_word_probability(hmm):
    realization = minimal_quasi_realization(hmm)
    for word in _words(5):
        assert realization.word_probability(word) == pytest.approx(oracles.word_probability(hmm, word), abs=1e-9)


@settings(max_examples=60, deadline=None)
@given(mealy_hmms(alphabet=ALPHABET, max_states=3))
def test_dimension_equals_brute_force_hankel_rank(hmm):
    n = len(list(hmm.states()))
    assert process_rank(hmm) == _hankel_rank(hmm, n)


@settings(max_examples=40, deadline=None)
@given(mealy_hmms(alphabet=ALPHABET, max_states=3), st.data())
def test_dimension_bounded_by_every_presentation(hmm, data):
    rank = process_rank(hmm)
    assert 1 <= rank <= len(list(hmm.states()))

    state = data.draw(st.sampled_from(sorted(hmm.states())))
    weight = data.draw(st.sampled_from([0.25, 0.5, 0.75]))
    redundant = _duplicate_state(hmm, state, weight)
    assert process_rank(redundant) == rank


@settings(max_examples=40, deadline=None)
@given(mealy_hmms(alphabet=ALPHABET, max_states=3))
def test_stationary_dimension_bounded_by_epsilon_machine_of_hmm(hmm):
    assume(hmm.is_irreducible())
    try:
        machine = EpsilonMachine.from_hmm(hmm, max_states=200)
    except MixedStateExplosionError:
        assume(False)
    stationary = hmm.copy()
    stationary.initial_distribution = dict(zip(hmm.reindex().states, hmm.stationary_distribution(), strict=True))
    assert process_rank(stationary) <= len(list(machine.states()))


@settings(max_examples=40, deadline=None)
@given(epsilon_machines(alphabet=ALPHABET, max_states=4))
def test_dimension_bounded_by_epsilon_machine(machine):
    assert process_rank(machine) <= len(list(machine.states()))


@settings(max_examples=40, deadline=None)
@given(mealy_hmms(alphabet=ALPHABET, max_states=3))
def test_idempotent(hmm):
    once = minimal_quasi_realization(hmm)
    twice = minimal_quasi_realization(once)
    assert twice.pi.size == once.pi.size
    for word in _words(4):
        assert twice.word_probability(word) == pytest.approx(once.word_probability(word), abs=1e-9)


def test_iid_has_rank_one():
    assert process_rank(iid(3)) == 1


def test_golden_mean_has_rank_two():
    assert golden_mean(0.5).process_rank() == 2


def test_even_process_has_rank_two():
    assert process_rank(even_process()) == 2


def test_redundant_states_reduce():
    hmm = MealyHMM(observation_alphabet=frozenset(ALPHABET), initial_distribution={0: 0.5, 1: 0.25, 2: 0.25})
    hmm.add_transition(0, 1, 0, 0.25)
    hmm.add_transition(0, 2, 0, 0.25)
    hmm.add_transition(0, 0, 1, 0.5)
    for state in (1, 2):
        hmm.add_transition(state, 0, 1, 1.0)
    assert len(list(hmm.states())) == 3
    realization = minimal_quasi_realization(hmm)
    assert realization.pi.size == 2
    assert realization.pi.sum() == pytest.approx(1.0)
    np.testing.assert_allclose(realization.tau, np.ones(2))


def test_returns_quasi_realization_with_unit_final_vector():
    realization = minimal_quasi_realization(golden_mean(0.5))
    assert isinstance(realization, QuasiRealization)
    realization.validate_quasistochastic()
    np.testing.assert_allclose(realization.tau, np.ones(2))


def test_exact_rational_input_stays_exact():
    sp = pytest.importorskip("sympy")
    third = sp.Rational(1, 3)
    hmm = MealyHMM(initial_distribution={0: sp.Rational(3, 4), 1: sp.Rational(1, 4)})
    hmm.add_transition(0, 0, 0, third)
    hmm.add_transition(0, 1, 1, 1 - third)
    hmm.add_transition(1, 0, 0, sp.Integer(1))
    realization = minimal_quasi_realization(hmm)
    assert realization.pi.dtype == object
    assert realization.pi.size == 2
    for word in _words(4):
        result = realization.pi
        for symbol in word:
            result = result @ realization.symbol_maps[symbol]
        assert sp.simplify(result @ realization.tau - _exact_word_probability(hmm, word)) == 0


def test_symbolic_parameter():
    sp = pytest.importorskip("sympy")
    a = sp.Symbol("a", positive=True)
    hmm = MealyHMM(initial_distribution={0: sp.Integer(1)})
    hmm.add_transition(0, 0, 0, a)
    hmm.add_transition(0, 1, 1, 1 - a)
    hmm.add_transition(1, 0, 0, sp.Integer(1))
    realization = minimal_quasi_realization(hmm)
    assert realization.pi.size == 2
    word = (0, 1, 0)
    result = realization.pi
    for symbol in word:
        result = result @ realization.symbol_maps[symbol]
    assert sp.simplify(result @ realization.tau - a * (1 - a)) == 0


def _exact_word_probability(hmm: MealyHMM, word: tuple[int, ...]) -> Any:
    sp = pytest.importorskip("sympy")
    states = sorted(hmm.states())
    vector = {s: sp.sympify(hmm.initial_distribution.get(s, 0)) for s in states}
    for symbol in word:
        nxt = dict.fromkeys(states, sp.Integer(0))
        for transition in hmm.transitions():
            if transition.data[ATTR_EMISSION] == symbol:
                nxt[transition.target] += vector[transition.source] * transition.data[ATTR_PROB]
        vector = nxt
    return sp.simplify(sum(vector.values()))
