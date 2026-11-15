"""Probabilities of regular and deterministic-Büchi properties of HMM output."""

from __future__ import annotations

from itertools import product

import numpy as np
import pytest

from sofic.automata.buchi import BuchiAutomaton
from sofic.automata.dfa import DFA
from sofic.examples import bernoulli, binary_markov_chain, even_process, golden_mean
from sofic.exceptions import NonDeterministicError
from sofic.generators.mealy import MealyHMM
from sofic.generators.omega_probability import omega_probability, regular_language_probability


def _automaton(cls, edges, accepting: str, initial: str = "a"):
    aut = cls(
        input_alphabet=frozenset("01"), initial_states=frozenset({initial}), accepting_states=frozenset(accepting)
    )
    aut.graph.add_state(initial)
    for source, target, _symbol in edges:
        aut.graph.add_state(source)
        aut.graph.add_state(target)
    for source, target, symbol in edges:
        aut.add_transition(source, target, symbol)
    return aut


# State b is entered exactly on reading a 1.
INFINITELY_MANY_ONES = [("a", "a", "0"), ("a", "b", "1"), ("b", "a", "0"), ("b", "b", "1")]
# b is absorbing once a 1 (resp. 11) has been read; a, c never saw one.
EVENTUALLY_ONE = [("a", "a", "0"), ("a", "b", "1"), ("b", "b", "0"), ("b", "b", "1")]
ALWAYS_ZERO = [("a", "a", "0")]
EVENTUALLY_ONE_ONE = [
    ("a", "a", "0"),
    ("a", "c", "1"),
    ("c", "a", "0"),
    ("c", "b", "1"),
    ("b", "b", "0"),
    ("b", "b", "1"),
]
NEVER_ONE_ONE = [("a", "a", "0"), ("a", "c", "1"), ("c", "a", "0")]


def _gf1() -> BuchiAutomaton:
    return _automaton(BuchiAutomaton, INFINITELY_MANY_ONES, "b")


def _split(mass_to_coin: float = 0.3) -> MealyHMM:
    """From ``S`` emit ``0`` and move to a fair coin ``B`` or to the all-zeros state ``Z``."""
    hmm = MealyHMM(observation_alphabet=frozenset("01"), initial_distribution={"S": 1.0})
    hmm.add_transition("S", "B", "0", mass_to_coin)
    hmm.add_transition("S", "Z", "0", 1 - mass_to_coin)
    hmm.add_transition("B", "B", "0", 0.5)
    hmm.add_transition("B", "B", "1", 0.5)
    hmm.add_transition("Z", "Z", "0", 1.0)
    return hmm


def _stops_emitting_ones(stop: float = 0.25) -> MealyHMM:
    """A fair coin that switches to emitting ``0`` forever with probability ``stop`` per step."""
    hmm = MealyHMM(observation_alphabet=frozenset("01"), initial_distribution={"C": 1.0})
    hmm.add_transition("C", "C", "0", 0.5 * (1 - stop))
    hmm.add_transition("C", "C", "1", 0.5 * (1 - stop))
    hmm.add_transition("C", "Z", "0", stop)
    hmm.add_transition("Z", "Z", "0", 1.0)
    return hmm


@pytest.mark.parametrize("p", [0.01, 0.3, 0.5, 0.99])
def test_infinitely_many_ones_is_almost_sure_for_iid(p):
    assert omega_probability(bernoulli(p), _gf1()) == pytest.approx(1.0)


@pytest.mark.parametrize("model", [golden_mean(0.4), even_process(0.3), binary_markov_chain(0.2, 0.7)])
def test_infinitely_many_ones_is_almost_sure_for_ergodic_processes(model):
    assert omega_probability(model, _gf1()) == pytest.approx(1.0)


def test_infinitely_many_ones_fails_once_ones_stop():
    model = _stops_emitting_ones()
    assert omega_probability(model, _gf1(), start="C") == pytest.approx(0.0)
    assert omega_probability(model, _gf1(), start={"C": 1.0}) == pytest.approx(0.0)
    # 'Eventually always 0' (finitely many 1s) is not DBA-expressible; it is the complement.
    assert 1 - omega_probability(_split(0.3), _gf1(), start="S") == pytest.approx(0.7)
    assert omega_probability(_split(0.3), _gf1(), start="B") == pytest.approx(1.0)


def test_eventually_one_is_the_mass_that_reaches_the_coin():
    eventually = _automaton(BuchiAutomaton, EVENTUALLY_ONE, "b")
    assert omega_probability(bernoulli(0.5), eventually) == pytest.approx(1.0)
    assert omega_probability(_split(0.0001), eventually, start="S") == pytest.approx(0.0001)
    assert omega_probability(_split(0.3), eventually, start="Z") == pytest.approx(0.0)


@pytest.mark.parametrize(
    ("prop", "complement"),
    [
        ((EVENTUALLY_ONE, "b"), (ALWAYS_ZERO, "a")),
        ((EVENTUALLY_ONE_ONE, "b"), (NEVER_ONE_ONE, "ac")),
    ],
)
@pytest.mark.parametrize("mass", [0.0, 0.3, 0.8])
def test_dba_and_complement_dba_sum_to_one(prop, complement, mass):
    model = _split(mass)
    p = omega_probability(model, _automaton(BuchiAutomaton, *prop), start="S")
    q = omega_probability(model, _automaton(BuchiAutomaton, *complement), start="S")
    assert p + q == pytest.approx(1.0)
    assert p == pytest.approx(mass)


def test_never_one_one_has_stationary_probability_zero_unless_forbidden():
    never = _automaton(BuchiAutomaton, NEVER_ONE_ONE, "ac")
    assert omega_probability(golden_mean(0.4), never) == pytest.approx(1.0)
    assert omega_probability(bernoulli(0.3), never) == pytest.approx(0.0)


def test_safety_probability_is_the_limit_of_prefix_probabilities():
    never = _automaton(DFA, NEVER_ONE_ONE, "ac")
    model = _stops_emitting_ones(0.4)
    omega = omega_probability(model, _automaton(BuchiAutomaton, NEVER_ONE_ONE, "ac"), start="C")
    finite = [regular_language_probability(model, never, n, start="C") for n in (10, 40, 160)]
    assert finite[0] >= finite[1] >= finite[2] >= omega - 1e-12
    assert finite[-1] == pytest.approx(omega, abs=1e-9)
    assert 0.0 < omega < 1.0


def test_monte_carlo_agrees_with_omega_probability():
    model = _split(0.3)
    exact = omega_probability(model, _gf1(), start="S")
    rng = np.random.default_rng(20261007)
    runs = 1000
    hits = sum("1" in model.sample(40, rng)[0][-15:] for _ in range(runs))
    assert hits / runs == pytest.approx(exact, abs=4 * np.sqrt(exact * (1 - exact) / runs))


def test_nondeterministic_buchi_is_rejected():
    nba = _automaton(BuchiAutomaton, [("a", "a", "0"), ("a", "a", "1"), ("a", "b", "0"), ("b", "b", "0")], "b")
    with pytest.raises(NonDeterministicError):
        omega_probability(bernoulli(), nba)


def _brute_language_probability(model, accepts, n: int, start=None) -> float:
    if start is None:
        start = model.to_mealy().stationary_distribution()
    return sum(model.word_probability(word, start=start) for word in product("01", repeat=n) if accepts(word))


@pytest.mark.parametrize("model", [bernoulli(0.3), golden_mean(0.4), even_process(0.6), binary_markov_chain(0.2, 0.7)])
@pytest.mark.parametrize("n", [0, 1, 2, 5, 8])
def test_regular_language_probability_matches_word_enumeration(model, n):
    even_ones = _automaton(DFA, [("a", "a", "0"), ("a", "b", "1"), ("b", "b", "0"), ("b", "a", "1")], "a")
    no_one_one = _automaton(DFA, NEVER_ONE_ONE, "ac")
    for dfa in (even_ones, no_one_one):
        expected = _brute_language_probability(model, dfa.recognizes, n)
        assert regular_language_probability(model, dfa, n) == pytest.approx(expected, abs=1e-12)


def test_regular_language_probability_with_explicit_start():
    model = _split(0.3)
    contains_one = _automaton(DFA, EVENTUALLY_ONE, "b")
    for n in range(6):
        expected = _brute_language_probability(model, contains_one.recognizes, n, start={"S": 1.0})
        assert regular_language_probability(model, contains_one, n, start="S") == pytest.approx(expected)


def test_regular_language_probability_validates_input():
    dfa = _automaton(DFA, ALWAYS_ZERO, "a")
    with pytest.raises(ValueError, match="nonnegative"):
        regular_language_probability(bernoulli(), dfa, -1)
