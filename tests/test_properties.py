"""Tests for is_unifilar / is_deterministic predicates."""

from __future__ import annotations

import pytest

from pensive.automata.dfa import DFA
from pensive.automata.nfa import NFA
from pensive.automata.transducers import MealyMachine
from pensive.automata.unifilar import UnifilarAutomaton
from pensive.examples.epsilon_machines import ellison_fig15_bidirectional, even_process, golden_mean
from pensive.exceptions import NonDeterministicError, UnifilarityError
from pensive.generators.epsilon_machine import EpsilonMachine
from pensive.generators.markov import MarkovChain
from pensive.generators.mealy import MealyHMM
from pensive.generators.moore import MooreHMM
from pensive.graph import ATTR_EMISSION, ATTR_EMISSION_DIST, ATTR_PROB, ATTR_SYMBOL, EPSILON
from pensive.shifts.sofic import SoficShift


def _dfa() -> DFA:
    dfa = DFA(
        input_alphabet=frozenset({"a", "b"}),
        initial_states=frozenset({"q0"}),
        accepting_states=frozenset({"q1"}),
    )
    dfa.graph.add_state("q0")
    dfa.graph.add_state("q1")
    dfa.graph.add_transition("q0", "q1", symbol="a")
    dfa.graph.add_transition("q1", "q1", symbol="b")
    return dfa


def _mealy_hmm() -> MealyHMM:
    hmm = MealyHMM(
        initial_distribution={"q0": 1.0},
        observation_alphabet=frozenset({"0", "1"}),
    )
    hmm.graph.add_state("q0")
    hmm.graph.add_transition("q0", "q0", **{ATTR_PROB: 0.6, ATTR_EMISSION: "0"})
    hmm.graph.add_transition("q0", "q0", **{ATTR_PROB: 0.4, ATTR_EMISSION: "1"})
    return hmm


def test_dfa_is_deterministic():
    assert _dfa().is_deterministic()


def test_dfa_not_deterministic_multiple_targets():
    dfa = _dfa()
    dfa.graph.add_transition("q0", "q0", symbol="a")
    assert not dfa.is_deterministic()
    with pytest.raises(NonDeterministicError):
        dfa.validate()


def test_nfa_not_deterministic_multiple_initials():
    nfa = NFA(
        input_alphabet=frozenset({"a"}),
        initial_states=frozenset({"q0", "q1"}),
        accepting_states=frozenset({"q0"}),
    )
    nfa.graph.add_state("q0")
    nfa.graph.add_state("q1")
    nfa.graph.add_transition("q0", "q0", symbol="a")
    assert not nfa.is_deterministic()


def test_nfa_not_deterministic_with_epsilon():
    nfa = NFA(
        input_alphabet=frozenset({"a"}),
        initial_states=frozenset({"q0"}),
        accepting_states=frozenset({"q0"}),
    )
    nfa.graph.add_state("q0")
    nfa.graph.add_transition("q0", "q0", symbol=EPSILON)
    assert not nfa.is_deterministic()


def test_unifilar_automaton_is_unifilar():
    aut = UnifilarAutomaton(
        input_alphabet=frozenset({"0", "1"}),
        initial_states=frozenset({"A"}),
        accepting_states=frozenset({"A"}),
    )
    aut.graph.add_state("A")
    aut.graph.add_state("B")
    aut.graph.add_transition("A", "B", symbol="0")
    aut.graph.add_transition("A", "A", symbol="1")
    assert aut.is_unifilar()
    aut.validate()


def test_unifilar_automaton_not_unifilar():
    aut = UnifilarAutomaton(
        input_alphabet=frozenset({"0"}),
        initial_states=frozenset({"A"}),
        accepting_states=frozenset({"A"}),
    )
    aut.graph.add_state("A")
    aut.graph.add_state("B")
    aut.graph.add_state("C")
    aut.graph.add_transition("A", "B", symbol="0")
    aut.graph.add_transition("A", "C", symbol="0")
    assert not aut.is_unifilar()
    with pytest.raises(UnifilarityError):
        aut.validate()


def test_sofic_shift_is_unifilar():
    shift = SoficShift(symbol_alphabet=frozenset({"0", "1"}))
    shift.graph.add_state("A")
    shift.graph.add_state("B")
    shift.graph.add_transition("A", "B", **{ATTR_SYMBOL: "0"})
    shift.graph.add_transition("A", "A", **{ATTR_SYMBOL: "1"})
    assert shift.is_unifilar()


def test_mealy_hmm_is_unifilar():
    assert _mealy_hmm().is_unifilar()


def test_mealy_hmm_not_unifilar():
    hmm = _mealy_hmm()
    hmm.graph.add_transition("q0", "q0", **{ATTR_PROB: 0.1, ATTR_EMISSION: "0"})
    assert not hmm.is_unifilar()


def test_epsilon_machine_is_unifilar():
    eps = EpsilonMachine.from_hmm(_mealy_hmm())
    assert eps.is_unifilar()
    eps.validate()


def test_epsilon_machine_validate_raises_when_not_unifilar():
    bad = EpsilonMachine(
        initial_distribution={"q0": 1.0},
        observation_alphabet=frozenset({"0"}),
    )
    bad.graph.add_state("q0")
    bad.graph.add_transition("q0", "q0", **{ATTR_PROB: 0.5, ATTR_EMISSION: "0"})
    bad.graph.add_transition("q0", "q0", **{ATTR_PROB: 0.5, ATTR_EMISSION: "0"})
    assert not bad.is_unifilar()
    with pytest.raises(UnifilarityError):
        bad.validate()


def test_moore_hmm_is_unifilar_delegates_to_mealy():
    hmm = MooreHMM(
        initial_distribution={"A": 1.0},
        observation_alphabet=frozenset({0, 1}),
    )
    hmm.graph.add_state("A", **{ATTR_EMISSION_DIST: {0: 0.5, 1: 0.5}})
    hmm.graph.add_transition("A", "A", **{ATTR_PROB: 1.0})
    assert hmm.is_unifilar()


def test_markov_chain_is_deterministic():
    chain = MarkovChain(initial_distribution={"A": 1.0})
    chain.graph.add_state("A")
    chain.graph.add_state("B")
    chain.graph.add_transition("A", "B", **{ATTR_PROB: 1.0})
    chain.graph.add_transition("B", "B", **{ATTR_PROB: 1.0})
    assert chain.is_deterministic()


def test_markov_chain_not_deterministic():
    chain = MarkovChain(initial_distribution={"A": 1.0})
    chain.graph.add_state("A")
    chain.graph.add_state("B")
    chain.graph.add_transition("A", "B", **{ATTR_PROB: 0.5})
    chain.graph.add_transition("A", "A", **{ATTR_PROB: 0.5})
    assert not chain.is_deterministic()


def test_transducer_is_deterministic():
    tr = MealyMachine(
        input_alphabet=frozenset({"a"}),
        output_alphabet=frozenset({"x"}),
        initial_states=frozenset({"q0"}),
    )
    tr.graph.add_state("q0")
    tr.graph.add_transition("q0", "q0", symbol="a", output="x")
    assert tr.is_deterministic()


def test_transducer_not_deterministic():
    tr = MealyMachine(
        input_alphabet=frozenset({"a"}),
        output_alphabet=frozenset({"x", "y"}),
        initial_states=frozenset({"q0"}),
    )
    tr.graph.add_state("q0")
    tr.graph.add_state("q1")
    tr.graph.add_transition("q0", "q0", symbol="a", output="x")
    tr.graph.add_transition("q0", "q1", symbol="a", output="y")
    assert not tr.is_deterministic()


def test_bidirectional_epsilon_machine_not_unifilar():
    bidir = ellison_fig15_bidirectional()
    assert not bidir.is_unifilar()


def test_golden_mean_epsilon_machine_is_unifilar():
    assert golden_mean(0.5).is_unifilar()


def test_mealy_structural_predicates_on_one_state_process():
    hmm = _mealy_hmm()
    assert hmm.is_counifilar()
    assert hmm.is_irreducible()
    assert hmm.is_ergodic()
    assert hmm.is_ergodic(weak=False)
    assert hmm.is_stationary()
    assert hmm.is_detailed_balance()
    assert not hmm.is_periodic()


def test_mealy_not_counifilar_when_target_symbol_has_multiple_sources():
    hmm = MealyHMM(
        initial_distribution={"q0": 1.0},
        observation_alphabet=frozenset({"0"}),
    )
    for state in ("q0", "q1", "q2"):
        hmm.graph.add_state(state)
    hmm.graph.add_transition("q0", "q1", **{ATTR_PROB: 1.0, ATTR_EMISSION: "0"})
    hmm.graph.add_transition("q2", "q1", **{ATTR_PROB: 1.0, ATTR_EMISSION: "0"})
    assert not hmm.is_counifilar()


def test_epsilon_machine_markov_and_strictly_sofic_predicates():
    assert golden_mean(0.5).is_markov()
    assert not golden_mean(0.5).is_strictly_sofic()
    assert not even_process(0.5).is_markov()
    assert even_process(0.5).is_strictly_sofic()
