"""Tests for hierarchy-wide reverse operations."""

from __future__ import annotations

import numpy as np

from pensive.automata.nfa import NFA
from pensive.generators.markov import MarkovChain
from pensive.graph import ATTR_PROB, ATTR_SYMBOL, TransitionGraph
from pensive.operations import reverse as reverse_model
from pensive.shifts.sofic import SoficShift


def test_transition_graph_reverse():
    graph = TransitionGraph()
    graph.add_state("u")
    graph.add_state("v")
    graph.add_transition("u", "v", **{ATTR_SYMBOL: "a", ATTR_PROB: 0.5})
    rev = graph.reverse()
    transitions = list(rev.transitions())
    assert len(transitions) == 1
    transition = transitions[0]
    assert transition.source == "v"
    assert transition.target == "u"
    assert transition.data[ATTR_SYMBOL] == "a"
    assert transition.data[ATTR_PROB] == 0.5


def test_sofic_shift_reverse_involution():
    shift = SoficShift(symbol_alphabet=frozenset({"0", "1"}))
    shift.graph.add_state("s")
    shift.graph.add_transition("s", "s", **{ATTR_SYMBOL: "0"})
    twice = shift.reverse().reverse()
    forward = {(t.source, t.target, t.data.get(ATTR_SYMBOL)) for t in shift.transitions()}
    backward = {(t.source, t.target, t.data.get(ATTR_SYMBOL)) for t in twice.transitions()}
    assert forward == backward


def _two_state_chain() -> MarkovChain:
    chain = MarkovChain(initial_distribution={"a": 1.0})
    for state in ("a", "b"):
        chain.graph.add_state(state)
    chain.graph.add_transition("a", "a", **{ATTR_PROB: 0.5})
    chain.graph.add_transition("a", "b", **{ATTR_PROB: 0.5})
    chain.graph.add_transition("b", "a", **{ATTR_PROB: 0.5})
    chain.graph.add_transition("b", "b", **{ATTR_PROB: 0.5})
    return chain


def test_markov_chain_stationary_distribution():
    chain = _two_state_chain()
    pi = chain.stationary_distribution()
    assert np.isclose(pi.sum(), 1.0)
    idx = chain.reindex()
    transition = np.zeros((len(idx), len(idx)))
    for source in idx.states:
        i = idx.index(source)
        for edge in chain.graph.out_transitions(source):
            j = idx.index(edge.target)
            transition[i, j] += edge.data.get(ATTR_PROB, 0.0)
    assert np.allclose(pi @ transition, pi, rtol=1e-8, atol=1e-10)


def test_markov_chain_time_reverse():
    chain = _two_state_chain()
    pi = chain.stationary_distribution()
    rev = chain.reverse()
    pi_rev = rev.stationary_distribution()
    assert np.allclose(pi, pi_rev, rtol=1e-8, atol=1e-10)

    idx = chain.reindex()
    for source in idx.states:
        i = idx.index(source)
        for edge in chain.graph.out_transitions(source):
            target = edge.target
            j = idx.index(target)
            forward = float(edge.data.get(ATTR_PROB, 0.0))
            reverse_prob = next(
                t.data.get(ATTR_PROB, 0.0)
                for t in rev.graph.out_transitions(target)
                if t.target == source
            )
            assert np.isclose(pi[i] * forward, pi_rev[j] * reverse_prob, rtol=1e-8, atol=1e-10)


def test_operations_reverse_dispatches():
    nfa = NFA(
        input_alphabet=frozenset({"a"}),
        initial_states=frozenset({"q0"}),
        accepting_states=frozenset({"q1"}),
    )
    nfa.graph.add_state("q0")
    nfa.graph.add_state("q1")
    nfa.add_transition("q0", "q1", "a")
    reversed_nfa = reverse_model(nfa)
    assert reversed_nfa.recognizes(("a",))


def test_mealy_hmm_reverse():
    from pensive.generators.mealy import MealyHMM
    from pensive.graph import ATTR_EMISSION, ATTR_PROB

    hmm = MealyHMM(
        initial_distribution={"q0": 1.0},
        observation_alphabet=frozenset({"0"}),
    )
    hmm.graph.add_state("q0")
    hmm.graph.add_transition("q0", "q0", **{ATTR_PROB: 1.0, ATTR_EMISSION: "0"})
    reversed_hmm = hmm.reverse()
    reversed_hmm.validate()
