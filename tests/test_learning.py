"""Tests for NL* learning of canonical RFSAs and maximized prime átomata."""

import pytest
from hypothesis import given, settings

from sofic.automata.active import AutomatonEquivalenceOracle, LanguageMembershipOracle
from sofic.automata.algorithms import equivalent
from sofic.automata.atomaton import MaximizedPrimeAtomaton
from sofic.automata.dfa import DFA
from sofic.automata.learning import learn_prime_atomaton_nlstar, learn_rfsa_from_language, learn_rfsa_nlstar
from sofic.automata.nfa import NFA
from sofic.automata.rfsa import CanonicalRFSA
from sofic.testing.strategies import dfas


def _a_then_n_symbols(n: int) -> NFA:
    nfa = NFA(input_alphabet=frozenset("ab"), initial_states=frozenset({0}), accepting_states=frozenset({n + 1}))
    for state in range(n + 2):
        nfa.graph.add_state(state)
    nfa.add_transition(0, 0, "a")
    nfa.add_transition(0, 0, "b")
    nfa.add_transition(0, 1, "a")
    for state in range(1, n + 1):
        nfa.add_transition(state, state + 1, "a")
        nfa.add_transition(state, state + 1, "b")
    return nfa


def _signature(aut) -> tuple[int, int, int, int]:
    return (
        len(list(aut.states())),
        len(list(aut.transitions())),
        len(aut.initial_states),
        len(aut.accepting_states),
    )


@pytest.mark.parametrize("n", [1, 2, 3])
def test_nlstar_learns_small_canonical_rfsa_of_exponential_dfa(n):
    target = _a_then_n_symbols(n)
    learned = learn_rfsa_from_language(target, frozenset("ab"))
    assert isinstance(learned, CanonicalRFSA)
    assert equivalent(learned, target)
    assert len(list(learned.states())) == n + 2
    assert _signature(learned) == _signature(CanonicalRFSA.from_language(target))


def test_nlstar_with_bounded_oracle_on_explicit_membership():
    dfa = DFA(
        input_alphabet=frozenset({"a", "b"}), initial_states=frozenset({"q0"}), accepting_states=frozenset({"q1"})
    )
    dfa.graph.add_state("q0")
    dfa.graph.add_state("q1")
    dfa.add_transition("q0", "q1", "a")
    dfa.add_transition("q1", "q1", "a")
    dfa.add_transition("q1", "q1", "b")
    learned = learn_rfsa_from_language(set(dfa.iter_language(max_length=6)), frozenset("ab"), max_length=5)
    for word in dfa.iter_language(max_length=5):
        assert learned.recognizes(word)


@settings(max_examples=40, deadline=None)
@given(dfas(min_states=1, max_states=4))
def test_nlstar_matches_canonical_rfsa_construction(dfa):
    alphabet = frozenset({"0", "1"})
    learned = learn_rfsa_nlstar(alphabet, LanguageMembershipOracle(dfa), AutomatonEquivalenceOracle(dfa, alphabet))
    assert equivalent(learned, dfa)
    assert _signature(learned) == _signature(CanonicalRFSA.from_language(dfa))


@settings(max_examples=25, deadline=None)
@given(dfas(min_states=1, max_states=4))
def test_prime_atomaton_learner_matches_construction(dfa):
    alphabet = frozenset({"0", "1"})
    learned = learn_prime_atomaton_nlstar(
        alphabet, LanguageMembershipOracle(dfa), AutomatonEquivalenceOracle(dfa, alphabet)
    )
    assert isinstance(learned, MaximizedPrimeAtomaton)
    assert equivalent(learned, dfa)
    assert _signature(learned) == _signature(MaximizedPrimeAtomaton.from_language(dfa))
