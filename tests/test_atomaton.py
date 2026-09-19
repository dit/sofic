"""Tests for átomaton skeletons."""

import pytest

from sofic.automata.atomaton import Atomaton, MaximizedPrimeAtomaton, atomic_states, is_atomic
from sofic.automata.dfa import DFA
from sofic.automata.nfa import NFA
from sofic.automata.rfsa import CanonicalRFSA
from sofic.exceptions import SoficValidationError


def test_atomaton_validate():
    auto = Atomaton(
        input_alphabet=frozenset({"a"}),
        initial_states=frozenset({"q0"}),
        accepting_states=frozenset({"q0"}),
    )
    auto.graph.add_state("q0")
    auto.validate()


def _lang_dfa() -> DFA:
    dfa = DFA(input_alphabet=frozenset({"a"}), initial_states=frozenset({"q0"}), accepting_states=frozenset({"q0"}))
    dfa.graph.add_state("q0")
    dfa.add_transition("q0", "q0", "a")
    return dfa


def test_atomaton_from_language():
    auto = Atomaton.from_language(_lang_dfa())
    auto.validate()


def test_mpa_from_canonical_rfsa():
    rfsa = CanonicalRFSA.from_language(_lang_dfa())
    mpa = MaximizedPrimeAtomaton.from_canonical_rfsa(rfsa)
    mpa.validate()


def _nfa(spec, initial, accepting) -> NFA:
    nfa = NFA(
        input_alphabet=frozenset("ab"),
        initial_states=frozenset(initial),
        accepting_states=frozenset(accepting),
    )
    for state in spec:
        nfa.graph.add_state(state)
    for source, moves in spec.items():
        for symbol, targets in moves.items():
            for target in targets:
                nfa.add_transition(source, target, symbol)
    return nfa


def _three_state_dfa() -> DFA:
    """Minimal DFA whose átomaton has six states (Brzozowski & Tamm, 2014)."""
    dfa = DFA(input_alphabet=frozenset("ab"), initial_states=frozenset({0}), accepting_states=frozenset({2}))
    for state in (0, 1, 2):
        dfa.graph.add_state(state)
    dfa.add_transition(0, 1, "a")
    dfa.add_transition(0, 0, "b")
    dfa.add_transition(1, 2, "a")
    dfa.add_transition(1, 1, "b")
    dfa.add_transition(2, 2, "a")
    dfa.add_transition(2, 0, "b")
    return dfa


def test_atomicity_of_individual_states_matches_example_5():
    """Brzozowski & Tamm (2014), Example 5: only state 0 is atomic."""
    nfa = _nfa(
        {0: {"a": [1, 2], "b": []}, 1: {"a": [1], "b": [2]}, 2: {"a": [2], "b": [1]}},
        initial=[0],
        accepting=[2],
    )
    assert atomic_states(nfa) == frozenset({0})
    assert not is_atomic(nfa)


# Brzozowski & Tamm (2014), Example 6: all four atomic/non-atomic combinations
# occur among NFAs accepting the same language, Sigma* a b Sigma*.
_EXAMPLE_6 = {
    "Na": (
        {0: {"a": [0, 1], "b": [0]}, 1: {"a": [], "b": [2]}, 2: {"a": [2], "b": [2]}},
        False,
        False,
    ),
    "Nb": (
        {0: {"a": [1], "b": [0]}, 1: {"a": [1], "b": [1, 2]}, 2: {"a": [1, 2], "b": [0]}},
        True,
        False,
    ),
    "Nc": (
        {0: {"a": [1], "b": [0]}, 1: {"a": [1], "b": [1, 2]}, 2: {"a": [2], "b": [2]}},
        True,
        True,
    ),
}


@pytest.mark.parametrize("name", sorted(_EXAMPLE_6))
def test_atomicity_of_an_nfa_and_its_reverse_are_independent(name):
    spec, forward, backward = _EXAMPLE_6[name]
    nfa = _nfa(spec, initial=[0], accepting=[2])
    assert is_atomic(nfa) is forward
    assert is_atomic(nfa.reverse()) is backward


def test_subset_construction_is_minimal_exactly_when_the_reverse_is_atomic():
    """Brzozowski & Tamm (2014), Theorem 5."""
    for spec, _, _ in _EXAMPLE_6.values():
        nfa = _nfa(spec, initial=[0], accepting=[2])
        determinized = nfa.determinize()
        states = len(tuple(determinized.states()))
        minimal = len(tuple(determinized.minimize().states()))
        assert is_atomic(nfa.reverse()) is (states == minimal)


def test_atomaton_is_the_reverse_of_the_minimal_dfa_of_the_reverse_language():
    dfa = _three_state_dfa()
    atomaton = Atomaton.from_language(dfa)
    expected = dfa.reverse().determinize().minimize().reverse()

    assert len(tuple(atomaton.states())) == len(tuple(expected.states()))
    # The átomaton is genuinely nondeterministic here: six atoms, three initial.
    assert len(tuple(atomaton.states())) == 6
    assert len(atomaton.initial_states) == 3


def test_atomaton_is_atomic_and_determinizes_to_the_minimal_dfa():
    dfa = _three_state_dfa()
    atomaton = Atomaton.from_language(dfa)

    assert is_atomic(atomaton)
    atomaton.validate()
    assert len(tuple(atomaton.determinize().minimize().states())) == 3


def test_validate_rejects_a_non_atomic_automaton():
    spec, _, _ = _EXAMPLE_6["Na"]
    non_atomic = _nfa(spec, initial=[0], accepting=[2])
    auto = Atomaton(
        input_alphabet=non_atomic.input_alphabet,
        initial_states=non_atomic.initial_states,
        accepting_states=non_atomic.accepting_states,
        graph=non_atomic.graph.copy(),
    )
    with pytest.raises(SoficValidationError, match="not a union of atoms"):
        auto.validate()
