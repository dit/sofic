"""Tests for NFA and epsilon closure."""

from pensive.automata.nfa import NFA
from pensive.graph import EPSILON


def _nfa_with_epsilon() -> NFA:
    nfa = NFA(
        input_alphabet=frozenset({"a"}),
        initial_states=frozenset({"q0"}),
        accepting_states=frozenset({"q1"}),
    )
    for state in ("q0", "q1", "q2"):
        nfa.graph.add_state(state)
    nfa.add_transition("q0", "q1")  # epsilon
    nfa.add_transition("q0", "q2", "a")
    return nfa


def test_epsilon_closure():
    nfa = _nfa_with_epsilon()
    closure = nfa.epsilon_closure({"q0"})
    assert closure == {"q0", "q1"}


def test_recognizes_via_epsilon():
    nfa = _nfa_with_epsilon()
    assert nfa.recognizes(())
    assert not nfa.recognizes(("a",))


def test_validate_and_round_trip():
    nfa = _nfa_with_epsilon()
    nfa.validate()
    restored = NFA.from_networkx(
        nfa.to_networkx(),
        input_alphabet=nfa.input_alphabet,
        initial_states=nfa.initial_states,
        accepting_states=nfa.accepting_states,
    )
    assert restored.recognizes(())
