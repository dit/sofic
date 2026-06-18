"""Tests for DFA."""

import pytest

from pensive.automata.dfa import DFA
from pensive.exceptions import NonDeterministicError
from pensive.graph import EPSILON


def _dfa() -> DFA:
    dfa = DFA(
        input_alphabet=frozenset({"a", "b"}),
        initial_states=frozenset({"q0"}),
        accepting_states=frozenset({"q1"}),
    )
    for state in ("q0", "q1"):
        dfa.graph.add_state(state)
    dfa.add_transition("q0", "q1", "a")
    dfa.add_transition("q0", "q0", "b")
    dfa.add_transition("q1", "q1", "a")
    dfa.add_transition("q1", "q0", "b")
    return dfa


def test_recognizes():
    dfa = _dfa()
    assert dfa.recognizes(("a",))
    assert dfa.recognizes(("b", "a"))
    assert not dfa.recognizes(())
    assert not dfa.recognizes(("b",))


def test_validate_passes():
    _dfa().validate()


def test_rejects_epsilon_transition():
    dfa = _dfa()
    with pytest.raises(NonDeterministicError):
        dfa.add_transition("q0", "q1", EPSILON)


def test_networkx_round_trip():
    dfa = _dfa()
    dfa.validate()
    restored = DFA.from_networkx(
        dfa.to_networkx(),
        input_alphabet=dfa.input_alphabet,
        initial_states=dfa.initial_states,
        accepting_states=dfa.accepting_states,
    )
    assert restored.recognizes(("a",))
    assert not restored.recognizes(("b",))


def test_reindex():
    dfa = _dfa()
    idx = dfa.reindex()
    assert len(idx) == 2
    assert "q0" in idx


def test_copy():
    dfa = _dfa()
    clone = dfa.copy()
    clone.graph.add_state("q_extra")
    assert "q_extra" not in set(dfa.states())
    assert "q_extra" in set(clone.states())


def test_minimize_returns_new_dfa():
    dfa = _dfa()
    minimized = dfa.minimize()
    assert minimized is not dfa
    assert dfa.recognizes(("a",)) == minimized.recognizes(("a",))
