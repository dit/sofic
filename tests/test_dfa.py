"""Tests for DFA."""

import re

import pytest

from pensive.automata import automaton_to_regex
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


def test_words_of_length():
    dfa = _dfa()
    assert list(dfa.words_of_length(2)) == [("a", "a"), ("b", "a")]


def test_iter_language_with_bound():
    dfa = _dfa()
    assert list(dfa.iter_language(max_length=2)) == [("a",), ("a", "a"), ("b", "a")]


def test_to_regex_matches_language_on_small_words():
    dfa = _dfa()
    pattern = dfa.to_regex()
    for word in [(), ("a",), ("b",), ("a", "a"), ("a", "b"), ("b", "a")]:
        assert bool(re.fullmatch(pattern, "".join(word))) == dfa.recognizes(word)


def test_automaton_to_regex_function():
    dfa = _dfa()
    assert automaton_to_regex(dfa) == dfa.to_regex()


def test_standard_regular_operations_methods():
    dfa = _dfa()
    complement = dfa.complement()
    union = dfa.union(complement)
    intersection = dfa.intersection(complement)
    difference = dfa.difference(complement)
    concat = dfa.concat(dfa)
    star = dfa.kleene_star()

    samples = [(), ("a",), ("b",), ("a", "a"), ("a", "b"), ("b", "a")]
    for word in samples:
        assert union.recognizes(word)
        assert not intersection.recognizes(word)
        assert difference.recognizes(word) == dfa.recognizes(word)

    assert concat.recognizes(("a", "a"))
    assert not concat.recognizes(("a",))
    assert star.recognizes(())
    assert star.recognizes(("a",))
    assert not star.recognizes(("b",))
