"""Tests for automata algorithms: reverse, determinize, minimize."""

from __future__ import annotations

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from sofic.automata.algorithms import (
    complete,
    determinize,
    equivalent,
    minimize,
    reverse,
    trim,
)
from sofic.automata.dfa import DFA
from sofic.automata.languages.base import AutomatonLanguage
from sofic.automata.languages.operations import reverse as reverse_language
from sofic.automata.nfa import NFA


def _epsilon_nfa() -> NFA:
    nfa = NFA(
        input_alphabet=frozenset({"a"}),
        initial_states=frozenset({"q0"}),
        accepting_states=frozenset({"q1"}),
    )
    for state in ("q0", "q1"):
        nfa.graph.add_state(state)
    nfa.add_transition("q0", "q1")
    return nfa


def _nondeterministic_ends_with_a() -> NFA:
    nfa = NFA(
        input_alphabet=frozenset({"a", "b"}),
        initial_states=frozenset({"q0"}),
        accepting_states=frozenset({"q2"}),
    )
    for state in ("q0", "q1", "q2"):
        nfa.graph.add_state(state)
    nfa.add_transition("q0", "q0", "a")
    nfa.add_transition("q0", "q0", "b")
    nfa.add_transition("q0", "q1", "a")
    nfa.add_transition("q0", "q1", "b")
    nfa.add_transition("q1", "q2", "a")
    nfa.add_transition("q1", "q1", "b")
    nfa.add_transition("q2", "q2", "a")
    nfa.add_transition("q2", "q2", "b")
    return nfa


def _reducible_dfa() -> DFA:
    dfa = DFA(
        input_alphabet=frozenset({"a", "b"}),
        initial_states=frozenset({"s0"}),
        accepting_states=frozenset({"s1", "s2"}),
    )
    for state in ("s0", "s1", "s2"):
        dfa.graph.add_state(state)
    dfa.add_transition("s0", "s1", "a")
    dfa.add_transition("s0", "s2", "b")
    dfa.add_transition("s1", "s1", "a")
    dfa.add_transition("s1", "s1", "b")
    dfa.add_transition("s2", "s2", "a")
    dfa.add_transition("s2", "s2", "b")
    return dfa


def _minimal_dfa() -> DFA:
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


def _hopcroft_split_regression_dfa() -> DFA:
    dfa = DFA(
        input_alphabet=frozenset({"0", "1"}),
        initial_states=frozenset({0}),
        accepting_states=frozenset({1}),
    )
    for state in range(4):
        dfa.graph.add_state(state)
    transitions = {
        (0, "1"): 3,
        (0, "0"): 2,
        (1, "1"): 2,
        (1, "0"): 1,
        (2, "1"): 3,
        (2, "0"): 1,
        (3, "1"): 0,
        (3, "0"): 1,
    }
    for (source, symbol), target in transitions.items():
        dfa.add_transition(source, target, symbol)
    return dfa


def _words(alphabet: frozenset[str], max_len: int) -> list[tuple[str, ...]]:
    from itertools import product

    base = sorted(alphabet)
    result: list[tuple[str, ...]] = [()]
    for length in range(1, max_len + 1):
        result.extend(tuple(word) for word in product(base, repeat=length))
    return result


def test_trim_removes_dead_states():
    dfa = _reducible_dfa()
    dfa.graph.add_state("dead")
    trimmed = trim(dfa)
    assert "dead" not in set(trimmed.states())
    assert dfa.recognizes(("a",)) == trimmed.recognizes(("a",))


def test_complete_adds_trap_transitions():
    dfa = _minimal_dfa()
    completed = complete(dfa)
    trap_states = [state for state in completed.states() if state not in {"q0", "q1"}]
    assert len(trap_states) == 1
    trap = trap_states[0]
    for symbol in completed.input_alphabet:
        assert completed.delta(trap, symbol) == {trap}


def test_reverse_involution_language():
    nfa = _nondeterministic_ends_with_a()
    alphabet = frozenset({"a", "b"})
    for word in _words(alphabet, 4):
        assert nfa.recognizes(word) == reverse(reverse(nfa)).recognizes(word)


def test_determinize_preserves_language():
    nfa = _nondeterministic_ends_with_a()
    dfa = determinize(nfa)
    alphabet = frozenset({"a", "b"})
    for word in _words(alphabet, 4):
        assert nfa.recognizes(word) == dfa.recognizes(word)


def test_determinize_epsilon_nfa():
    nfa = _epsilon_nfa()
    dfa = determinize(nfa)
    assert dfa.recognizes(())
    assert not dfa.recognizes(("a",))


@pytest.mark.parametrize("algorithm", ["hopcroft", "moore", "brzozowski"])
def test_minimize_preserves_language(algorithm: str):
    source = _reducible_dfa()
    minimized = minimize(source, algorithm=algorithm)  # type: ignore[arg-type]
    alphabet = frozenset({"a", "b"})
    for word in _words(alphabet, 4):
        assert source.recognizes(word) == minimized.recognizes(word)


def test_hopcroft_moore_equivalent():
    source = _reducible_dfa()
    hopcroft = minimize(source, algorithm="hopcroft")
    moore = minimize(source, algorithm="moore")
    assert equivalent(hopcroft, moore, frozenset({"a", "b"}))
    assert len(list(hopcroft.states())) == len(list(moore.states()))


def test_hopcroft_refines_all_split_blocks():
    source = _hopcroft_split_regression_dfa()
    hopcroft = minimize(source, algorithm="hopcroft")
    moore = minimize(source, algorithm="moore")
    alphabet = frozenset({"0", "1"})
    assert equivalent(hopcroft, moore, alphabet)
    assert len(list(hopcroft.states())) == len(list(moore.states()))
    for word in _words(alphabet, 4):
        assert source.recognizes(word) == hopcroft.recognizes(word)


def test_brzozowski_matches_hopcroft_on_nfa():
    nfa = _nondeterministic_ends_with_a()
    brz = minimize(nfa, algorithm="brzozowski")
    hop = minimize(nfa, algorithm="hopcroft")
    assert equivalent(brz, hop, frozenset({"a", "b"}))


def test_minimize_idempotent():
    source = _reducible_dfa()
    once = minimize(source, algorithm="hopcroft")
    twice = minimize(once, algorithm="hopcroft")
    assert equivalent(once, twice, frozenset({"a", "b"}))


def test_minimize_shrinks_reducible_dfa():
    source = _reducible_dfa()
    minimized = minimize(source, algorithm="hopcroft")
    assert len(list(minimized.states())) < len(list(source.states()))


def test_minimize_already_minimal():
    source = _minimal_dfa()
    minimized = minimize(source, algorithm="hopcroft")
    assert equivalent(source, minimized, frozenset({"a", "b"}))


def test_instance_methods():
    nfa = _nondeterministic_ends_with_a()
    assert equivalent(nfa.determinize(), determinize(nfa), frozenset({"a", "b"}))
    assert nfa.reverse().recognizes(("a",)) == reverse(nfa).recognizes(("a",))
    assert equivalent(nfa.minimize(), minimize(nfa), frozenset({"a", "b"}))


def test_language_reverse_automaton():
    nfa = _minimal_dfa()
    lang = AutomatonLanguage(nfa)
    rev = reverse_language(lang)
    assert isinstance(rev, AutomatonLanguage)
    assert rev.automaton.recognizes(("a",)) == lang.automaton.recognizes(("a",))


@settings(max_examples=25, deadline=None)
@given(st.lists(st.sampled_from(["a", "b"]), max_size=5))
def test_hypothesis_determinize_preserves_language(word: list[str]):
    nfa = _nondeterministic_ends_with_a()
    assert nfa.recognizes(tuple(word)) == determinize(nfa).recognizes(tuple(word))
