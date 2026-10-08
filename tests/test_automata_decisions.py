"""Emptiness, shortest-witness, universality, and inclusion on finite automata."""

from __future__ import annotations

from itertools import product

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from sofic.automata import DFA, NFA
from sofic.graph import EPSILON
from sofic.testing.strategies import dfas, nfas

ALPHABET = ("a", "b")
MAX_LENGTH = 6

small_nfas = nfas(alphabet=ALPHABET, max_states=4, max_transitions=8)
small_automata = st.one_of(small_nfas, dfas(alphabet=ALPHABET, max_states=3))


def _words(max_length: int = MAX_LENGTH):
    for length in range(max_length + 1):
        yield from product(ALPHABET, repeat=length)


def _language(aut, max_length: int = MAX_LENGTH) -> set[tuple[str, ...]]:
    return {word for word in _words(max_length) if aut.recognizes(word)}


def _nfa(initial, accepting, edges, alphabet=ALPHABET) -> NFA:
    nfa = NFA(
        input_alphabet=frozenset(alphabet),
        initial_states=frozenset(initial),
        accepting_states=frozenset(accepting),
    )
    for source, target, _symbol in edges:
        nfa.graph.add_state(source)
        nfa.graph.add_state(target)
    for state in set(initial) | set(accepting):
        nfa.graph.add_state(state)
    for source, target, symbol in edges:
        nfa.add_transition(source, target, symbol)
    return nfa


def test_empty_language_has_no_witness():
    nfa = _nfa({0}, {1}, [(0, 0, "a")])
    assert nfa.is_empty()
    assert nfa.accepted_word() is None


def test_epsilon_reaches_acceptance_with_empty_word():
    nfa = _nfa({0}, {2}, [(0, 1, EPSILON), (1, 2, EPSILON)])
    assert nfa.accepted_word() == ()


def test_shortest_witness_ignores_epsilon_cost():
    nfa = _nfa({0}, {3}, [(0, 1, EPSILON), (1, 3, "b"), (0, 2, "a"), (2, 3, "a")])
    assert nfa.accepted_word() == ("b",)


def test_universal_nfa_without_determinization():
    nfa = _nfa({0}, {0}, [(0, 0, "a"), (0, 0, "b")])
    assert nfa.is_universal()
    assert not nfa.is_universal(alphabet=frozenset({"a", "b", "c"}))


def test_includes_detects_missing_word():
    sigma_star = _nfa({0}, {0}, [(0, 0, "a"), (0, 0, "b")])
    only_a = _nfa({0}, {0}, [(0, 0, "a")])
    assert sigma_star.includes(only_a)
    assert not only_a.includes(sigma_star)


def test_dfa_methods():
    dfa = DFA(input_alphabet=frozenset(ALPHABET), initial_states=frozenset({0}), accepting_states=frozenset({1}))
    dfa.graph.add_state(0)
    dfa.graph.add_state(1)
    dfa.add_transition(0, 1, "a")
    dfa.add_transition(1, 1, "a")
    assert dfa.accepted_word() == ("a",)
    assert not dfa.is_universal()
    assert dfa.includes(_nfa({0}, {1}, [(0, 1, "a"), (1, 1, "a")]))


@pytest.mark.parametrize("states", [3, 4])
def test_universality_exponential_witness(states):
    """``(a|b)* a (a|b)^{k}`` complements need words of length ``k + 1`` to reject."""
    edges = [(0, 0, "a"), (0, 0, "b"), (0, 1, "a")]
    edges += [(i, i + 1, symbol) for i in range(1, states) for symbol in ALPHABET]
    nfa = _nfa({0}, {states}, edges)
    assert not nfa.is_universal()
    assert nfa.union(nfa.complement()).is_universal()


@settings(max_examples=150, deadline=None)
@given(small_automata)
def test_is_empty_matches_brute_force(aut):
    # With at most 4 states a non-empty language has a word of length < 4.
    assert aut.is_empty() == (not _language(aut, max_length=4))


@settings(max_examples=150, deadline=None)
@given(small_automata)
def test_accepted_word_is_accepted_and_shortest(aut):
    word = aut.accepted_word()
    language = _language(aut)
    if word is None:
        assert not language
        return
    assert aut.recognizes(word)
    assert len(word) == min(len(w) for w in language)


@settings(max_examples=150, deadline=None)
@given(small_automata)
def test_is_universal_matches_complement_and_brute_force(aut):
    universal = aut.is_universal()
    assert universal == aut.complement().is_empty()
    assert universal == (len(_language(aut)) == len(list(_words())))
    assert universal == aut.minimize(alphabet=frozenset(ALPHABET)).is_universal()


@settings(max_examples=150, deadline=None)
@given(small_automata, small_automata)
def test_includes_matches_brute_force_and_difference(left, right):
    included = left.includes(right)
    assert included == right.difference(left, alphabet=frozenset(ALPHABET)).is_empty()
    assert included == (_language(right) <= _language(left))


@settings(max_examples=100, deadline=None)
@given(small_automata, small_automata, small_automata)
def test_includes_is_a_preorder(a, b, c):
    assert a.includes(a)
    if a.includes(b) and b.includes(c):
        assert a.includes(c)


@settings(max_examples=100, deadline=None)
@given(small_automata, small_automata)
def test_union_includes_both_operands(a, b):
    union = a.union(b)
    assert union.includes(a)
    assert union.includes(b)
