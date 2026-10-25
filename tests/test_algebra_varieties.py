"""Tests for language varieties decided on syntactic monoids."""

import itertools

import pytest
from hypothesis import given
from hypothesis import strategies as st

from sofic.algebra import factor_monoid, syntactic_monoid
from sofic.algebra.varieties import (
    is_commutative_language,
    is_definite,
    is_group_language,
    is_l_trivial,
    is_locally_testable,
    is_piecewise_testable,
    is_r_trivial,
    is_reverse_definite,
)
from sofic.automata.dfa import DFA
from sofic.shifts.sofic import SoficShift

#: (regex, piecewise testable, locally testable, definite, reverse definite, group, commutative)
KNOWN = [
    ("(a|b)*a", False, True, True, False, False, False),
    ("a(a|b)*", False, True, False, True, False, False),
    ("(a|b)*ab(a|b)*", True, True, False, False, False, False),
    ("(ab)*", False, True, False, False, False, False),
    ("a*b*", True, True, False, False, False, False),
    ("a*ba*", True, False, False, False, False, True),
    ("((a|b)(a|b))*", False, False, False, False, True, True),
    ("(a|b)*", True, True, True, True, True, True),
]


@pytest.mark.parametrize(("regex", "pt", "lt", "definite", "reverse", "group", "commutative"), KNOWN)
def test_known_languages(regex, pt, lt, definite, reverse, group, commutative):
    assert is_piecewise_testable(regex, alphabet="ab") is pt
    assert is_locally_testable(regex, alphabet="ab") is lt
    assert is_definite(regex, alphabet="ab") is definite
    assert is_reverse_definite(regex, alphabet="ab") is reverse
    assert is_group_language(regex, alphabet="ab") is group
    assert is_commutative_language(regex, alphabet="ab") is commutative


def test_green_trivialities_bracket_piecewise_testability():
    assert is_l_trivial("(a|b)*a", alphabet="ab") and not is_r_trivial("(a|b)*a", alphabet="ab")
    assert is_r_trivial("a(a|b)*", alphabet="ab") and not is_l_trivial("a(a|b)*", alphabet="ab")


@st.composite
def complete_dfas(draw, max_states=4):
    n = draw(st.integers(1, max_states))
    dfa = DFA(
        input_alphabet=frozenset("ab"),
        initial_states=frozenset({0}),
        accepting_states=frozenset(draw(st.sets(st.integers(0, n - 1)))),
    )
    for q in range(n):
        dfa.graph.add_state(q)
    for q in range(n):
        for symbol in "ab":
            dfa.add_transition(q, draw(st.integers(0, n - 1)), symbol)
    return dfa


def _words(max_length):
    for length in range(max_length + 1):
        yield from itertools.product("ab", repeat=length)


@given(complete_dfas())
def test_definite_languages_depend_only_on_a_suffix(dfa):
    if not is_definite(dfa):
        return
    k = len(syntactic_monoid(dfa))
    for word in _words(k + 3):
        if len(word) >= k:
            assert dfa.recognizes(word) == dfa.recognizes(word[-k:])


@given(complete_dfas())
def test_reverse_definite_languages_depend_only_on_a_prefix(dfa):
    if not is_reverse_definite(dfa):
        return
    k = len(syntactic_monoid(dfa))
    for word in _words(k + 3):
        if len(word) >= k:
            assert dfa.recognizes(word) == dfa.recognizes(word[:k])


@given(complete_dfas())
def test_inclusions_between_varieties(dfa):
    if is_definite(dfa) or is_reverse_definite(dfa):
        assert is_locally_testable(dfa)
    if is_piecewise_testable(dfa):
        assert is_r_trivial(dfa) and is_l_trivial(dfa)


def shift(edges):
    s = SoficShift(symbol_alphabet=frozenset(symbol for _, _, symbol in edges))
    for source, target, symbol in edges:
        s.add_transition(source, target, symbol)
    return s


@pytest.mark.parametrize(
    "edges",
    [
        [("A", "A", "0"), ("A", "B", "1"), ("B", "A", "0")],
        [("A", "A", "0"), ("A", "A", "1")],
        [("A", "A", "0"), ("A", "B", "1"), ("B", "C", "1"), ("C", "A", "0")],
    ],
)
def test_shifts_of_finite_type_have_locally_testable_languages(edges):
    assert is_locally_testable(factor_monoid(shift(edges), monoid=False))


def test_the_even_shift_is_not_locally_testable():
    even = shift([("A", "A", "0"), ("A", "B", "1"), ("B", "A", "1")])
    assert not is_locally_testable(factor_monoid(even, monoid=False))
