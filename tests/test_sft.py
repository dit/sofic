"""Tests for shifts of finite type."""

from pensive.graph import ATTR_SYMBOL, TransitionGraph
from pensive.shifts.sft import ShiftOfFiniteType


def test_from_presentation():
    g = TransitionGraph()
    g.add_state("s")
    g.add_transition("s", "s", **{ATTR_SYMBOL: "0"})
    sft = ShiftOfFiniteType.from_presentation(g, symbol_alphabet=frozenset({"0"}))
    sft.validate()


def test_from_forbidden_words():
    sft = ShiftOfFiniteType.from_forbidden_words({("1", "1")}, frozenset({"0", "1"}))
    sft.validate()
    factors = set(sft.factor_language(2))
    assert ("1", "1") not in factors


def test_forbidden_words_returns_defining_set():
    sft = ShiftOfFiniteType.from_forbidden_words({("1", "1")}, frozenset({"0", "1"}))
    assert sft.forbidden_words() == frozenset({("1", "1")})


def test_forbidden_words_allows_empty_defining_set():
    sft = ShiftOfFiniteType.from_forbidden_words(set(), frozenset({"0", "1"}))
    assert sft.forbidden_words() == frozenset()


def test_forbidden_words_by_length_from_presentation():
    sft = ShiftOfFiniteType.from_forbidden_words({("1", "1")}, frozenset({"0", "1"}))
    assert sft.forbidden_words(length=2) == frozenset({("1", "1")})


def test_words_of_length_aliases_factor_language():
    sft = ShiftOfFiniteType.from_forbidden_words({("1", "1")}, frozenset({"0", "1"}))
    assert set(sft.words_of_length(2)) == set(sft.factor_language(2))
