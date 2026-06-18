"""Tests for shifts of finite type."""

import pytest

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
