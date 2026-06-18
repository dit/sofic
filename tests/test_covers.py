"""Tests for Fischer and Krieger covers."""

import pytest

from pensive.graph import ATTR_SYMBOL
from pensive.shifts.covers import LeftFischerCover, LeftKriegerCover, RightFischerCover, RightKriegerCover
from pensive.shifts.sofic import SoficShift


def _golden_mean() -> SoficShift:
    shift = SoficShift(symbol_alphabet=frozenset({"0", "1"}))
    shift.graph.add_state("A")
    shift.graph.add_state("B")
    shift.graph.add_transition("A", "B", **{ATTR_SYMBOL: "1"})
    shift.graph.add_transition("B", "A", **{ATTR_SYMBOL: "0"})
    shift.graph.add_transition("B", "B", **{ATTR_SYMBOL: "1"})
    return shift


@pytest.mark.parametrize("cls", [LeftFischerCover, RightFischerCover])
def test_fischer_cover_from_sofic(cls):
    cover = cls.from_sofic(_golden_mean())
    cover.validate()
    assert len(list(cover.states())) >= 1


@pytest.mark.parametrize("cls", [LeftKriegerCover, RightKriegerCover])
def test_krieger_cover_not_implemented(cls):
    with pytest.raises(NotImplementedError, match="Krieger"):
        cls.from_sofic(_golden_mean())
