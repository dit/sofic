"""Tests for SoficShift."""

from pensive.graph import ATTR_SYMBOL
from pensive.shifts.sofic import SoficShift


def test_validate():
    shift = SoficShift(symbol_alphabet=frozenset({"0", "1"}))
    shift.graph.add_state("s")
    shift.graph.add_transition("s", "s", **{ATTR_SYMBOL: "0"})
    shift.validate()


def test_networkx_round_trip():
    shift = SoficShift(symbol_alphabet=frozenset({"0"}))
    shift.graph.add_state("s")
    shift.graph.add_transition("s", "s", **{ATTR_SYMBOL: "0"})
    restored = SoficShift.from_networkx(shift.to_networkx(), symbol_alphabet=frozenset({"0"}))
    restored.validate()
