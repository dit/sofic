"""Tests for SoficShift."""

from sofic.graph import ATTR_SYMBOL
from sofic.shifts.sofic import SoficShift


def test_validate():
    shift = SoficShift(symbol_alphabet=frozenset({"0", "1"}))
    shift.graph.add_state("s")
    shift.add_transition("s", "s", "0")
    shift.validate()


def test_add_transition_sets_symbol():
    shift = SoficShift(symbol_alphabet=frozenset({"0"}))
    shift.graph.add_state("s")
    shift.add_transition("s", "s", "0")
    edge = next(shift.transitions())
    assert edge.data[ATTR_SYMBOL] == "0"


def test_networkx_round_trip():
    shift = SoficShift(symbol_alphabet=frozenset({"0"}))
    shift.graph.add_state("s")
    shift.add_transition("s", "s", "0")
    restored = SoficShift.from_networkx(shift.to_networkx(), symbol_alphabet=frozenset({"0"}))
    restored.validate()
