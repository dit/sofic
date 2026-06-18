"""Tests for Mealy and Moore transducers."""

from pensive.automata.transducers import MealyMachine, MooreMachine
from pensive.graph import ATTR_OUTPUT, ATTR_SYMBOL


def _mealy() -> MealyMachine:
    m = MealyMachine(
        input_alphabet=frozenset({"a"}),
        output_alphabet=frozenset({"0", "1"}),
        initial_states=frozenset({"q0"}),
    )
    m.graph.add_state("q0")
    m.graph.add_transition("q0", "q0", **{ATTR_SYMBOL: "a", ATTR_OUTPUT: "1"})
    return m


def _moore() -> MooreMachine:
    m = MooreMachine(
        input_alphabet=frozenset({"a"}),
        output_alphabet=frozenset({"0", "1"}),
        initial_states=frozenset({"q0"}),
    )
    m.graph.add_state("q0", **{ATTR_OUTPUT: "0"})
    m.graph.add_transition("q0", "q0", **{ATTR_SYMBOL: "a"})
    return m


def test_mealy_validate():
    _mealy().validate()


def test_moore_validate():
    _moore().validate()


def test_mealy_transduce():
    mealy = _mealy()
    assert mealy.transduce(()) == {()}
    assert mealy.transduce(("a",)) == {("1",)}
    assert mealy.transduce(("a", "a")) == {("1", "1")}


def test_moore_transduce():
    moore = _moore()
    assert moore.transduce(()) == {("0",)}
    assert moore.transduce(("a",)) == {("0", "0")}
    assert moore.transduce(("a", "a")) == {("0", "0", "0")}


def _nondeterministic_mealy() -> MealyMachine:
    m = MealyMachine(
        input_alphabet=frozenset({"a"}),
        output_alphabet=frozenset({"0", "1"}),
        initial_states=frozenset({"q0"}),
    )
    m.graph.add_state("q0")
    m.graph.add_state("q1")
    m.graph.add_transition("q0", "q0", **{ATTR_SYMBOL: "a", ATTR_OUTPUT: "0"})
    m.graph.add_transition("q0", "q1", **{ATTR_SYMBOL: "a", ATTR_OUTPUT: "1"})
    m.graph.add_transition("q1", "q1", **{ATTR_SYMBOL: "a", ATTR_OUTPUT: "1"})
    return m


def test_mealy_nondeterministic_outputs():
    mealy = _nondeterministic_mealy()
    assert mealy.transduce(("a",)) == {("0",), ("1",)}
