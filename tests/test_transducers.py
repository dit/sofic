"""Tests for Mealy and Moore transducers."""

import pytest

from sofic.automata.transducers import MealyMachine, MooreMachine
from sofic.exceptions import InfiniteTransductionError
from sofic.graph import ATTR_OUTPUT, ATTR_SYMBOL, EPSILON


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


def test_mealy_epsilon_input_and_output():
    mealy = MealyMachine(
        input_alphabet=frozenset({"a"}),
        output_alphabet=frozenset({"p", "s"}),
        initial_states=frozenset({"q0"}),
    )
    for state in ("q0", "q1", "q2", "q3"):
        mealy.graph.add_state(state)
    mealy.add_transition("q0", "q1", EPSILON, "p")
    mealy.add_transition("q1", "q2", "a", EPSILON)
    mealy.add_transition("q2", "q3", EPSILON, "s")

    mealy.validate()
    assert mealy.transduce(("a",)) == {("p",), ("p", "s")}


def test_moore_epsilon_transition_and_output():
    moore = MooreMachine(
        input_alphabet=frozenset({"a"}),
        output_alphabet=frozenset({"0", "1"}),
        initial_states=frozenset({"q0"}),
    )
    moore.graph.add_state("q0", **{ATTR_OUTPUT: EPSILON})
    moore.graph.add_state("q1", **{ATTR_OUTPUT: "0"})
    moore.graph.add_state("q2", **{ATTR_OUTPUT: "1"})
    moore.add_transition("q0", "q1", EPSILON)
    moore.add_transition("q1", "q2", "a")

    moore.validate()
    assert moore.transduce(("a",)) == {("0", "1")}


def test_productive_epsilon_cycle_raises():
    mealy = MealyMachine(
        input_alphabet=frozenset(),
        output_alphabet=frozenset({"x"}),
        initial_states=frozenset({"q0"}),
    )
    mealy.graph.add_state("q0")
    mealy.add_transition("q0", "q0", EPSILON, "x")

    with pytest.raises(InfiniteTransductionError):
        mealy.transduce(())


def test_transducer_yaml_round_trip_preserves_epsilon():
    mealy = MealyMachine(
        input_alphabet=frozenset({"a"}),
        output_alphabet=frozenset({"1"}),
        initial_states=frozenset({"q0"}),
    )
    mealy.graph.add_state("q0")
    mealy.graph.add_state("q1")
    mealy.add_transition("q0", "q1", EPSILON, "1")
    mealy.add_transition("q1", "q1", "a", EPSILON)

    restored = MealyMachine.from_yaml(mealy.to_yaml())
    assert restored.transduce(("a",)) == {("1",)}
    assert any(t.data[ATTR_SYMBOL] is EPSILON for t in restored.transitions())
