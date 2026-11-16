"""Tests for Mealy and Moore transducers."""

import numpy as np
import pytest

from sofic.automata.transducers import MealyMachine, MooreMachine
from sofic.exceptions import InfiniteTransductionError, SoficValidationError, StochasticValidationError
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


def _partial() -> MealyMachine:
    """Stochastic two-state channel; input ``b`` is missing in state ``t``."""
    m = MealyMachine(
        input_alphabet=frozenset({"a", "b"}),
        output_alphabet=frozenset({"x", "y"}),
        initial_states=frozenset({"s"}),
    )
    m.add_transition("s", "s", "a", "x", prob=0.25)
    m.add_transition("s", "t", "a", "y", prob=0.75)
    m.add_transition("s", "t", "b", "y", prob=1.0)
    m.add_transition("t", "s", "a", "x", prob=1.0)
    return m


def test_alphabets_by_state_and_completeness():
    m = _partial()
    inputs, outputs = m.alphabets(node=True)
    assert inputs == {"s": frozenset("ab"), "t": frozenset("a")}
    assert outputs == {"s": frozenset("xy"), "t": frozenset("x")}
    assert not m.is_complete()
    assert m.is_complete(frozenset("a"))


def test_complete_adds_a_reject_sink_only_when_needed():
    m = _partial()
    completed = m.complete()
    assert completed is not m
    assert completed.is_complete()
    assert "?" in set(completed.states()) and "?" in completed.output_alphabet
    assert completed.transduce(("b", "b")) == {("y", "?")}
    completed.validate_stochastic()
    already = completed.complete(copy=False)
    assert already is completed
    assert len(list(already.states())) == 3
    assert MealyMachine().complete().input_alphabet == frozenset()


def test_input_and_output_machines():
    m = _partial()
    inputs = m.input_machine()
    assert inputs.observation_alphabet == frozenset("ab")
    assert sum(t.data["prob"] for t in inputs.graph.out_transitions("s")) == pytest.approx(1.0)
    outputs = m.output_machine()
    assert outputs.observation_alphabet == frozenset("xy")
    built = _mealy().output_machine(build=True)
    assert len(list(built.states())) == 1


def test_labeled_transition_matrices_options():
    m = _partial()
    matrices = m.labeled_transition_matrices(["t", "s"])
    assert matrices[("a", "y")][1, 0] == pytest.approx(0.75)
    nan = m.labeled_transition_matrices(["s", "t"], mode="nan")
    assert np.isnan(nan[("b", "y")][1]).all()
    assert not np.isnan(nan[("b", "y")][0]).any()
    restricted = m.labeled_transition_matrices(["s"])
    assert restricted[("a", "x")][0, 0] == pytest.approx(0.25)
    with pytest.raises(ValueError, match="unknown states"):
        m.labeled_transition_matrices(["s", "nope"])
    with pytest.raises(ValueError, match="mode"):
        m.labeled_transition_matrices(mode="inf")


def test_validate_stochastic_rejects_bad_rows():
    m = _partial()
    m.validate_stochastic()
    m.add_transition("t", "t", "a", "y", prob=0.5)
    with pytest.raises(StochasticValidationError, match="sums to"):
        m.validate_stochastic()
    negative = _partial()
    negative.add_transition("t", "t", "b", "y", prob=-1.0)
    with pytest.raises(StochasticValidationError, match="negative"):
        negative.validate_stochastic()


def test_validate_rejects_symbols_outside_the_alphabets():
    bad_output = _partial()
    bad_output.add_transition("t", "t", "b", "z")
    with pytest.raises(SoficValidationError, match="output 'z'"):
        bad_output.validate()
    bad_input = _partial()
    bad_input.add_transition("t", "t", "c", "x")
    with pytest.raises(SoficValidationError, match="symbol 'c'"):
        bad_input.validate()
    missing_initial = _partial()
    missing_initial.initial_states = frozenset({"nowhere"})
    with pytest.raises(SoficValidationError, match="initial state"):
        missing_initial.validate()


def test_moore_outputs_and_validation():
    m = MooreMachine(input_alphabet=frozenset("a"), output_alphabet=frozenset("0"), initial_states=frozenset({0}))
    m.graph.add_state(0)
    m.add_transition(0, 0, "a", prob=1.0)
    m.set_output(0, "0")
    m.validate()
    assert m.transduce(("a",)) == {("0", "0")}
    m.set_output(0, "9")
    with pytest.raises(SoficValidationError, match="output '9'"):
        m.validate()
    m.set_output(0, "0")
    m.add_transition(0, 0, "b")
    with pytest.raises(SoficValidationError, match="symbol 'b'"):
        m.validate()


def test_serial_composition_of_binary_channels():
    from sofic.examples.processes import binary_channel

    composed = binary_channel(0.1, 0.2).compose(binary_channel(0.1, 0.1))
    matrices, _states = composed.labeled_transition_matrices()
    assert matrices[("0", "1")][0, 0] == pytest.approx(0.1 * 0.9 + 0.9 * 0.1)
    assert matrices[("1", "0")][0, 0] == pytest.approx(0.8 * 0.1 + 0.2 * 0.9)


def test_conversions_to_other_representations():
    from sofic.examples.processes import binary_channel

    channel = binary_channel(0.1, 0.2)
    assert channel.to_wfst().__class__.__name__ == "WeightedFiniteStateTransducer"
    assert channel.to_textile_system().__class__.__name__ == "TextileSystem"
    assert channel.to_product_alphabet_shift().__class__.__name__ == "ProductAlphabetShift"
    assert channel.__dit_channel__().__class__.__name__ == "FiniteStateChannel"
