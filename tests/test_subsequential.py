"""Tests for subsequential and weighted finite-state transducers."""

import math

import pytest

from sofic.automata.subsequential import SubsequentialTransducer, WeightedFiniteStateTransducer
from sofic.examples.processes import BinaryChannel, BitFlip
from sofic.exceptions import SoficValidationError
from sofic.properties import is_sequential_transducer, is_subsequential_transducer


def _subsequential() -> SubsequentialTransducer:
    t = SubsequentialTransducer(
        input_alphabet=frozenset("ab"),
        output_alphabet=frozenset("xy"),
        initial_states=frozenset({"q0"}),
        final_output={"q0": ("y",)},
    )
    t.graph.add_state("q0")
    t.add_transition("q0", "q0", "a", "x")
    return t


def test_subsequential_appends_final_output():
    t = _subsequential()
    t.validate()
    assert t.transduce("aa") == {("x", "x", "y")}
    assert t.transduce("") == {("y",)}


def test_subsequential_predicates():
    t = _subsequential()
    assert is_sequential_transducer(t)
    assert is_subsequential_transducer(t)
    assert t.is_subsequential()


def test_subsequential_rejects_nondeterministic():
    t = _subsequential()
    t.add_transition("q0", "q0", "a", "y")  # second edge on same (state, input)
    with pytest.raises(SoficValidationError):
        t.validate()


def test_wfst_probability_weight():
    w = WeightedFiniteStateTransducer.from_transducer(BinaryChannel(0.1, 0.2), semiring="probability")
    assert w.weight(["0"], ["0"]) == pytest.approx(0.9)
    assert w.weight(["0"], ["1"]) == pytest.approx(0.1)
    assert w.weight(["0", "1"], ["0", "1"]) == pytest.approx(0.9 * 0.8)


def test_wfst_tropical_weight():
    w = WeightedFiniteStateTransducer.from_transducer(BinaryChannel(0.1, 0.2), semiring="tropical")
    assert w.weight(["0"], ["0"]) == pytest.approx(-math.log(0.9))
    assert math.isinf(w.weight(["0"], ["0", "0"]))


def test_wfst_bad_semiring():
    w = WeightedFiniteStateTransducer.from_transducer(BitFlip(), semiring="probability")
    w.semiring = "nonsense"
    with pytest.raises(SoficValidationError):
        w.validate()


def test_wfst_yaml_round_trip():
    w = WeightedFiniteStateTransducer.from_transducer(BinaryChannel(0.1, 0.2), semiring="tropical")
    restored = WeightedFiniteStateTransducer.from_yaml(w.to_yaml())
    assert restored.semiring == "tropical"
    assert restored.weight(["1"], ["1"]) == pytest.approx(w.weight(["1"], ["1"]))


def test_subsequential_yaml_round_trip():
    t = _subsequential()
    restored = SubsequentialTransducer.from_yaml(t.to_yaml())
    assert restored.final_output == {"q0": ("y",)}
    assert restored.transduce("a") == {("x", "y")}
