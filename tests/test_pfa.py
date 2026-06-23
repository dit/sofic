"""Tests for probabilistic finite automata."""

import pytest

from pensive.exceptions import StochasticValidationError
from pensive.generators.pfa import ProbabilisticFiniteAutomaton
from pensive.graph import ATTR_EMISSION, ATTR_PROB


def _pfa() -> ProbabilisticFiniteAutomaton:
    pfa = ProbabilisticFiniteAutomaton(
        initial_distribution={"q0": 1.0},
        output_alphabet=frozenset({"x"}),
    )
    pfa.graph.add_state("q0")
    pfa.add_transition("q0", "q0", "x", 1.0)
    return pfa


def test_validate():
    _pfa().validate()


def test_add_transition_sets_symbol_and_probability():
    edge = next(_pfa().transitions())
    assert edge.data[ATTR_EMISSION] == "x"
    assert edge.data[ATTR_PROB] == pytest.approx(1.0)


def test_negative_probability():
    pfa = _pfa()
    pfa.add_transition("q0", "q0", "x", -0.1)
    with pytest.raises(StochasticValidationError):
        pfa.validate()
