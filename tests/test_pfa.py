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
    pfa.graph.add_transition("q0", "q0", **{ATTR_PROB: 1.0, ATTR_EMISSION: "x"})
    return pfa


def test_validate():
    _pfa().validate()


def test_negative_probability():
    pfa = _pfa()
    pfa.graph.add_transition("q0", "q0", **{ATTR_PROB: -0.1, ATTR_EMISSION: "x"})
    with pytest.raises(StochasticValidationError):
        pfa.validate()
