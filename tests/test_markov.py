"""Tests for Markov chains."""

import pytest

from pensive.exceptions import StochasticValidationError
from pensive.generators.markov import MarkovChain
from pensive.graph import ATTR_PROB


def _chain() -> MarkovChain:
    mc = MarkovChain(initial_distribution={"q0": 1.0})
    mc.graph.add_state("q0")
    mc.graph.add_state("q1")
    mc.graph.add_transition("q0", "q0", **{ATTR_PROB: 0.5})
    mc.graph.add_transition("q0", "q1", **{ATTR_PROB: 0.5})
    return mc


def test_validate():
    _chain().validate()


def test_bad_row_sums():
    mc = MarkovChain(initial_distribution={"q0": 1.0})
    mc.graph.add_state("q0")
    mc.graph.add_transition("q0", "q0", **{ATTR_PROB: 0.3})
    with pytest.raises(StochasticValidationError):
        mc.validate()
