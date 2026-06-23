"""Tests for Markov chains."""

import pytest

from pensive.exceptions import StochasticValidationError
from pensive.generators.markov import MarkovChain
from pensive.graph import ATTR_PROB


def _chain() -> MarkovChain:
    mc = MarkovChain(initial_distribution={"q0": 1.0})
    mc.graph.add_state("q0")
    mc.graph.add_state("q1")
    mc.add_transition("q0", "q0", 0.5)
    mc.add_transition("q0", "q1", 0.5)
    return mc


def test_validate():
    _chain().validate()


def test_add_transition_sets_probability():
    edge = next(_chain().transitions())
    assert edge.data[ATTR_PROB] == pytest.approx(0.5)


def test_bad_row_sums():
    mc = MarkovChain(initial_distribution={"q0": 1.0})
    mc.graph.add_state("q0")
    mc.add_transition("q0", "q0", 0.3)
    with pytest.raises(StochasticValidationError):
        mc.validate()


def test_stationary_distribution_periodic_chain():
    mc = MarkovChain(initial_distribution={"A": 1.0})
    mc.graph.add_state("A")
    mc.graph.add_state("B")
    mc.add_transition("A", "B", 1.0)
    mc.add_transition("B", "A", 1.0)

    pi = mc.stationary_distribution()
    assert pi == pytest.approx([0.5, 0.5], abs=1e-12)
