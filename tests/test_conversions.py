"""Tests for generator conversions."""

import pytest

from pensive.generators.mealy import MealyHMM
from pensive.generators.moore import MooreHMM
from pensive.generators.pfa import ProbabilisticFiniteAutomaton
from pensive.graph import ATTR_EMISSION, ATTR_EMISSION_DIST, ATTR_PROB


def test_moore_to_mealy():
    moore = MooreHMM(
        initial_distribution={"q0": 1.0},
        observation_alphabet=frozenset({"0"}),
    )
    moore.graph.add_state("q0")
    moore.graph.add_state("q0", **{ATTR_EMISSION_DIST: {"0": 1.0}})
    moore.graph.add_transition("q0", "q0", **{ATTR_PROB: 1.0})
    mealy = moore.to_mealy()
    mealy.validate()
    assert isinstance(mealy, MealyHMM)


def test_pfa_to_mealy_hmm():
    pfa = ProbabilisticFiniteAutomaton(
        initial_distribution={"q0": 1.0},
        output_alphabet=frozenset({"a"}),
    )
    pfa.graph.add_state("q0")
    pfa.graph.add_transition("q0", "q0", **{ATTR_PROB: 1.0, ATTR_EMISSION: "a"})
    hmm = pfa.to_mealy_hmm()
    hmm.validate()
