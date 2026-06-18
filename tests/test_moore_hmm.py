"""Tests for Moore HMM."""

import pytest

from pensive.exceptions import StochasticValidationError
from pensive.generators.moore import MooreHMM
from pensive.graph import ATTR_EMISSION_DIST, ATTR_PROB


def _moore_hmm() -> MooreHMM:
    hmm = MooreHMM(
        initial_distribution={"q0": 1.0},
        observation_alphabet=frozenset({"0", "1"}),
    )
    hmm.graph.add_state("q0", **{ATTR_EMISSION_DIST: {"0": 0.7, "1": 0.3}})
    hmm.graph.add_transition("q0", "q0", **{ATTR_PROB: 1.0})
    return hmm


def test_validate():
    _moore_hmm().validate()


def test_bad_emission_dist():
    hmm = MooreHMM(
        initial_distribution={"q0": 1.0},
        observation_alphabet=frozenset({"0"}),
    )
    hmm.graph.add_state("q0", **{ATTR_EMISSION_DIST: {"0": 0.5}})
    with pytest.raises(StochasticValidationError):
        hmm.validate()
