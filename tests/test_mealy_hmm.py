"""Tests for Mealy HMM (joint edge masses)."""

import pytest

from pensive.exceptions import StochasticValidationError, UnifilarityError
from pensive.generators.epsilon_machine import EpsilonMachine
from pensive.generators.mealy import MealyHMM
from pensive.graph import ATTR_EMISSION, ATTR_PROB


def _mealy_hmm() -> MealyHMM:
    hmm = MealyHMM(
        initial_distribution={"q0": 1.0},
        observation_alphabet=frozenset({"0", "1"}),
    )
    hmm.graph.add_state("q0")
    hmm.graph.add_transition("q0", "q0", **{ATTR_PROB: 0.6, ATTR_EMISSION: "0"})
    hmm.graph.add_transition("q0", "q0", **{ATTR_PROB: 0.4, ATTR_EMISSION: "1"})
    return hmm


def test_validate_joint_masses():
    _mealy_hmm().validate()


def test_mixed_state_presentation_on_mealy_hmm():
    from pensive.generators.mixed_state import MixedStatePresentation

    msp = _mealy_hmm().mixed_state_presentation()
    assert isinstance(msp, MixedStatePresentation)
    assert len(list(msp.states())) >= 1


def test_bad_joint_sum():
    hmm = _mealy_hmm()
    hmm.graph.add_transition("q0", "q0", **{ATTR_PROB: 0.1, ATTR_EMISSION: "0"})
    with pytest.raises(StochasticValidationError):
        hmm.validate()


def test_unifilarity_check():
    eps = EpsilonMachine(
        initial_distribution={"q0": 1.0},
        observation_alphabet=frozenset({"0", "1"}),
    )
    eps.graph.add_state("q0")
    eps.graph.add_transition("q0", "q0", **{ATTR_PROB: 0.5, ATTR_EMISSION: "0"})
    eps.graph.add_transition("q0", "q0", **{ATTR_PROB: 0.5, ATTR_EMISSION: "1"})
    eps.validate()

    bad = EpsilonMachine(
        initial_distribution={"q0": 1.0},
        observation_alphabet=frozenset({"0"}),
    )
    bad.graph.add_state("q0")
    bad.graph.add_transition("q0", "q0", **{ATTR_PROB: 0.5, ATTR_EMISSION: "0"})
    bad.graph.add_transition("q0", "q0", **{ATTR_PROB: 0.5, ATTR_EMISSION: "0"})
    with pytest.raises(UnifilarityError):
        bad.validate()


def test_entropy_rate_requires_unifilar_presentation():
    hmm = MealyHMM(
        initial_distribution={"q0": 1.0},
        observation_alphabet=frozenset({"0", "1"}),
    )
    hmm.graph.add_state("q0")
    hmm.graph.add_state("q1")
    hmm.graph.add_transition("q0", "q0", **{ATTR_PROB: 0.5, ATTR_EMISSION: "0"})
    hmm.graph.add_transition("q0", "q1", **{ATTR_PROB: 0.5, ATTR_EMISSION: "0"})
    hmm.graph.add_transition("q1", "q1", **{ATTR_PROB: 1.0, ATTR_EMISSION: "1"})

    with pytest.raises(NotImplementedError, match="unifilar"):
        hmm.entropy_rate()
