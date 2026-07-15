"""Tests for Mealy HMM (joint edge masses)."""

import pytest

from sofic.exceptions import StochasticValidationError, UnifilarityError
from sofic.generators.epsilon_machine import EpsilonMachine
from sofic.generators.mealy import MealyHMM
from sofic.graph import ATTR_EMISSION, ATTR_PROB


def _mealy_hmm() -> MealyHMM:
    hmm = MealyHMM(
        initial_distribution={"q0": 1.0},
        observation_alphabet=frozenset({"0", "1"}),
    )
    hmm.graph.add_state("q0")
    hmm.add_transition("q0", "q0", "0", 0.6)
    hmm.add_transition("q0", "q0", "1", 0.4)
    return hmm


def test_validate_joint_masses():
    _mealy_hmm().validate()


def test_add_transition_sets_symbol_and_probability():
    edge = next(_mealy_hmm().transitions())
    assert edge.data[ATTR_EMISSION] == "0"
    assert edge.data[ATTR_PROB] == pytest.approx(0.6)


def test_to_mealy_returns_self():
    hmm = _mealy_hmm()
    assert hmm.to_mealy() is hmm


def test_mixed_state_presentation_on_mealy_hmm():
    from sofic.generators.mixed_state import MixedStatePresentation

    msp = _mealy_hmm().mixed_state_presentation()
    assert isinstance(msp, MixedStatePresentation)
    assert len(list(msp.states())) >= 1


def test_bad_joint_sum():
    hmm = _mealy_hmm()
    hmm.add_transition("q0", "q0", "0", 0.1)
    with pytest.raises(StochasticValidationError):
        hmm.validate()


def test_unifilarity_check():
    eps = EpsilonMachine(
        initial_distribution={"q0": 1.0},
        observation_alphabet=frozenset({"0", "1"}),
    )
    eps.graph.add_state("q0")
    eps.add_transition("q0", "q0", "0", 0.5)
    eps.add_transition("q0", "q0", "1", 0.5)
    eps.validate()

    bad = EpsilonMachine(
        initial_distribution={"q0": 1.0},
        observation_alphabet=frozenset({"0"}),
    )
    bad.graph.add_state("q0")
    bad.add_transition("q0", "q0", "0", 0.5)
    bad.add_transition("q0", "q0", "0", 0.5)
    with pytest.raises(UnifilarityError):
        bad.validate()


def test_entropy_rate_requires_unifilar_presentation():
    hmm = MealyHMM(
        initial_distribution={"q0": 1.0},
        observation_alphabet=frozenset({"0", "1"}),
    )
    hmm.graph.add_state("q0")
    hmm.graph.add_state("q1")
    hmm.add_transition("q0", "q0", "0", 0.5)
    hmm.add_transition("q0", "q1", "0", 0.5)
    hmm.add_transition("q1", "q1", "1", 1.0)

    with pytest.raises(NotImplementedError, match="unifilar"):
        hmm.entropy_rate()
