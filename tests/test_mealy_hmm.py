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


def test_entropy_rate_hmm_requires_unifilar_but_method_dispatches():
    hmm = MealyHMM(
        initial_distribution={"q0": 1.0},
        observation_alphabet=frozenset({"0", "1"}),
    )
    hmm.graph.add_state("q0")
    hmm.graph.add_state("q1")
    hmm.add_transition("q0", "q0", "0", 0.5)
    hmm.add_transition("q0", "q1", "0", 0.5)
    hmm.add_transition("q1", "q1", "1", 1.0)

    from sofic.generators.measures import entropy_rate_hmm

    with pytest.raises(NotImplementedError, match="unifilar"):
        entropy_rate_hmm(hmm)
    assert hmm.entropy_rate() == pytest.approx(0.0)


def test_mealy_without_alphabet_or_initial_distribution():
    """Regression: symbols missing from the alphabet raised KeyError, and an empty
    initial distribution made sampling fail on NaN and every word probability 0."""
    import numpy as np

    hmm = MealyHMM()
    for source, target, prob in [("a", "a", 0.9), ("a", "b", 0.1), ("b", "a", 0.2), ("b", "b", 0.8)]:
        hmm.add_transition(source, target, target, prob)
    assert hmm.observation_alphabet == frozenset({"a", "b"})
    hmm.validate()
    symbols, _ = hmm.sample(50, rng=np.random.default_rng(0))
    assert len(symbols) == 50
    assert hmm.word_probability("ab") == pytest.approx(2 / 3 * 0.1)
    assert hmm.entropy_rate() == pytest.approx(0.5533064, abs=1e-6)


def test_entropy_with_mixed_int_and_tuple_state_labels():
    """Regression: dit could not sort labels mixing ``0`` and an encoded tuple."""
    hmm = MealyHMM(initial_distribution={0: 1.0}, observation_alphabet=frozenset({"0", "1"}))
    for state in (0, (1, 0)):
        hmm.graph.add_state(state)
    hmm.add_transition(0, (1, 0), "0", 0.5)
    hmm.add_transition(0, 0, "1", 0.5)
    hmm.add_transition((1, 0), 0, "1", 1.0)

    assert hmm.entropy_rate() == pytest.approx(2 / 3)
    assert hmm.state_entropy() == pytest.approx(0.9182958340544896)
