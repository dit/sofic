"""Tests for n-machine construction."""

import pytest

from sofic.exceptions import QuasiStochasticValidationError
from sofic.generators.epsilon_machine import EpsilonMachine
from sofic.generators.nmachine import NMachine
from sofic.graph import ATTR_EMISSION, ATTR_PROB, ATTR_QUASIPROB


def _nmachine() -> NMachine:
    nm = NMachine(
        initial_quasidistribution={"q0": 1.0},
        observation_alphabet=frozenset({"0", "1"}),
    )
    nm.graph.add_state("q0")
    nm.graph.add_transition("q0", "q0", **{ATTR_QUASIPROB: 0.6, ATTR_EMISSION: "0"})
    nm.graph.add_transition("q0", "q0", **{ATTR_QUASIPROB: 0.4, ATTR_EMISSION: "1"})
    return nm


def _epsilon() -> EpsilonMachine:
    eps = EpsilonMachine(
        initial_distribution={"q0": 1.0},
        observation_alphabet=frozenset({"0", "1"}),
    )
    eps.graph.add_state("q0")
    eps.graph.add_transition("q0", "q0", **{ATTR_PROB: 0.5, ATTR_EMISSION: "0"})
    eps.graph.add_transition("q0", "q0", **{ATTR_PROB: 0.5, ATTR_EMISSION: "1"})
    return eps


def test_validate_quasistochastic_rows():
    _nmachine().validate()


def test_negative_output_marginal():
    nm = NMachine(
        initial_quasidistribution={"q0": 1.0},
        observation_alphabet=frozenset({"0", "1"}),
    )
    nm.graph.add_state("q0")
    nm.graph.add_transition("q0", "q0", **{ATTR_QUASIPROB: 1.2, ATTR_EMISSION: "0"})
    nm.graph.add_transition("q0", "q0", **{ATTR_QUASIPROB: -0.2, ATTR_EMISSION: "1"})
    with pytest.raises(QuasiStochasticValidationError):
        nm.validate()


def test_to_quasi_realization():
    qr = _nmachine().to_quasi_realization()
    assert qr.pi.sum() == pytest.approx(1.0)


def test_from_epsilon_machine():
    nm = NMachine.from_epsilon_machine(_epsilon())
    nm.validate()
    assert nm.word_probability(("0",)) > 0.0


def test_coarse_grained_distribution():
    eps = _epsilon()
    nm = NMachine.from_epsilon_machine(eps, splits={"q0": 2})
    coarse = nm.coarse_grained_distribution()
    assert "q0" in coarse
