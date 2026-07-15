"""Tests for Moore HMM."""

import pytest

from sofic.exceptions import StochasticValidationError
from sofic.generators.moore import MooreHMM
from sofic.graph import ATTR_EMISSION_DIST, ATTR_PROB


def _moore_hmm() -> MooreHMM:
    hmm = MooreHMM(
        initial_distribution={"q0": 1.0},
        observation_alphabet=frozenset({"0", "1"}),
    )
    hmm.graph.add_state("q0")
    hmm.set_emission_distribution("q0", {"0": 0.7, "1": 0.3})
    hmm.add_transition("q0", "q0", 1.0)
    return hmm


def test_validate():
    _moore_hmm().validate()


def test_builder_methods_set_emissions_and_probability():
    hmm = _moore_hmm()
    assert hmm.graph.state_attrs("q0")[ATTR_EMISSION_DIST] == {"0": 0.7, "1": 0.3}
    edge = next(hmm.transitions())
    assert edge.data[ATTR_PROB] == pytest.approx(1.0)


def test_bad_emission_dist():
    hmm = MooreHMM(
        initial_distribution={"q0": 1.0},
        observation_alphabet=frozenset({"0"}),
    )
    hmm.graph.add_state("q0")
    hmm.set_emission_distribution("q0", {"0": 0.5})
    with pytest.raises(StochasticValidationError):
        hmm.validate()
