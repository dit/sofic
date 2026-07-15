"""Tests for epsilon machine construction."""

from sofic.examples.epsilon_machines import ellison_fig9_forward, golden_mean
from sofic.generators.epsilon_machine import EpsilonMachine
from sofic.generators.mealy import MealyHMM
from sofic.generators.reversal import time_reverse_stochastic
from sofic.graph import ATTR_EMISSION, ATTR_PROB


def _unifilar_mealy() -> MealyHMM:
    hmm = MealyHMM(
        initial_distribution={"q0": 1.0},
        observation_alphabet=frozenset({"0", "1"}),
    )
    hmm.graph.add_state("q0")
    hmm.graph.add_transition("q0", "q0", **{ATTR_PROB: 0.5, ATTR_EMISSION: "0"})
    hmm.graph.add_transition("q0", "q0", **{ATTR_PROB: 0.5, ATTR_EMISSION: "1"})
    return hmm


def test_epsilon_machine_validate():
    eps = EpsilonMachine(
        initial_distribution={"q0": 1.0},
        observation_alphabet=frozenset({"0", "1"}),
    )
    eps.graph.add_state("q0")
    eps.graph.add_transition("q0", "q0", **{ATTR_PROB: 0.5, ATTR_EMISSION: "0"})
    eps.graph.add_transition("q0", "q0", **{ATTR_PROB: 0.5, ATTR_EMISSION: "1"})
    eps.validate()


def test_from_hmm():
    eps = EpsilonMachine.from_hmm(_unifilar_mealy())
    eps.validate()
    assert len(list(eps.states())) >= 1


def test_reverse_via_time_reversed_generator():
    rev = EpsilonMachine.from_hmm(_unifilar_mealy().reverse())
    rev.validate()


def test_from_hmm_merges_golden_mean_msp_to_two_states():
    msp = golden_mean(0.5).mixed_state_presentation()
    eps = EpsilonMachine.from_hmm(msp)
    eps.validate()
    assert len(list(eps.states())) == 2


def test_from_hmm_via_msp_on_nonunifilar_reverse():
    forward = ellison_fig9_forward()
    rev_hmm = time_reverse_stochastic(forward)
    direct = EpsilonMachine.from_hmm(rev_hmm)
    via_msp = EpsilonMachine.from_hmm(rev_hmm.mixed_state_presentation())
    direct.validate()
    via_msp.validate()
    assert len(list(direct.states())) == len(list(via_msp.states())) == 3


def test_row_normalized_presentation_fallback():
    from unittest.mock import patch

    from sofic.examples import golden_mean_forward
    from sofic.exceptions import UnifilarityError
    from sofic.generators.epsilon_machine import _row_normalized_presentation
    from sofic.generators.reversal import time_reverse_stochastic

    forward = golden_mean_forward(0.5)
    rev_hmm = time_reverse_stochastic(forward)
    with patch.object(EpsilonMachine, "from_hmm", side_effect=UnifilarityError("non-unifilar")):
        eps = EpsilonMachine.from_time_reversed(forward)
    eps.validate_stochastic()
    direct = _row_normalized_presentation(rev_hmm)
    direct.validate_stochastic()
    assert set(direct.states()) == set(eps.states())
