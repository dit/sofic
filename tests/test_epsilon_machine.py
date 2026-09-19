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


def _explosive_forward() -> EpsilonMachine:
    """A three-state ε-machine with infinitely many retrodictive causal states.

    The support is the full binary shift, so there is a single atom, yet the
    reverse belief set never closes. Reading the future ``(01)^k`` multiplies the
    posterior ratio ``z / y`` by exactly ``9 / 7`` per block, so those futures have
    pairwise distinct posteriors for every ``k`` while all keeping full support.
    """
    spec = {
        "A": [("0", "A", 1 / 3), ("1", "B", 2 / 3)],
        "B": [("0", "A", 2 / 5), ("1", "C", 3 / 5)],
        "C": [("0", "B", 4 / 7), ("1", "A", 3 / 7)],
    }
    eps = EpsilonMachine(
        initial_distribution={"A": 1.0},
        observation_alphabet=frozenset({"0", "1"}),
    )
    for state in spec:
        eps.graph.add_state(state)
    for state, edges in spec.items():
        for symbol, target, prob in edges:
            eps.graph.add_transition(state, target, **{ATTR_PROB: prob, ATTR_EMISSION: symbol})
    return eps


def test_explosive_reverse_belief_set_does_not_close():
    import pytest

    from sofic.exceptions import MixedStateExplosionError, StochasticValidationError
    from sofic.generators.mixed_state_construction import build_mixed_state_presentation

    forward = _explosive_forward()
    forward.validate()
    assert forward.is_unifilar()

    rev_hmm = time_reverse_stochastic(forward)
    for cap in (32, 128):
        with pytest.raises(MixedStateExplosionError):
            build_mixed_state_presentation(rev_hmm, max_states=cap)

    # Still a StochasticValidationError, so existing handlers keep working.
    assert issubclass(MixedStateExplosionError, StochasticValidationError)


def test_reverse_is_finite_decides_explosion():
    from sofic.examples import golden_mean_forward

    assert not _explosive_forward().reverse_is_finite()
    assert golden_mean_forward(0.5).reverse_is_finite()
    assert ellison_fig9_forward().reverse_is_finite()


def test_reverse_is_finite_sees_parallel_pair_edges():
    """Regression: two symbols may carry a pair to the same successor.

    Here ``(B, C)`` reaches ``(C, A)`` under both ``0`` and ``1`` with different
    ratios. Collapsing those parallel edges hides the non-unit cycle and reports
    this explosive machine as finite.
    """
    spec = {
        "A": [("0", "B", 1 / 3), ("1", "A", 2 / 3)],
        "B": [("0", "C", 2 / 5), ("1", "C", 3 / 5)],
        "C": [("0", "A", 4 / 7), ("1", "A", 3 / 7)],
    }
    eps = EpsilonMachine(
        initial_distribution={"A": 1.0},
        observation_alphabet=frozenset({"0", "1"}),
    )
    for state in spec:
        eps.graph.add_state(state)
    for state, edges in spec.items():
        for symbol, target, prob in edges:
            eps.graph.add_transition(state, target, **{ATTR_PROB: prob, ATTR_EMISSION: symbol})

    assert not eps.reverse_is_finite()


def test_reverse_is_finite_is_not_a_structural_condition():
    """Same transition structure as the witness, but finite for these probabilities.

    The pair-graph cycle weight happens to be one here, so the belief set closes.
    A structure-only test cannot distinguish these two machines.
    """
    finite = EpsilonMachine(
        initial_distribution={"A": 1.0},
        observation_alphabet=frozenset({"0", "1"}),
    )
    spec = {
        "A": [("0", "A", 1 / 3), ("1", "B", 2 / 3)],
        "B": [("0", "A", 1 / 2), ("1", "C", 1 / 2)],
        "C": [("0", "B", 2 / 3), ("1", "A", 1 / 3)],
    }
    for state in spec:
        finite.graph.add_state(state)
    for state, edges in spec.items():
        for symbol, target, prob in edges:
            finite.graph.add_transition(state, target, **{ATTR_PROB: prob, ATTR_EMISSION: symbol})

    assert finite.reverse_is_finite()
    assert not _explosive_forward().reverse_is_finite()

    # and the prediction is borne out by actually building it
    reverse = EpsilonMachine.from_time_reversed(finite)
    reverse.validate()
    assert len(list(reverse.states())) >= 1


def test_from_time_reversed_propagates_explosion_instead_of_degrading():
    """Regression: the row-normalized fallback must not mask an infinite reverse.

    Falling back here returns a non-unifilar machine whose state entropy equals
    the forward statistical complexity, so ``causal_irreversibility`` silently
    reports ``0.0`` for a process whose reverse ε-machine is infinite.
    """
    from unittest.mock import patch

    import pytest

    from sofic.exceptions import MixedStateExplosionError

    forward = _explosive_forward()
    with (
        patch.object(EpsilonMachine, "from_hmm", side_effect=MixedStateExplosionError("did not close")),
        pytest.raises(MixedStateExplosionError),
    ):
        EpsilonMachine.from_time_reversed(forward)
