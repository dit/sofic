"""Finite-state channels driven one use at a time by dit's feedback codes."""

import math

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from sofic.examples.processes import binary_channel, ising_channel, trapdoor
from sofic.generators import FiniteStateChannel

dit_coding = pytest.importorskip("dit.coding")
if not hasattr(dit_coding, "horstein"):
    pytest.skip("dit.coding has no feedback codes", allow_module_level=True)

from dit.coding import binary_symmetric_channel, horstein  # noqa: E402

CHANNELS = [trapdoor, ising_channel]
LOG2_PHI = math.log2((1 + math.sqrt(5)) / 2)
NO_FEEDBACK_UPPER = 0.584  # Huleihel et al. (2021)

words = st.integers(1, 6).flatmap(
    lambda n: st.tuples(
        st.lists(st.sampled_from((0, 1)), min_size=n, max_size=n),
        st.lists(st.sampled_from((0, 1)), min_size=n, max_size=n),
    )
)


@pytest.mark.parametrize("factory", CHANNELS)
@settings(max_examples=40, deadline=None)
@given(pair=words)
def test_feedback_steps_reproduce_log_likelihood(factory, pair):
    channel = FiniteStateChannel(factory())
    codeword, received = pair
    belief, total = channel.feedback_initial(), 0.0
    for x, y in zip(codeword, received, strict=True):
        outcomes = {out: (p, b) for out, p, b in channel.feedback_step(belief, x)}
        assert sum(p for p, _ in outcomes.values()) == pytest.approx(1.0)
        if y not in outcomes:
            total = -math.inf
            break
        p, belief = outcomes[y]
        assert belief.sum() == pytest.approx(1.0)
        total += math.log(p)
    assert total == pytest.approx(channel.log_likelihood(codeword, received))


def test_one_state_channel_matches_bsc_under_feedback():
    code = horstein(3, n=7)
    expected = code.probability_of_error(binary_symmetric_channel(0.1), method="exact")
    assert code.probability_of_error(binary_channel(0.1, 0.1), method="exact") == pytest.approx(expected)


def test_horstein_exact_matches_montecarlo_on_trapdoor():
    code = horstein(2, n=6)
    exact = code.probability_of_error(trapdoor(), method="exact")
    sampled = code.probability_of_error(trapdoor(), method="montecarlo", samples=6000, prng=np.random.default_rng(0))
    assert sampled == pytest.approx(exact, abs=0.015)


def test_feedback_capacity_brackets_log2_phi():
    lower, upper = FiniteStateChannel(trapdoor()).feedback_capacity()
    assert lower - 1e-6 <= LOG2_PHI <= upper + 1e-6


def test_horstein_beats_the_no_feedback_capacity_of_the_trapdoor():
    """Feedback lifts the trapdoor's rate above every no-feedback code, but not above log2(phi)."""
    code = horstein(8, delta=1e-3)
    pe, uses = code.performance(trapdoor(), method="montecarlo", samples=400, prng=np.random.default_rng(1))
    rate = 8 / uses
    assert pe <= 0.01
    assert NO_FEEDBACK_UPPER < rate < LOG2_PHI
    lower, upper = code.capacity_gap(trapdoor(), method="montecarlo", samples=100, prng=np.random.default_rng(2))
    assert 0 < lower <= upper
