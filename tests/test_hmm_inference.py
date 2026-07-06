"""Tests for HMM inference (forward/backward/Viterbi/sampling)."""

from __future__ import annotations

import numpy as np
import pytest

from pensive.examples import fair_coin
from pensive.generators.hmm_inference import backward, forward, log_likelihood, sample, viterbi
from pensive.generators.mealy import MealyHMM
from pensive.graph import ATTR_EMISSION, ATTR_PROB


def test_forward_coin_initial_and_likelihood():
    coin = fair_coin()
    observations = ["0", "1", "0"]
    alpha = forward(coin, observations)
    assert alpha.shape == (4, 1)
    assert alpha[0].sum() == pytest.approx(1.0, abs=1e-9)
    assert alpha[-1].sum() == pytest.approx(np.exp(log_likelihood(coin, observations)), abs=1e-9)


def test_backward_coin():
    coin = fair_coin()
    beta = backward(coin, ["0", "1"])
    assert beta.shape == (3, 1)
    assert beta[-1].sum() == pytest.approx(1.0, abs=1e-9)


def test_log_likelihood_coin():
    coin = fair_coin()
    ll = log_likelihood(coin, ["0", "1", "0", "1"])
    assert np.isfinite(ll)
    assert ll < 0.0


def test_viterbi_coin_constant_state():
    coin = fair_coin()
    path = viterbi(coin, ["0", "1", "0"])
    assert path == ["A", "A", "A"]


def test_viterbi_impossible_observation_has_no_path():
    coin = fair_coin()
    assert log_likelihood(coin, ["2"]) == float("-inf")
    assert viterbi(coin, ["2"]) == []


def test_stationary_distribution_periodic_hmm_is_invariant():
    hmm = MealyHMM(
        initial_distribution={"A": 1.0},
        observation_alphabet=frozenset({"0", "1"}),
    )
    hmm.graph.add_state("A")
    hmm.graph.add_state("B")
    hmm.graph.add_transition("A", "B", **{ATTR_PROB: 1.0, ATTR_EMISSION: "0"})
    hmm.graph.add_transition("B", "A", **{ATTR_PROB: 1.0, ATTR_EMISSION: "1"})

    pi = hmm.stationary_distribution()
    assert pi == pytest.approx([0.5, 0.5], abs=1e-12)
    transition = np.array([[0.0, 1.0], [1.0, 0.0]])
    assert pi @ transition == pytest.approx(pi, abs=1e-12)


def test_sample_coin_length():
    coin = fair_coin()
    rng = np.random.default_rng(0)
    observations, states = sample(coin, 20, rng=rng)
    assert len(observations) == 20
    assert len(states) == 20
    assert all(symbol in {"0", "1"} for symbol in observations)


def test_log_likelihood_long_sequence_stays_finite():
    """The scaled forward recursion must not underflow to -inf on long sequences."""
    coin = fair_coin()
    observations = ["0", "1"] * 1500
    ll = log_likelihood(coin, observations)
    assert np.isfinite(ll)
    assert ll == pytest.approx(-3000 * np.log(2), rel=1e-9)


def test_forward_scaled_rows_are_normalized():
    coin = fair_coin()
    alpha = forward(coin, ["0", "1", "0"], scaled=True)
    assert alpha.shape == (4, 1)
    assert np.allclose(alpha.sum(axis=1), 1.0)
