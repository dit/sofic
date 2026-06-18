"""Tests for HMM inference (forward/backward/Viterbi/sampling)."""

from __future__ import annotations

import numpy as np
import pytest

from pensive.examples import fair_coin
from pensive.generators.hmm_inference import backward, forward, log_likelihood, sample, viterbi


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


def test_sample_coin_length():
    coin = fair_coin()
    rng = np.random.default_rng(0)
    observations, states = sample(coin, 20, rng=rng)
    assert len(observations) == 20
    assert len(states) == 20
    assert all(symbol in {"0", "1"} for symbol in observations)
