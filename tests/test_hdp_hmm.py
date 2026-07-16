"""Tests for the weak-limit (sticky) HDP-HMM sampler."""

from __future__ import annotations

import numpy as np
import pytest

from sofic.generators.moore import MooreHMM
from sofic.inference.bayesian import HDPHMMPosterior, infer_hdp_hmm
from sofic.inference.bayesian.counts import BayesianInferenceError


def _two_state_moore() -> MooreHMM:
    machine = MooreHMM(observation_alphabet=frozenset({0, 1}), initial_distribution={"A": 0.5, "B": 0.5})
    machine.graph.add_state("A")
    machine.graph.add_state("B")
    machine.set_emission_distribution("A", {0: 0.92, 1: 0.08})
    machine.set_emission_distribution("B", {0: 0.08, 1: 0.92})
    machine.add_transition("A", "A", 0.85)
    machine.add_transition("A", "B", 0.15)
    machine.add_transition("B", "B", 0.85)
    machine.add_transition("B", "A", 0.15)
    machine.validate()
    return machine


def _periodic(period, n_seq, seq_len, seed):
    rng = np.random.default_rng(seed)
    p = len(period)
    return [[period[(int(rng.integers(0, p)) + t) % p] for t in range(seq_len)] for _ in range(n_seq)]


# ---------------------------------------------------------------------- recovery


def test_hdp_recovers_periodic_state_count():
    sequences = _periodic([0, 1, 2], n_seq=6, seq_len=120, seed=2)
    posterior = infer_hdp_hmm(sequences, max_states=10, iterations=200, burn_in=100, thin=5, rng=2)
    assert posterior.map_state_count() == 3
    assert posterior.state_count_posterior()[3] >= 0.35


def test_hdp_sticky_recovers_two_state():
    truth = _two_state_moore()
    srng = np.random.default_rng(2)
    sequences = [truth.sample(300, rng=srng)[0] for _ in range(6)]
    posterior = infer_hdp_hmm(sequences, max_states=12, kappa=30.0, iterations=250, burn_in=120, thin=5, rng=2)
    assert posterior.map_state_count() == 2


def test_hdp_stickiness_does_not_increase_states():
    truth = _two_state_moore()
    srng = np.random.default_rng(4)
    sequences = [truth.sample(300, rng=srng)[0] for _ in range(6)]
    plain = infer_hdp_hmm(sequences, max_states=12, kappa=0.0, iterations=200, burn_in=100, thin=5, rng=4)
    sticky = infer_hdp_hmm(sequences, max_states=12, kappa=30.0, iterations=200, burn_in=100, thin=5, rng=4)
    assert sticky.map_state_count() <= plain.map_state_count()


# ----------------------------------------------------------------- posterior API


def _small_posterior(seed=0) -> HDPHMMPosterior:
    sequences = _periodic([0, 1], n_seq=4, seq_len=40, seed=seed)
    return infer_hdp_hmm(sequences, max_states=6, iterations=60, burn_in=30, thin=3, rng=seed)


def test_hdp_posterior_is_a_distribution():
    posterior = _small_posterior()
    dist = posterior.state_count_posterior()
    assert dist
    assert all(count >= 1 for count in dist)
    assert sum(dist.values()) == pytest.approx(1.0)
    assert len(posterior.samples) == len(posterior.state_counts) == len(posterior.log_likelihoods)


def test_hdp_samples_are_valid_moore_hmms():
    posterior = _small_posterior()
    for sample in posterior.samples:
        assert isinstance(sample, MooreHMM)
        sample.validate()
        assert sample.observation_alphabet == frozenset({0, 1})
        assert np.isfinite(sample.log_likelihood([0, 1, 0, 1]))


def test_hdp_best_sample_has_max_loglik():
    posterior = _small_posterior()
    best = posterior.best_sample()
    assert best is posterior.samples[int(np.argmax(posterior.log_likelihoods))]


def test_hdp_accepts_single_flat_sequence():
    rng = np.random.default_rng(0)
    sequence = rng.integers(0, 2, size=120).tolist()
    posterior = infer_hdp_hmm(sequence, max_states=6, iterations=60, burn_in=30, thin=3, rng=0)
    assert posterior.samples
    assert posterior.alphabet == (0, 1)


# ------------------------------------------------------------------- validation


def test_hdp_rejects_empty_input():
    with pytest.raises(BayesianInferenceError):
        infer_hdp_hmm([])


def test_hdp_rejects_bad_hyperparameters():
    sequences = _periodic([0, 1], 2, 20, seed=0)
    with pytest.raises(BayesianInferenceError):
        infer_hdp_hmm(sequences, alpha=0.0)
    with pytest.raises(BayesianInferenceError):
        infer_hdp_hmm(sequences, eta=-1.0)
    with pytest.raises(BayesianInferenceError):
        infer_hdp_hmm(sequences, max_states=0)


def test_hdp_rejects_burn_in_exceeding_iterations():
    sequences = _periodic([0, 1], 2, 20, seed=0)
    with pytest.raises(BayesianInferenceError):
        infer_hdp_hmm(sequences, iterations=50, burn_in=50)
