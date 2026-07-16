"""Tests for HMM inference (forward/backward/Viterbi/sampling)."""

from __future__ import annotations

import numpy as np
import pytest

from sofic.examples import fair_coin, golden_mean
from sofic.generators.hmm_inference import (
    _emission_transition_tensors,
    _forward_scaled,
    backward,
    baum_welch,
    forward,
    free_parameter_labels,
    log_likelihood,
    observed_information,
    sample,
    score,
    smooth,
    standard_errors,
    two_slice_marginals,
    viterbi,
)
from sofic.generators.mealy import MealyHMM
from sofic.graph import ATTR_EMISSION, ATTR_PROB


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


def _iid_source(probs: dict) -> MealyHMM:
    """Single-state IID emitter over integer symbols with the given emission law."""
    hmm = MealyHMM(
        initial_distribution={"A": 1.0},
        observation_alphabet=frozenset(probs),
    )
    hmm.graph.add_state("A")
    for symbol, prob in probs.items():
        hmm.add_transition("A", "A", symbol, prob)
    hmm.validate()
    return hmm


# --- Smoothing ------------------------------------------------------------


def test_smooth_rows_sum_to_one_and_match_unscaled_forward_backward():
    gm = golden_mean(0.4)
    obs = [0, 1, 0, 0, 1, 0]
    gamma = smooth(gm, obs)
    assert gamma.shape == (len(obs) + 1, 2)
    assert np.allclose(gamma.sum(axis=1), 1.0)

    alpha = forward(gm, obs)
    beta = backward(gm, obs)
    reference = alpha * beta
    reference = reference / reference.sum(axis=1, keepdims=True)
    assert np.allclose(gamma, reference)


def test_two_slice_marginals_sum_to_one_and_marginalize_to_gamma():
    gm = golden_mean(0.4)
    obs = [0, 1, 0, 0, 1, 0]
    xi = two_slice_marginals(gm, obs)
    gamma = smooth(gm, obs)
    assert xi.shape == (len(obs), 2, 2)
    assert np.allclose(xi.sum(axis=(1, 2)), 1.0)
    # Marginalizing X_{t+1} out of xi recovers gamma[t] for t < n.
    assert np.allclose(xi.sum(axis=2), gamma[:-1])


def test_smooth_impossible_sequence_is_zero():
    coin = fair_coin()
    gamma = smooth(coin, ["2"])
    assert np.all(gamma == 0.0)


# --- Baum-Welch EM --------------------------------------------------------


def test_baum_welch_loglik_is_monotone_and_recovers_parameters():
    rng = np.random.default_rng(0)
    truth = golden_mean(0.3)
    sequences = [sample(truth, 400, rng=rng)[0] for _ in range(8)]

    fitted, trace = baum_welch(golden_mean(0.6), sequences, max_iter=200, tol=1e-10)

    assert len(trace) >= 2
    assert np.all(np.diff(trace) >= -1e-9)  # EM never decreases the log-likelihood

    fitted_p = {
        (tr.source, tr.target, tr.data[ATTR_EMISSION]): float(tr.data[ATTR_PROB]) for tr in fitted.transitions()
    }
    assert fitted_p[("A", "A", 0)] == pytest.approx(0.3, abs=0.03)
    assert fitted_p[("B", "A", 0)] == pytest.approx(1.0, abs=1e-9)


def test_baum_welch_preserves_topology_and_returns_mealy():
    rng = np.random.default_rng(1)
    data = sample(golden_mean(0.5), 300, rng=rng)[0]
    start = golden_mean(0.6)
    fitted, _trace = baum_welch(start, data)

    assert type(fitted) is MealyHMM
    start_edges = {(tr.source, tr.data[ATTR_EMISSION], tr.target) for tr in start.transitions()}
    fitted_edges = {(tr.source, tr.data[ATTR_EMISSION], tr.target) for tr in fitted.transitions()}
    assert fitted_edges <= start_edges  # no new edges introduced
    fitted.validate()  # row sums remain stochastic


def test_baum_welch_accepts_single_sequence():
    rng = np.random.default_rng(2)
    data = sample(golden_mean(0.35), 500, rng=rng)[0]
    fitted, trace = baum_welch(golden_mean(0.6), data)
    assert np.all(np.diff(trace) >= -1e-9)
    assert log_likelihood(fitted, data) >= log_likelihood(golden_mean(0.6), data)


# --- Score and observed information ---------------------------------------


def test_score_matches_finite_difference_gradient():
    gm = golden_mean(0.4)
    obs = [0, 1, 0, 0, 1, 0, 1, 0]
    pi, joint = _emission_transition_tensors(gm)
    idx = gm.to_mealy().reindex()
    a, b = idx.index("A"), idx.index("B")

    def loglik_entry(symbol: int, i: int, j: int, value: float) -> float:
        perturbed = {sym: matrix.copy() for sym, matrix in joint.items()}
        perturbed[symbol][i, j] = value
        _alpha, log_scales = _forward_scaled(pi, perturbed, list(obs))
        return float(log_scales.sum())

    analytic = score(gm, obs)
    h = 1e-6
    for (symbol, i, j), key in (((0, a, a), ("A", 0, "A")), ((1, a, b), ("A", 1, "B")), ((0, b, a), ("B", 0, "A"))):
        base = float(joint[symbol][i, j])
        numeric = (loglik_entry(symbol, i, j, base + h) - loglik_entry(symbol, i, j, base - h)) / (2 * h)
        assert analytic[key] == pytest.approx(numeric, rel=1e-4, abs=1e-4)


def test_observed_information_matches_numeric_hessian_scalar():
    gm = golden_mean(0.5)
    obs = [0, 1, 0, 0, 1, 0, 1, 0]
    pi, joint = _emission_transition_tensors(gm)
    idx = gm.to_mealy().reindex()
    a, b = idx.index("A"), idx.index("B")

    def loglik_theta(theta: float) -> float:
        perturbed = {sym: matrix.copy() for sym, matrix in joint.items()}
        perturbed[0][a, a] = theta
        perturbed[1][a, b] = 1.0 - theta
        _alpha, log_scales = _forward_scaled(pi, perturbed, list(obs))
        return float(log_scales.sum())

    assert free_parameter_labels(gm) == [("A", 0, "A")]
    theta0 = 0.5
    h = 1e-5
    numeric = -(loglik_theta(theta0 + h) - 2 * loglik_theta(theta0) + loglik_theta(theta0 - h)) / (h * h)

    info = observed_information(gm, obs)
    assert info.shape == (1, 1)
    assert info[0, 0] == pytest.approx(numeric, rel=1e-3)


def test_observed_information_multi_parameter_symmetric_and_matches_hessian():
    hmm = _iid_source({0: 0.2, 1: 0.3, 2: 0.5})
    rng = np.random.default_rng(3)
    obs = sample(hmm, 200, rng=rng)[0]

    labels = free_parameter_labels(hmm)
    assert labels == [("A", 0, "A"), ("A", 1, "A")]

    pi, joint = _emission_transition_tensors(hmm)

    def loglik_free(theta: np.ndarray) -> float:
        perturbed = {sym: matrix.copy() for sym, matrix in joint.items()}
        perturbed[0][0, 0] = theta[0]
        perturbed[1][0, 0] = theta[1]
        perturbed[2][0, 0] = 1.0 - theta[0] - theta[1]
        _alpha, log_scales = _forward_scaled(pi, perturbed, list(obs))
        return float(log_scales.sum())

    base = np.array([0.2, 0.3])
    h = 1e-5
    hessian = np.zeros((2, 2))
    for p in range(2):
        for q in range(2):
            step_p = np.zeros(2)
            step_q = np.zeros(2)
            step_p[p] = h
            step_q[q] = h
            hessian[p, q] = (
                loglik_free(base + step_p + step_q)
                - loglik_free(base + step_p - step_q)
                - loglik_free(base - step_p + step_q)
                + loglik_free(base - step_p - step_q)
            ) / (4 * h * h)
    numeric_info = -hessian

    info = observed_information(hmm, obs)
    assert info.shape == (2, 2)
    assert np.allclose(info, info.T)
    assert np.allclose(info, numeric_info, rtol=1e-2, atol=1e-2)


def test_observed_information_at_mle_is_positive_semidefinite():
    hmm = _iid_source({0: 0.2, 1: 0.3, 2: 0.5})
    rng = np.random.default_rng(4)
    obs = sample(hmm, 1000, rng=rng)[0]
    fitted, _trace = baum_welch(hmm, obs)
    info = observed_information(fitted, obs)
    eigenvalues = np.linalg.eigvalsh(info)
    assert np.all(eigenvalues >= -1e-6)


def test_standard_errors_are_finite_and_labeled():
    hmm = _iid_source({0: 0.2, 1: 0.3, 2: 0.5})
    rng = np.random.default_rng(5)
    obs = sample(hmm, 500, rng=rng)[0]
    errors = standard_errors(hmm, obs)
    assert set(errors) == {("A", 0, "A"), ("A", 1, "A")}
    assert all(np.isfinite(value) and value > 0.0 for value in errors.values())


def test_observed_information_empty_when_no_free_parameters():
    hmm = MealyHMM(
        initial_distribution={"A": 1.0},
        observation_alphabet=frozenset({"0", "1"}),
    )
    hmm.graph.add_state("A")
    hmm.graph.add_state("B")
    hmm.graph.add_transition("A", "B", **{ATTR_PROB: 1.0, ATTR_EMISSION: "0"})
    hmm.graph.add_transition("B", "A", **{ATTR_PROB: 1.0, ATTR_EMISSION: "1"})

    info = observed_information(hmm, ["0", "1", "0"])
    assert info.shape == (0, 0)
    assert standard_errors(hmm, ["0", "1", "0"]) == {}


def test_hmm_methods_delegate_to_inference_functions():
    gm = golden_mean(0.4)
    obs = [0, 1, 0, 0, 1]
    assert np.allclose(gm.smooth(obs), smooth(gm, obs))
    assert np.allclose(gm.two_slice_marginals(obs), two_slice_marginals(gm, obs))
    assert gm.score(obs) == score(gm, obs)
    assert np.allclose(gm.observed_information(obs), observed_information(gm, obs))
    _fitted, trace = gm.baum_welch(obs)
    assert trace
