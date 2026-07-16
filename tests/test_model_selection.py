"""Tests for classical model-selection criteria."""

from __future__ import annotations

import numpy as np
import pytest

from sofic.examples import fair_coin, golden_mean
from sofic.generators.mealy import MealyHMM
from sofic.inference.bayesian import BayesianInferenceError, ModelComparisonEM
from sofic.inference.bayesian.epsilon import EpsilonMachinePosterior
from sofic.inference.model_selection import (
    compare_information_criteria,
    count_free_parameters,
    cross_validated_log_likelihood,
    information_criterion,
    rank_topological_epsilon_machines,
    score_model,
    waic,
    waic_epsilon_machine,
)


def _iid_binary(p: float = 0.5) -> MealyHMM:
    hmm = MealyHMM(initial_distribution={"A": 1.0}, observation_alphabet=frozenset({0, 1}))
    hmm.graph.add_state("A")
    hmm.add_transition("A", "A", 0, 1.0 - p)
    hmm.add_transition("A", "A", 1, p)
    hmm.validate()
    return hmm


def _iid_ternary() -> MealyHMM:
    hmm = MealyHMM(initial_distribution={"A": 1.0}, observation_alphabet=frozenset({0, 1, 2}))
    hmm.graph.add_state("A")
    for symbol, prob in {0: 0.2, 1: 0.3, 2: 0.5}.items():
        hmm.add_transition("A", "A", symbol, prob)
    hmm.validate()
    return hmm


# --- Parameter counting ----------------------------------------------------


def test_count_free_parameters():
    assert count_free_parameters(golden_mean(0.3)) == 1  # state A has 2 edges, B has 1
    assert count_free_parameters(fair_coin()) == 1
    assert count_free_parameters(_iid_ternary()) == 2  # 3 symbols -> 2 free


def test_count_free_parameters_include_initial():
    # golden mean has a single (stationary) start state in initial_distribution.
    base = count_free_parameters(golden_mean(0.3))
    assert count_free_parameters(golden_mean(0.3), include_initial=True) >= base


# --- Score relationships ---------------------------------------------------


def test_score_model_relationships():
    rng = np.random.default_rng(0)
    data, _ = golden_mean(0.3).sample(2000, rng=rng)
    scores = score_model(golden_mean(0.3), data)
    k, n, ll = scores.num_parameters, scores.num_observations, scores.log_likelihood
    assert n == 2000
    assert scores.aic == pytest.approx(2 * k - 2 * ll)
    assert scores.bic == pytest.approx(k * np.log(n) - 2 * ll)
    assert scores.mdl == pytest.approx(0.5 * k * np.log(n) - ll)
    assert scores.aicc == pytest.approx(scores.aic + 2 * k * (k + 1) / (n - k - 1))


def test_impossible_data_is_infinite():
    scores = score_model(fair_coin(), ["2", "2"])  # symbol not in the coin's alphabet
    assert scores.log_likelihood == float("-inf")
    assert scores.aic == float("inf")
    assert scores.bic == float("inf")


def test_aicc_infinite_without_enough_data():
    scores = score_model(_iid_ternary(), [0])  # n=1, k=2 -> n-k-1 < 0
    assert scores.aicc == float("inf")


# --- Model preference ------------------------------------------------------


def test_criteria_prefer_true_structure_over_iid():
    rng = np.random.default_rng(1)
    data, _ = golden_mean(0.3).sample(3000, rng=rng)
    for criterion in ("aic", "aicc", "bic", "mdl"):
        true_score = information_criterion(golden_mean(0.3), data, criterion=criterion)
        iid_score = information_criterion(_iid_binary(0.5), data, criterion=criterion)
        assert true_score < iid_score


def test_compare_information_criteria_keys():
    rng = np.random.default_rng(2)
    data, _ = golden_mean(0.3).sample(1000, rng=rng)
    gm = golden_mean(0.3)
    gm.name = "golden"
    iid = _iid_binary()
    iid.name = "iid"
    scores = compare_information_criteria([gm, iid], data)
    assert set(scores) == {"golden", "iid"}
    assert scores["golden"].bic < scores["iid"].bic


# --- Cross-validation ------------------------------------------------------


def test_cross_validated_log_likelihood_prefers_better_fit():
    rng = np.random.default_rng(3)
    data, _ = golden_mean(0.3).sample(4000, rng=rng)

    def fit_golden(train):
        model, _trace = golden_mean(0.6).baum_welch(train)
        return model

    def fit_iid(train):
        model, _trace = _iid_binary(0.5).baum_welch(train)
        return model

    cv_golden = cross_validated_log_likelihood(fit_golden, data, folds=5, rng=0)
    cv_iid = cross_validated_log_likelihood(fit_iid, data, folds=5, rng=0)
    assert np.isfinite(cv_golden)
    assert cv_golden > cv_iid


def test_cross_validation_requires_two_folds():
    with pytest.raises(ValueError):
        cross_validated_log_likelihood(lambda train: fair_coin(), [0, 1, 0], folds=1)


# --- WAIC ------------------------------------------------------------------


def test_waic_zero_variance_matches_deviance():
    matrix = np.tile(np.array([[-1.0, -2.0, -0.5]]), (10, 1))
    result = waic(matrix)
    assert result.p_waic == pytest.approx(0.0, abs=1e-12)
    assert result.lppd == pytest.approx(-3.5)
    assert result.waic == pytest.approx(7.0)


def test_waic_positive_effective_parameters():
    rng = np.random.default_rng(4)
    matrix = rng.normal(-2.0, 0.5, size=(200, 8))
    result = waic(matrix)
    assert result.p_waic > 0.0


def test_waic_epsilon_machine_runs():
    rng = np.random.default_rng(5)
    truth = golden_mean(0.3)
    train, _ = truth.sample(2000, rng=rng)
    posterior = EpsilonMachinePosterior(golden_mean(0.3), train)
    sequences = [truth.sample(150, rng=rng)[0] for _ in range(15)]
    result = waic_epsilon_machine(posterior, sequences, n_samples=40, rng=0)
    assert np.isfinite(result.waic)
    assert result.p_waic >= 0.0


# --- Wiring ----------------------------------------------------------------


def test_model_comparison_em_information_criteria():
    rng = np.random.default_rng(6)
    data, _ = golden_mean(0.3).sample(2000, rng=rng)
    comparison = ModelComparisonEM([golden_mean(0.5)], data)
    scores = comparison.information_criteria()
    assert scores
    best = comparison.best_by_information_criterion("bic")
    assert best in scores


def test_model_comparison_em_requires_data_for_criteria():
    comparison = ModelComparisonEM([golden_mean(0.5)], None)
    with pytest.raises(BayesianInferenceError):
        comparison.information_criteria()


def test_rank_topological_epsilon_machines_prefers_two_states():
    rng = np.random.default_rng(7)
    data, _ = golden_mean(0.3).sample(3000, rng=rng)
    ranked = rank_topological_epsilon_machines(data, alphabet=[0, 1], num_states=[1, 2], criterion="bic")
    assert ranked
    best = ranked[0]
    assert len(list(best.machine.states())) == 2
    assert best.criterion_value < ranked[-1].criterion_value
