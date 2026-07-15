"""Tests for posterior machine and process diversity diagnostics."""

from __future__ import annotations

import numpy as np
import pytest

from sofic.examples import fair_coin
from sofic.examples.processes import Even, EvenRedundant
from sofic.inference.bayesian import (
    InferEM,
    ModelComparisonEM,
    machine_diversity,
    posterior_process_diversity,
    process_identification_word_length,
)


def test_single_topology_has_zero_diversity():
    pytest.importorskip("dit")
    data = list("1111101100")
    comparison = ModelComparisonEM([Even()], data)
    result = posterior_process_diversity(comparison)
    assert result.n_components == 1
    assert result.process_diversity == pytest.approx(0.0, abs=1e-12)
    assert result.machine_diversity == pytest.approx(0.0, abs=1e-12)
    assert comparison.machine_diversity() == pytest.approx(0.0, abs=1e-12)


def test_duplicate_topology_has_zero_process_diversity():
    pytest.importorskip("dit")
    data = list("1111101100")
    even1 = Even()
    even2 = Even()
    even2.name = "Even-copy"
    comparison = ModelComparisonEM([even1, even2], data)
    assert len(comparison.em_dict) == 2
    result = posterior_process_diversity(comparison)
    assert result.machine_diversity == pytest.approx(1.0, abs=1e-9)
    assert result.process_diversity == pytest.approx(0.0, abs=1e-12)


def test_same_process_different_topologies_have_low_process_diversity():
    pytest.importorskip("dit")
    data = list("1111101100")
    comparison = ModelComparisonEM([Even(), EvenRedundant()], data)
    assert len(comparison.em_dict) == 2
    result = posterior_process_diversity(comparison)
    assert result.machine_diversity > 0.5
    assert result.process_diversity < result.machine_diversity
    different = posterior_process_diversity(ModelComparisonEM([Even(), fair_coin()], data))
    assert result.process_diversity < different.process_diversity


def test_different_processes_have_positive_process_diversity():
    pytest.importorskip("dit")
    data = list("1111101100")
    comparison = ModelComparisonEM([Even(), fair_coin()], data)
    assert len(comparison.em_dict) == 2
    result = posterior_process_diversity(comparison)
    assert result.process_diversity > 1e-6


def test_monte_carlo_is_reproducible_with_rng():
    pytest.importorskip("dit")
    data = list("1111101100")
    comparison = ModelComparisonEM([Even(), EvenRedundant()], data)
    rng = np.random.default_rng(0)
    first = posterior_process_diversity(comparison, method="monte_carlo", n_samples=32, rng=rng)
    rng = np.random.default_rng(0)
    second = posterior_process_diversity(comparison, method="monte_carlo", n_samples=32, rng=rng)
    assert first.process_diversity == pytest.approx(second.process_diversity)
    assert first.method == "monte_carlo"
    assert first.n_components == 32


def test_word_length_override_is_respected():
    pytest.importorskip("dit")
    data = list("1111101100")
    comparison = ModelComparisonEM([Even(), EvenRedundant()], data)
    assert process_identification_word_length(comparison, word_length=2) == 2
    result = posterior_process_diversity(comparison, word_length=2)
    assert result.word_length == 2


def test_process_identification_word_length_conventions():
    pytest.importorskip("dit")
    data = list("1111101100")
    comparison = ModelComparisonEM([Even(), EvenRedundant()], data)
    n_max = max(len(p.dirichlet.nodes) for p in comparison.em_dict.values())
    assert process_identification_word_length(comparison, convention="paz") == 2 * n_max - 1
    assert process_identification_word_length(comparison, convention="conservative") == 2 * n_max + 1
    upper_length = process_identification_word_length(comparison, convention="upper_list")
    assert upper_length >= 0


def test_model_comparison_convenience_methods_match_module():
    pytest.importorskip("dit")
    data = list("1111101100")
    comparison = ModelComparisonEM([Even(), EvenRedundant()], data)
    module_result = posterior_process_diversity(comparison)
    method_result = comparison.process_diversity()
    assert method_result == module_result
    assert comparison.machine_diversity() == machine_diversity(comparison)


def test_posterior_mean_and_monte_carlo_same_order_of_magnitude():
    pytest.importorskip("dit")
    data = list("1111101100")
    comparison = ModelComparisonEM([Even(), fair_coin()], data)
    mean_result = posterior_process_diversity(comparison, method="posterior_mean")
    mc_result = posterior_process_diversity(
        comparison,
        method="monte_carlo",
        n_samples=200,
        rng=np.random.default_rng(1),
    )
    assert mean_result.process_diversity > 0.0
    assert mc_result.process_diversity > 0.0
    ratio = mc_result.process_diversity / mean_result.process_diversity
    assert 0.1 < ratio < 10.0


def test_infer_em_posterior_mean_word_distribution_matches_machine():
    pytest.importorskip("dit")
    from sofic.inference.bayesian.diversity import posterior_mean_word_distribution

    data = list("1111101100")
    posterior = InferEM(Even(), data)
    length = 3
    distribution = posterior_mean_word_distribution(posterior, length)
    start = max(posterior.start_node_probabilities(), key=posterior.start_node_probabilities().get)
    machine = posterior.posterior_mean_machine(start)
    assert machine is not None
    machine_words = machine.words_of_length(length)
    assert set(distribution) == set(machine_words)
    for word, prob in machine_words.items():
        assert distribution[word] == pytest.approx(prob, abs=1e-12)
