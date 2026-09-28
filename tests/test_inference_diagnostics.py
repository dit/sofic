"""Tests for sofic.inference.diagnostics."""

from __future__ import annotations

import numpy as np
import pytest

from sofic.examples.epsilon_machines import even_process, golden_mean
from sofic.generators.epsilon_inference import cssr
from sofic.generators.hmm_inference import sample
from sofic.inference.diagnostics import (
    goodness_of_fit,
    reconstruction_sweep,
    structure_stability,
    topology_key,
)


@pytest.fixture(scope="module")
def even_sample():
    observations, _ = sample(even_process(0.5), 4000, np.random.default_rng(0))
    return observations


@pytest.mark.parametrize("statistic", ["g", "entropy_rate"])
def test_goodness_of_fit_accepts_correct_machine(even_sample, statistic):
    machine = cssr(even_sample, Lmax=4, alpha=0.001)
    result = goodness_of_fit(machine, even_sample, L=5, statistic=statistic, n_samples=49, rng=1)
    assert result.pvalue > 0.05
    assert result.null.shape == (49,)
    assert result.forbidden_words == ()


@pytest.mark.parametrize("statistic", ["g", "entropy_rate"])
def test_goodness_of_fit_rejects_short_lmax(even_sample, statistic):
    """CSSR with Lmax=1 cannot capture the even process's parity."""
    machine = cssr(even_sample, Lmax=1, alpha=0.001)
    result = goodness_of_fit(machine, even_sample, L=6, statistic=statistic, n_samples=49, rng=1)
    assert result.pvalue <= 0.05


def test_goodness_of_fit_forbidden_word():
    observations = [0, 1, 1, 0, 1, 0, 0, 1] * 20
    result = goodness_of_fit(golden_mean(0.5), observations, L=2, n_samples=9, rng=0)
    assert result.value == float("inf")
    assert (1, 1) in result.forbidden_words
    assert result.pvalue == pytest.approx(0.1)
    with pytest.raises(ValueError):
        goodness_of_fit(golden_mean(0.5), observations, L=2, statistic="nope")


def test_topology_key_is_isomorphism_invariant(even_sample):
    inferred = cssr(even_sample, Lmax=4, alpha=0.001)
    assert topology_key(inferred) == topology_key(even_process(0.5))
    assert topology_key(golden_mean(0.3)) == topology_key(golden_mean(0.7))
    assert topology_key(golden_mean(0.5)) != topology_key(even_process(0.5))


def test_structure_stability_subsample(even_sample):
    result = structure_stability(even_sample, n_resamples=12, rng=0, Lmax=4, alpha=0.001)
    assert result.reference == topology_key(even_process(0.5))
    assert result.reference_fraction >= 0.6
    assert result.n_resamples == 12
    assert result.modal_topology == result.reference


def _has_stationary_bootstrap() -> bool:
    import dit.inference

    return hasattr(dit.inference, "stationary_bootstrap")


@pytest.mark.skipif(not _has_stationary_bootstrap(), reason="needs dit.inference.stationary_bootstrap")
def test_structure_stability_block_runs(even_sample):
    result = structure_stability(
        even_sample, n_resamples=4, rng=0, resample="block", mean_block_length=500, Lmax=3, alpha=0.001
    )
    assert result.n_resamples == 4
    with pytest.raises(ValueError):
        structure_stability(even_sample, n_resamples=1, resample="nope", Lmax=2)


def test_reconstruction_sweep_golden_mean():
    observations, _ = sample(golden_mean(0.5), 4000, np.random.default_rng(2))
    sweep = reconstruction_sweep(observations, alphas=(0.01, 0.001), lmaxes=(1, 2, 3))
    target = topology_key(golden_mean(0.5))
    assert all(key == target for key in sweep.values())
