"""Tests for generative_model entry point."""

from __future__ import annotations

import numpy as np

from sofic.examples import golden_mean, golden_mean_bidirectional
from sofic.generators.bidirectional_epsilon_machine import BidirectionalEpsilonMachine
from sofic.generators.generative_search import generative_model, generative_model_from_bidirectional


def test_generative_model_from_forward_epsilon_machine_certifies_golden_mean() -> None:
    forward = golden_mean(0.5)
    rng = np.random.default_rng(0)
    result = generative_model(
        forward,
        max_states=3,
        starts=8,
        exact_starts=15,
        entropy_sweep_starts=20,
        rng=rng,
    )
    assert result.stages[0]["stage"] == "bidirectional build"
    assert result.stages[-1]["stage"] == "reliability summary"
    assert result.stages[-1]["certified"] is True
    assert result.certified
    target = float(-(1 / 3) * np.log2(1 / 3) - (2 / 3) * np.log2(2 / 3))
    assert abs(result.upper - target) < 1e-5
    assert "universal coupled" not in result.upper_source


def test_generative_model_accepts_bidirectional_input() -> None:
    bidir = golden_mean_bidirectional(0.5)
    assert isinstance(bidir, BidirectionalEpsilonMachine)
    result = generative_model(bidir, max_states=3, starts=6, rng=np.random.default_rng(1))
    assert result.stages[0]["source"] == "bidirectional input"
    assert result.certified


def test_generative_model_from_bidirectional_matches_pipeline_on_bidir() -> None:
    bidir = golden_mean_bidirectional(0.5)
    full = generative_model(bidir, max_states=3, starts=4, rng=np.random.default_rng(2))
    direct = generative_model_from_bidirectional(
        bidir,
        max_states=3,
        starts=4,
        universal_coupled=True,
        early_theorem_g_exit=True,
        rng=np.random.default_rng(2),
    )
    assert full.certified and direct.certified
    assert abs(full.upper - direct.upper) < 1e-5
