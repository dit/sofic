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


def test_zero_time_budget_returns_a_valid_bracket_from_the_machines() -> None:
    from tests.test_generator_caps import _survey_machine

    result = generative_model_from_bidirectional(_survey_machine(), time_budget=0.0, rng=np.random.default_rng(0))
    assert result.budget_exhausted
    assert result.upper_source == "epsilon machine" or result.upper_source.endswith("epsilon machine")
    assert result.lower <= result.upper + 1e-9
    skipped = {s["stage"] for s in result.stages if s.get("skipped") == "time budget exhausted"}
    assert {"generator coverage bound", "exact common information", "coupled search"} <= skipped
    assert all("elapsed" in s for s in result.stages[:-1])
    assert result.stages[-1]["budget_exhausted"] is True


def test_unlimited_budget_is_not_reported_as_exhausted() -> None:
    result = generative_model(golden_mean(0.5), rng=np.random.default_rng(0))
    assert not result.budget_exhausted


def test_dit_fallbacks_are_skipped_on_large_joints(monkeypatch) -> None:
    import sofic.generators.generative_search as gs
    import sofic.generators.lifted_common_information as lci
    from tests.test_generator_caps import _survey_machine

    def fail(*args, **kwargs):
        raise AssertionError("dit fallback ran")

    monkeypatch.setattr(gs, "_DIT_MAX_PAIRS", 0)
    monkeypatch.setattr(lci, "_unconstrained_channel", fail)
    result = generative_model_from_bidirectional(
        _survey_machine(),
        stages=("machines", "exact"),
        exact_starts=2,
        coverage="off",
        generator_caps=0,
        early_certified=False,
        time_budget=5.0,
        rng=np.random.default_rng(0),
    )
    skipped = [s for s in result.stages if s["stage"] == "dit fallbacks"]
    assert skipped and skipped[0]["skipped"] == "more than 0 causal-state pairs"
