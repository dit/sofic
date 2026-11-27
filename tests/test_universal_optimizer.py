"""Tests for universal_generative_optimizer."""

from __future__ import annotations

import time

import numpy as np
import pytest

from sofic.examples import golden_mean_bidirectional
from sofic.examples.epsilon_machines import from_symbol_matrices
from sofic.exceptions import StochasticValidationError
from sofic.generators import universal_optimizer as uo
from sofic.generators.bidirectional_epsilon_machine import BidirectionalEpsilonMachine
from sofic.generators.minimal_generative_model import _reproduction_error, _reproduction_length
from sofic.generators.universal_optimizer import (
    compare_universal_to_pipeline,
    universal_generative_optimizer,
)


def _rll() -> BidirectionalEpsilonMachine:
    machine = from_symbol_matrices(
        ("A", "B", "C"),
        (0, 1),
        {0: np.array([[0.5, 0, 0], [0, 0, 1.0], [1.0, 0, 0]]), 1: np.array([[0, 0.5, 0], [0, 0, 0], [0, 0, 0]])},
    )
    return BidirectionalEpsilonMachine.from_forward(machine)


def test_universal_optimizer_certifies_golden_mean() -> None:
    bidir = golden_mean_bidirectional(0.5)
    rng = np.random.default_rng(0)
    result = universal_generative_optimizer(bidir, max_states=3, starts=8, rng=rng)
    target = float(-(1 / 3) * np.log2(1 / 3) - (2 / 3) * np.log2(2 / 3))
    assert abs(result.value - target) < 1e-5
    ref = bidir.forward_machine
    err = _reproduction_error(result.model, ref, max_length=_reproduction_length(ref))
    assert err < 1e-5
    assert result.n_verified >= 1


def test_universal_optimizer_finds_rll_upper_bound() -> None:
    bidir = _rll()
    rng = np.random.default_rng(1)
    result = universal_generative_optimizer(bidir, max_states=4, starts=10, memory_lift_m=(2,), rng=rng)
    assert result.value <= 1.5 + 1e-6
    assert result.n_verified >= 1


def test_universal_optimizer_builds_no_seeds_past_its_deadline(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []
    monkeypatch.setattr(uo, "memory_lift_search", lambda *a, **k: calls.append("memory lift") or [])
    monkeypatch.setattr(uo, "separating_optima", lambda *a, **k: calls.append("G optima") or (np.inf, []))
    monkeypatch.setattr(uo, "coupled_search", lambda *a, **k: calls.append("coupled") or [])
    with pytest.raises(StochasticValidationError):
        universal_generative_optimizer(_rll(), max_states=4, deadline=time.monotonic() - 1.0)
    assert calls == []


def test_universal_optimizer_stops_at_its_deadline_without_a_verified_generator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[int] = []
    monkeypatch.setattr(uo, "memory_lift_search", lambda *a, **k: [])
    monkeypatch.setattr(uo, "separating_optima", lambda *a, **k: (np.inf, []))

    def nothing_verifies(bidir, k, **kwargs):
        calls.append(k)
        time.sleep(0.3)
        return []

    monkeypatch.setattr(uo, "coupled_search", nothing_verifies)
    with pytest.raises(StochasticValidationError):
        universal_generative_optimizer(_rll(), max_states=4, deadline=time.monotonic() + 0.2)
    assert calls == [2]


def test_compare_universal_to_pipeline_golden_mean() -> None:
    bidir = golden_mean_bidirectional(0.5)
    rng = np.random.default_rng(2)
    cmp = compare_universal_to_pipeline(bidir, rng=rng, max_states=3, starts=6, memory_lift_m=(), include_g_seeds=False)
    assert cmp.pipeline_certified
    assert abs(cmp.universal.value - cmp.pipeline_upper) < 1e-5
