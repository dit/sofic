"""Tests for bidirectional preflight and retry-ladder builds."""

from __future__ import annotations

import numpy as np
import pytest

from sofic.examples.epsilon_machines import from_symbol_matrices, golden_mean_bidirectional
from sofic.generators.bidirectional_build import preflight_bidir, try_build_bidirectional
from sofic.generators.bidirectional_epsilon_machine import BidirectionalEpsilonMachine
from sofic.generators.epsilon_machine import EpsilonMachine


def _rll_forward() -> EpsilonMachine:
    return from_symbol_matrices(
        ("A", "B", "C"),
        (0, 1),
        {0: np.array([[0.5, 0, 0], [0, 0, 1.0], [1.0, 0, 0]]), 1: np.array([[0, 0.5, 0], [0, 0, 0], [0, 0, 0]])},
    )


def test_preflight_golden_mean_is_finite() -> None:
    forward = golden_mean_bidirectional(0.5).forward_machine
    pre = preflight_bidir(forward)
    assert pre.bidir_class == "finite"
    assert pre.reverse_recurrence_finite is True


def test_try_build_matches_from_forward_on_rll() -> None:
    forward = _rll_forward()
    built = try_build_bidirectional(forward)
    assert built.bidir is not None
    assert built.preflight.bidir_class == "finite"
    direct = BidirectionalEpsilonMachine.from_forward(forward)
    j1 = dict(built.bidir.joint_distribution())
    j2 = dict(direct.joint_distribution())
    assert set(j1) == set(j2)
    for k in j1:
        assert j1[k] == pytest.approx(float(j2[k]), rel=1e-9, abs=1e-9)


def test_survey_explosion_code_may_build_with_ladder() -> None:
    """Topology [[0, 1], [0, 2], [1, 0]] often hits MSP explosion at 10k."""
    code = ((0, 1), (0, 2), (1, 0))
    n = 3
    rng = np.random.default_rng(73)
    mats = {x: np.zeros((n, n)) for x in (0, 1)}
    for s, row in enumerate(code):
        live = [x for x in (0, 1) if row[x] is not None]
        probs = [1.0] if len(live) == 1 else list(rng.dirichlet([1, 1]))
        for x, p in zip(live, probs, strict=True):
            mats[x][s, row[x]] += p
    forward = from_symbol_matrices(tuple(range(n)), (0, 1), mats)
    pre_default = preflight_bidir(forward, reverse_max_states=10_000)
    if pre_default.bidir_class == "explosion":
        built = try_build_bidirectional(forward, use_ladder=True, caps=(10_000, 50_000))
        assert built.bidir is not None or built.preflight.bidir_class == "explosion"
    else:
        built = try_build_bidirectional(forward)
        assert built.bidir is not None


def test_ladder_budget_skips_caps_forecast_to_overrun(monkeypatch: pytest.MonkeyPatch) -> None:
    """A cap whose forecast (previous time x cap ratio squared) overruns the budget is not attempted."""
    import sofic.generators.bidirectional_build as bb

    clock = {"now": 0.0}
    attempted: list[int] = []

    def fake_preflight(forward, *, reverse_max_states):
        attempted.append(reverse_max_states)
        clock["now"] += 10.0 * (reverse_max_states / 10_000) ** 2
        return bb.BidirPreflightResult("explosion", reverse_max_states=reverse_max_states)

    monkeypatch.setattr(bb, "preflight_bidir", fake_preflight)
    monkeypatch.setattr(bb.time, "monotonic", lambda: clock["now"])
    built = bb.build_bidirectional_with_ladder(None, caps=(10_000, 20_000, 40_000), time_budget=100.0)
    assert attempted == [10_000, 20_000]
    assert built.bidir is None
    assert built.budget_exhausted
    assert built.preflight.reverse_max_states == 20_000

    attempted.clear()
    clock["now"] = 0.0
    built = bb.build_bidirectional_with_ladder(None, caps=(10_000, 20_000, 40_000))
    assert attempted == [10_000, 20_000, 40_000]
    assert not built.budget_exhausted


def test_truncated_reverse_machine_prunes_to_complete_rows() -> None:
    """Regression: dropping sub-tolerance joint states left (2, 'CE') with 86% of its outgoing mass."""
    mats = {x: np.zeros((4, 4)) for x in (0, 1)}
    for x, i, j, p in (
        (0, 1, 2, 0.8848986395305858),
        (0, 2, 3, 0.42866489513010786),
        (0, 3, 1, 0.3568362368298324),
        (1, 0, 1, 1.0),
        (1, 1, 0, 0.11510136046941416),
        (1, 2, 1, 0.571335104869892),
        (1, 3, 2, 0.6431637631701677),
    ):
        mats[x][i, j] = p
    forward = from_symbol_matrices(tuple(range(4)), (0, 1), mats)
    built = try_build_bidirectional(forward)
    assert built.preflight.bidir_class == "truncated"
    assert built.bidir is not None
    bidir_pi = built.bidir.forward_machine.stationary_distribution()
    marginal = np.zeros(4)
    for (alpha, _gamma), mass in built.bidir._joint_pi.items():
        marginal[alpha] += float(mass)
    assert np.allclose(marginal, bidir_pi, atol=1e-8)
