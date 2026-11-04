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
