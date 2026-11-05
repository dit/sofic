"""Phase A/B: coverage caps, exact cover bound, systematic seeds."""

from __future__ import annotations

import numpy as np
import pytest

from sofic.examples.epsilon_machines import from_symbol_matrices, golden_mean_bidirectional
from sofic.generators import coupled_generators as cg
from sofic.generators.bidirectional_epsilon_machine import BidirectionalEpsilonMachine
from sofic.generators.coverage_bound import coverage_bound
from sofic.generators.generator_bounds import support_exact_cover_bound


def _rll() -> BidirectionalEpsilonMachine:
    machine = from_symbol_matrices(
        ("A", "B", "C"),
        (0, 1),
        {0: np.array([[0.5, 0, 0], [0, 0, 1.0], [1.0, 0, 0]]), 1: np.array([[0, 0.5, 0], [0, 0, 0], [0, 0, 0]])},
    )
    return BidirectionalEpsilonMachine.from_forward(machine)


def test_coverage_bound_uses_multiple_certified_caps_on_full_support() -> None:
    eps = 9 / 20
    d = np.array([[1 - eps, eps], [eps, 1 - eps]]) / 2
    cov = coverage_bound(d, k_max=6, seed=0)
    assert cov.certified
    assert cov.k_used >= 2
    assert len(cov.caps) == cov.k_used


def test_support_exact_cover_on_full_product_support() -> None:
    joint = {
        ("0", "0"): 0.25,
        ("0", "1"): 0.25,
        ("1", "0"): 0.25,
        ("1", "1"): 0.25,
    }
    exact = support_exact_cover_bound(joint)
    assert exact is not None
    assert exact.certified
    assert exact.value == pytest.approx(0.0, abs=1e-12)


def test_support_exact_cover_none_when_rectangles_overlap() -> None:
    joint = {pair: float(m) for pair, m in _rll().joint_distribution().items()}
    assert support_exact_cover_bound(joint) is None


def test_generator_state_cap_is_at_most_carath() -> None:
    problem = cg._GeneratorProblem(golden_mean_bidirectional(0.5))
    na, nb = len(problem.p_plus), len(problem.p_minus)
    cap = cg.generator_state_cap(problem, hard_cap=99)
    assert cap <= na + nb - 1
    assert cap >= 2


def test_systematic_coupled_seeds_include_permutations() -> None:
    from tests.test_coupled_generators import _rll_ci_optima_channels

    problem = cg._GeneratorProblem(_rll())
    channels = _rll_ci_optima_channels(problem)
    seeds = cg.systematic_coupled_seeds(problem, channels[:1], max_permutations=2)
    names = {n for n, _ in seeds}
    assert any("perm" in n for n in names)
