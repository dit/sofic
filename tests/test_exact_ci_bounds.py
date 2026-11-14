"""Certified heaviest-term lower bound on the exact common information."""

from __future__ import annotations

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

from sofic.generators.coupled_generators import _decompose
from sofic.generators.exact_ci_bounds import capped_coverage_entropy, heaviest_term_bound


def _h(p: float) -> float:
    return float(-p * np.log2(p) - (1 - p) * np.log2(1 - p))


def test_capped_coverage_is_below_a_feasible_vector_the_greedy_misses() -> None:
    # caps (0.5, 0.6, 1.0): the sorted vector (0.34, 0.26, 0.26, 0.14) is feasible with S_3 = 0.86 > 0.7
    y = np.array([0.34, 0.26, 0.26, 0.14])
    assert capped_coverage_entropy([0.5, 0.6, 1.0], 1.0) <= float(-(y * np.log2(y)).sum())


def test_dsbs_reaches_its_exact_common_information() -> None:
    eps = 0.45
    D = np.array([[1 - eps, eps], [eps, 1 - eps]]) / 2
    G = _h(1 / 11)  # Kumar-Li-El Gamal Prop. 8 at eps = 9/20: weights (10/11, 1/11)
    result = heaviest_term_bound(D, target=G - 1e-4, max_nodes=50_000)
    assert result.reached
    assert result.value <= G + 1e-9


@settings(max_examples=8, deadline=None)
@given(st.integers(0, 2**32 - 1))
def test_bound_is_below_every_exact_decomposition(seed: int) -> None:
    rng = np.random.default_rng(seed)
    P = rng.dirichlet(np.ones(9)).reshape(3, 3)
    result = heaviest_term_bound(P, max_nodes=2_000)
    for k in (3, 4, 5):
        lam, A, B, residual = _decompose(
            P, rng.dirichlet(np.ones(k)), rng.dirichlet(np.ones(3), size=k), rng.dirichlet(np.ones(3), size=k)
        )
        if residual <= 1e-10:
            p = lam[lam > 1e-12]
            assert result.value <= float(-(p * np.log2(p)).sum()) + 1e-9


def test_survey_case_closes_against_its_exact_common_information() -> None:
    """((0,1),(2,·),(0,1)): the bound rises to within 0.01 of the best exact decomposition (0.93349 bits)."""
    P = np.array([[0.107, 0.3019, 0.1449], [0.0781, 0.1449, 0.0], [0.0379, 0.107, 0.0781]])
    P = P / P.sum()
    best = np.inf
    rng = np.random.default_rng(0)
    for k in (2, 3, 4):
        for _ in range(20):
            lam, A, B, residual = _decompose(
                P, rng.dirichlet(np.ones(k)), rng.dirichlet(np.ones(3), size=k), rng.dirichlet(np.ones(3), size=k)
            )
            if residual <= 1e-10:
                p = lam[lam > 1e-12]
                best = min(best, float(-(p * np.log2(p)).sum()))
    result = heaviest_term_bound(P, target=best - 1e-2, max_nodes=100_000)
    assert result.reached
    assert best - 1e-2 <= result.value <= best + 1e-9
