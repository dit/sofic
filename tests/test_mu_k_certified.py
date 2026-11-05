"""Certified mu_k caps (Phase A2)."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from sofic.generators.coverage_bound import (
    coverage_bound,
    lp_rank_k_weights,
    mu1_rank1_mass,
    mu1_vertex,
    mu_caps,
    mu_k_spectral_cap,
)


def _h(*p: float) -> float:
    q = np.array([x for x in p if x > 0])
    return float(-(q * np.log2(q)).sum())


def _dsbs() -> np.ndarray:
    eps = 9 / 20
    return np.array([[1 - eps, eps], [eps, 1 - eps]]) / 2


_joints = (
    st.tuples(st.integers(2, 4), st.integers(2, 4))
    .flatmap(lambda shape: arrays(np.float64, shape, elements=st.sampled_from([0.0, 0.05, 0.2, 0.5, 1.0, 3.0])))
    .filter(lambda x: x.sum() > 0)
    .map(lambda x: x / x.sum())
)


@settings(max_examples=60, deadline=None)
@given(_joints, st.integers(0, 2**31 - 1))
def test_mu1_vertex_dominates_every_rank1_shape(m: np.ndarray, seed: int) -> None:
    exact = mu1_vertex(m)
    assert exact is not None and exact.certified
    rng = np.random.default_rng(seed)
    for _ in range(20):
        u, v = rng.dirichlet(np.ones(m.shape[0])), rng.dirichlet(np.ones(m.shape[1]))
        assert mu1_rank1_mass(m, u, v) <= exact.value + 1e-12
    if np.all(m > 0):
        assert mu1_rank1_mass(m, exact.u, exact.v) == pytest.approx(exact.value, rel=1e-6)


@settings(max_examples=60, deadline=None)
@given(_joints, st.integers(1, 3), st.integers(0, 2**31 - 1))
def test_certified_caps_dominate_feasible_rank_k_mass(m: np.ndarray, k: int, seed: int) -> None:
    caps, certified = mu_caps(m, k=k)
    assert certified
    assert list(caps) == sorted(caps)
    rng = np.random.default_rng(seed)
    for _ in range(10):
        U = rng.dirichlet(np.ones(m.shape[0]), size=k)
        V = rng.dirichlet(np.ones(m.shape[1]), size=k)
        assert lp_rank_k_weights(m, U, V) <= caps[-1] + 1e-9
        assert lp_rank_k_weights(m, U, V) <= mu_k_spectral_cap(m, k) + 1e-9


def test_mu1_vertex_on_dsbs_powers() -> None:
    d = _dsbs()
    m = d
    for n in (1, 2, 3):
        assert mu1_vertex(m).value == pytest.approx((10 / 11) ** n, rel=1e-6)
        m = np.kron(m, d)


def test_spectral_caps_tighten_interleaved_product() -> None:
    m = np.kron(_dsbs(), _dsbs())
    caps, certified = mu_caps(m, k=4)
    assert certified
    # singular values of D (x) D: 1/4, 1/40, 1/40, 1/400
    assert caps == pytest.approx((100 / 121, 1 - 11 / 400, 1 - 1 / 400, 1.0), rel=1e-6)
    cov = coverage_bound(m)
    assert cov.certified
    assert cov.value == pytest.approx(_h(caps[0], caps[1] - caps[0], caps[2] - caps[1], 1 - caps[2]), rel=1e-9)
    assert cov.value <= 2 * _h(10 / 11, 1 / 11) + 1e-9


def test_coverage_exact_on_block_diagonal_support() -> None:
    m = np.diag([0.4, 0.6])
    cov = coverage_bound(m)
    assert cov.certified
    assert cov.value == pytest.approx(_h(0.4, 0.6), abs=1e-6)
