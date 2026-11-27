"""Certified mu_k caps (Phase A2)."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from sofic.generators.coverage_bound import (
    coverage_bound,
    coverage_entropy_lower,
    lp_rank_k_weights,
    mu1_rank1_mass,
    mu1_vertex,
    mu_caps,
    mu_hyperplane_cap,
    mu_k_spectral_cap,
)
from tests.oracles import least_entropy_under_caps_reference


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


def test_hyperplane_cap_closed_forms() -> None:
    assert mu_hyperplane_cap(np.eye(3) / 3).value == pytest.approx(2 / 3, abs=1e-8)
    rng = np.random.default_rng(1)
    U, V = rng.dirichlet(np.ones(3), size=2), rng.dirichlet(np.ones(5), size=2)
    rank_two = 0.6 * np.outer(U[0], V[0]) + 0.4 * np.outer(U[1], V[1])
    assert mu_hyperplane_cap(rank_two).value == pytest.approx(1.0, abs=1e-9)
    assert mu_hyperplane_cap(rank_two.T).value == pytest.approx(1.0, abs=1e-9)


@settings(max_examples=6, deadline=None)
@given(st.integers(3, 6), st.floats(0.01, 0.3), st.integers(0, 2**31 - 1))
def test_hyperplane_cap_brackets_planted_rank_two_mass(n: int, delta: float, seed: int) -> None:
    rng = np.random.default_rng(seed)
    U, V = rng.dirichlet(np.ones(3), size=2), rng.dirichlet(np.ones(n), size=2)
    w = rng.dirichlet(np.ones(2))
    noise = rng.random((3, n)) * (rng.random((3, n)) < 0.5)
    planted = (1 - delta) * (w[0] * np.outer(U[0], V[0]) + w[1] * np.outer(U[1], V[1]))
    raw = planted + (delta * noise / noise.sum() if noise.sum() > 0 else 0.0)
    m = raw / raw.sum()
    cap = mu_hyperplane_cap(m)
    assert cap.certified
    assert (cap.feasible or 0.0) <= cap.value
    assert cap.value >= planted.sum() / raw.sum() - 1e-9
    assert cap.value <= mu_k_spectral_cap(m, 2) + 1e-9
    assert mu_caps(m, k=3)[0][1] <= cap.value + 1e-12


def test_coverage_exact_on_block_diagonal_support() -> None:
    m = np.diag([0.4, 0.6])
    cov = coverage_bound(m)
    assert cov.certified
    assert cov.value == pytest.approx(_h(0.4, 0.6), abs=1e-6)


def _increment_entropy(caps: list[float]) -> float:
    c = [0.0, *caps]
    return _h(*[c[j] - c[j - 1] for j in range(1, len(c))], 1.0 - c[-1])


def test_coverage_entropy_lower_is_the_least_entropy_when_the_cap_increments_increase() -> None:
    # The increments c_2 - c_1 and 1 - c_2 increase; the least entropy fills the second cap and repeats w_2.
    tight, loose = (0.4720301, 0.7239705), (0.4720301, 0.7269834)
    for c1, c2 in (tight, loose):
        value = coverage_entropy_lower((c1, c2))
        assert value == pytest.approx(_h(2 * c2 - 1, 1 - c2, 1 - c2), abs=1e-9)
        assert value > _increment_entropy([c1, c2]) + 0.01
    assert coverage_entropy_lower(tight) == pytest.approx(1.5442250, abs=1e-7)
    assert coverage_entropy_lower(loose) < coverage_entropy_lower(tight)


def test_coverage_entropy_lower_reaches_past_ten_atoms() -> None:
    assert coverage_entropy_lower((0.09,)) == pytest.approx(_h(*[0.09] * 11, 0.01), abs=1e-9)
    caps = (0.1777, 0.2473, 0.2531)
    assert least_entropy_under_caps_reference(caps, 10) == np.inf
    assert coverage_entropy_lower(caps) == pytest.approx(least_entropy_under_caps_reference(caps, 14), abs=1e-9)
    assert coverage_entropy_lower((0.1, 0.15, 0.999)) == pytest.approx(_h(*[0.075] * 13, 0.025), abs=1e-9)


def test_coverage_entropy_lower_with_a_vanishing_head_weight_and_caps_near_one() -> None:
    # The head polytope of these caps has an edge ending at a fifth weight of 1e-17.
    caps = (0.22122052, 0.45389342, 0.61437035, 0.78020509, 0.8070901)
    value = coverage_entropy_lower(caps)
    assert _increment_entropy(list(caps)) - 1e-12 <= value <= least_entropy_under_caps_reference(caps, 8) + 1e-9
    near_one = (0.3, 0.6, 0.9, 0.99, 0.999, 0.9999, 0.99999, 0.999999)
    assert coverage_entropy_lower(near_one) == pytest.approx(_increment_entropy(list(near_one)), abs=1e-12)


def test_coverage_entropy_lower_with_more_than_eight_caps() -> None:
    assert coverage_entropy_lower(np.arange(1, 13) / 13) == pytest.approx(np.log2(13), abs=1e-9)
    caps = (0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.68, 0.7, 0.72)
    value = coverage_entropy_lower(caps)
    assert value >= coverage_entropy_lower(caps[:8]) - 1e-12
    assert value >= _increment_entropy(list(caps)) - 1e-12


_CAPS = st.lists(st.floats(0.05, 1.0), min_size=1, max_size=4).map(sorted)


@settings(max_examples=60, deadline=None)
@given(_CAPS)
def test_coverage_entropy_lower_is_between_the_increment_entropy_and_every_few_atom_distribution(caps) -> None:
    value = coverage_entropy_lower(caps)
    assert value >= _increment_entropy(caps) - 1e-12
    assert value <= least_entropy_under_caps_reference(caps, 6) + 1e-9


@settings(max_examples=60, deadline=None)
@given(st.lists(st.floats(0.01, 1.0), min_size=2, max_size=5))
def test_coverage_entropy_lower_is_the_increment_entropy_when_increments_do_not_increase(raw) -> None:
    w = np.sort(np.asarray(raw) / sum(raw))[::-1]
    assert coverage_entropy_lower(np.cumsum(w)[:-1]) == pytest.approx(_h(*w), abs=1e-9)


@settings(max_examples=60, deadline=None)
@given(_CAPS, st.lists(st.floats(0.5, 1.0), min_size=4, max_size=4))
def test_coverage_entropy_lower_does_not_fall_as_the_caps_tighten(caps, shrink) -> None:
    tighter = np.minimum.accumulate((np.asarray(caps) * shrink[: len(caps)])[::-1])[::-1]
    assert coverage_entropy_lower(tighter) >= coverage_entropy_lower(caps) - 1e-9


@settings(max_examples=60, deadline=None)
@given(
    st.lists(st.floats(0.01, 1.0), min_size=2, max_size=12),
    st.lists(st.floats(0.0, 0.2), min_size=3, max_size=3),
    st.integers(1, 3),
)
def test_coverage_entropy_lower_is_at_most_the_entropy_of_any_distribution_under_the_caps(raw, slack, k) -> None:
    w = np.sort(np.asarray(raw) / sum(raw))[::-1]
    k = min(k, len(w))
    caps = np.cumsum(w)[:k] + np.cumsum(slack)[:k]
    assert coverage_entropy_lower(caps) <= _h(*w) + 1e-9


@settings(max_examples=60, deadline=None)
@given(st.integers(1, 4), st.integers(1, 4), st.integers(0, 2**32 - 1), st.floats(0.0, 0.6))
def test_mu1_vertex_matches_the_union_find_reference(na: int, nb: int, seed: int, sparsity: float) -> None:
    from sofic.generators.coverage_bound import mu1_vertex
    from tests.oracles import mu1_vertex_reference

    rng = np.random.default_rng(seed)
    M = rng.random((na, nb)) * (rng.random((na, nb)) >= sparsity)
    if M.sum() == 0.0:
        M[0, 0] = 1.0
    M /= M.sum()
    got = mu1_vertex(M)
    value, u, v = mu1_vertex_reference(M)
    assert got.value == value
    np.testing.assert_array_equal(got.u, u)
    np.testing.assert_array_equal(got.v, v)


@settings(max_examples=60, deadline=None)
@given(st.integers(2, 4), st.integers(1, 4), st.integers(1, 5), st.integers(0, 2**32 - 1))
def test_column_box_lp_matches_the_lapack_reference(m: int, nb: int, npairs: int, seed: int) -> None:
    from sofic.generators.coverage_bound import _column_box_lp
    from tests.oracles import column_box_lp_reference

    rng = np.random.default_rng(seed)
    U = rng.random((nb, m))
    A, C = rng.normal(size=(npairs, m)), rng.normal(size=(npairs, m))
    np.testing.assert_allclose(_column_box_lp(U, A, C), column_box_lp_reference(U, A, C), rtol=1e-9, atol=1e-12)
