"""Tests for generators without the lifting restriction, and support-type lower bounds."""

from __future__ import annotations

from itertools import product

import numpy as np
import pytest

from sofic.examples.epsilon_machines import from_symbol_matrices, golden_mean_bidirectional
from sofic.generators import coupled_generators as cg
from sofic.generators.bidirectional_epsilon_machine import BidirectionalEpsilonMachine
from sofic.generators.generator_bounds import generator_support_bound, maximal_rectangles, support_type_bound
from sofic.generators.minimal_generative_model import _reproduction_error, _reproduction_length


def _h(p: float) -> float:
    return float(-(p * np.log2(p) + (1 - p) * np.log2(1 - p)))


def _rll() -> BidirectionalEpsilonMachine:
    machine = from_symbol_matrices(
        ("A", "B", "C"),
        (0, 1),
        {0: np.array([[0.5, 0, 0], [0, 0, 1.0], [1.0, 0, 0]]), 1: np.array([[0, 0.5, 0], [0, 0, 0], [0, 0, 0]])},
    )
    return BidirectionalEpsilonMachine.from_forward(machine)


def _interleaved(eps: float) -> BidirectionalEpsilonMachine:
    T = np.array([[1 - eps, eps], [eps, 1 - eps]])
    names = ["00", "01", "10", "11"]
    mats = {x: np.zeros((4, 4)) for x in (0, 1)}
    for i, (a, b) in enumerate(names):
        for c in "01":
            mats[int(c)][i, names.index(b + c)] = T[int(a), int(c)]
    return BidirectionalEpsilonMachine.from_forward(from_symbol_matrices(names, (0, 1), mats))


def _rll_exact_channel(problem: cg._GeneratorProblem) -> np.ndarray:
    """The deterministic exact-common-information optimum W* of the (2, inf) process as a channel."""
    pairs = problem.lifted.pairs
    partner: dict = {}
    for a, b in pairs:
        partner.setdefault(b, set()).add(a)
    r00 = next(b for b, s in partner.items() if "B" in s)
    labels = [0 if b == r00 else 2 if a == "C" else 1 for a, b in pairs]
    return np.eye(3)[labels]


def _rll_ci_optima_channels(problem: cg._GeneratorProblem) -> list[np.ndarray]:
    """The two distinct ``G[S+ : S-]`` optima for the length-3 RLL machine (non-generating)."""
    labels = (
        (2, 2, 2, 0, 0, 1),
        (0, 2, 2, 0, 0, 1),
    )
    return [cg.polish_separating_channel(problem, np.eye(3)[list(lab)]) for lab in labels]


def _dedupe_channels(channels: list[np.ndarray], *, atol: float = 1e-5) -> list[np.ndarray]:
    unique: list[np.ndarray] = []
    for channel in channels:
        if not any(np.allclose(channel, other, atol=atol) for other in unique):
            unique.append(channel)
    return unique


def _flag_channel(problem: cg._GeneratorProblem, eps: float) -> np.ndarray:
    """V_t = flag of the pair (z_{t-2}, z_t) of the interleaved process, as a channel of the bidirectional state."""
    T = np.array([[1 - eps, eps], [eps, 1 - eps]])
    joint = dict(zip(problem.lifted.pairs, problem.lifted.p, strict=False))
    first_future = {}
    for _, m in problem.lifted.pairs:
        for c, d in product((0, 1), repeat=2):
            if all(
                abs(joint.get((s, m), 0.0) - 0.25 * T[int(s[0]), c] * T[int(s[1]), d]) < 1e-12
                for s in ("00", "01", "10", "11")
            ):
                first_future[m] = c
    alpha = (1 - 2 * eps) / (2 * (1 - eps))
    p_flag = alpha / ((1 - eps) / 2)
    nu = np.zeros((len(problem.lifted.pairs), 2))
    for i, (s, m) in enumerate(problem.lifted.pairs):
        f = p_flag if (s[0], first_future[m]) == ("0", 0) else 0.0
        nu[i] = (f, 1 - f)
    return nu


def test_loss_gradient_matches_finite_differences() -> None:
    problem = cg._GeneratorProblem(_interleaved(0.1))
    k = 3
    rng = np.random.default_rng(0)
    theta = rng.normal(size=sum(int(np.prod(s)) for s in problem.sizes(k)))
    shifts = tuple(0.01 * rng.normal(size=r.shape) for r in problem.residuals(*problem.unpack(theta, k)))
    _, grad = problem.loss_and_grad(theta, k, 10.0, shifts, 0.7)
    step = 1e-6
    numeric = np.array(
        [
            (
                problem.loss_and_grad(theta + step * e, k, 10.0, shifts, 0.7)[0]
                - problem.loss_and_grad(theta - step * e, k, 10.0, shifts, 0.7)[0]
            )
            / (2 * step)
            for e in np.eye(len(theta))
        ]
    )
    assert np.max(np.abs(numeric - grad)) <= 1e-6 * np.max(np.abs(grad))


@pytest.mark.parametrize("bidir", [_rll(), _interleaved(0.1), golden_mean_bidirectional(0.4)])
def test_machine_seeds_solve_the_generator_equations(bidir) -> None:
    problem = cg._GeneratorProblem(bidir)
    for seed in (problem.epsilon_seed(), problem.reverse_seed(), problem.pair_seed()):
        assert problem.residual(*seed) < 1e-12


def test_rll_coupled_search_reaches_the_epsilon_machine() -> None:
    results = cg.coupled_search(_rll(), 3, starts=10, rng=np.random.default_rng(2))
    assert results[0].entropy == pytest.approx(1.5, abs=1e-5)


def test_memory_lift_of_the_flag_is_the_minimal_interleaved_generator() -> None:
    """The coupled generator of the interleaved process is the memory-1 lift of a single flag."""
    problem = cg._GeneratorProblem(_interleaved(0.1))
    pi, Q, R, K = cg.memory_lift(problem, _flag_channel(problem, 0.1))
    assert problem.residual(pi, Q, R, K) < 1e-12
    assert float(-(pi * np.log2(pi)).sum()) == pytest.approx(2 * _h(4 / 9), abs=1e-12)


def test_generator_kernel_decides_generator_states() -> None:
    rll = cg._GeneratorProblem(_rll())
    assert cg.generator_kernel(rll, np.eye(len(rll.lifted.pairs)))[0] == pytest.approx(0.0, abs=1e-10)
    assert cg.generator_kernel(rll, _rll_exact_channel(rll))[0] > 0.05  # no generator has W* as its state
    z = cg._GeneratorProblem(_interleaved(0.1))
    nu = _flag_channel(z, 0.1)
    joint = np.einsum("pxq,pv,qu->vup", z.lifted.edges, nu, nu).reshape(4, -1)
    product_flag = (joint / z.lifted.p[None, :]).T
    violation, kernel = cg.generator_kernel(z, product_flag)
    assert violation == pytest.approx(0.0, abs=1e-9)
    assert np.allclose(kernel.reshape(4, -1).sum(axis=1), 1.0)


def test_support_bound_is_the_exact_common_information_of_rll() -> None:
    value, types, masses = support_type_bound({k: float(v) for k, v in _rll().joint_distribution().items()})
    assert value == pytest.approx(2 - 3 / 8 * np.log2(3), abs=1e-12)
    assert len(types) == 3
    assert sorted(masses) == sorted([0.5, 0.375, 0.125])


def test_support_bound_proves_golden_mean_generative_complexity() -> None:
    bidir = golden_mean_bidirectional(0.5)
    assert generator_support_bound(bidir) == pytest.approx(_h(1 / 3), abs=1e-12)


def test_maximal_rectangles_of_a_full_support_is_one_rectangle() -> None:
    cells = {(a, b) for a in "xy" for b in "uv"}
    assert maximal_rectangles(cells) == [(frozenset("xy"), frozenset("uv"))]


def test_coupled_generative_model_reproduces_the_process() -> None:
    bidir = golden_mean_bidirectional(0.5)
    model = cg.coupled_generative_model(bidir, states=2, starts=4, rng=np.random.default_rng(0))
    reference = bidir.forward_machine
    assert _reproduction_error(model, reference, max_length=_reproduction_length(reference)) < 1e-5
    assert model.generative_complexity() == pytest.approx(_h(1 / 3), abs=1e-5)


def test_polishing_makes_the_flag_channel_exact() -> None:
    """A slightly perturbed product-flag channel is polished back to an exact generator state."""
    z = cg._GeneratorProblem(_interleaved(0.1))
    nu = _flag_channel(z, 0.1)
    joint = np.einsum("pxq,pv,qu->vup", z.lifted.edges, nu, nu).reshape(4, -1)
    c = (joint / z.lifted.p[None, :]).T
    rng = np.random.default_rng(0)
    noisy = np.clip(c + 1e-4 * rng.random(c.shape), 0, None)
    noisy /= noisy.sum(axis=1, keepdims=True)
    assert cg.generator_kernel(z, noisy)[0] > 1e-6
    polished = cg.polish_separating_channel(z, noisy)
    assert cg.generator_kernel(z, polished)[0] == pytest.approx(0.0, abs=1e-9)
    pi = z.lifted.p @ polished
    assert float(-(pi[pi > 0] * np.log2(pi[pi > 0])).sum()) == pytest.approx(2 * _h(4 / 9), abs=1e-7)


def test_generative_model_certifies_the_golden_mean() -> None:
    from sofic.generators.generative_search import generative_model_from_bidirectional

    result = generative_model_from_bidirectional(
        golden_mean_bidirectional(0.5), starts=3, max_states=3, rng=np.random.default_rng(0)
    )
    assert result.certified
    assert result.upper == pytest.approx(_h(1 / 3), abs=1e-6)
    assert result.separating_variables_generate


def test_generative_model_brackets_the_rll_process() -> None:
    from sofic.generators.generative_search import generative_model_from_bidirectional

    result = generative_model_from_bidirectional(
        _rll(), stages=("machines", "exact", "coupled"), starts=6, max_states=3, rng=np.random.default_rng(1)
    )
    h11 = -(0.5 * np.log2(0.5) + (3 / 11) * np.log2(3 / 11) + (5 / 22) * np.log2(5 / 22))
    assert result.lower == pytest.approx(h11, abs=1e-6)
    assert result.lower_source.startswith("flow bound")
    assert result.upper == pytest.approx(1.5, abs=1e-5)
    assert not result.certified and not result.separating_variables_generate
    assert result.cg_exceeds_g is True
    assert result.n_g_optima_generating == 0
    assert result.lower_certified is True


def test_flow_bound_result_certified_on_rll() -> None:
    from sofic.generators.generator_bounds import flow_bound, flow_bound_result, rll_flow_bound_certified

    rll = _rll()
    h11 = -(0.5 * np.log2(0.5) + (3 / 11) * np.log2(3 / 11) + (5 / 22) * np.log2(5 / 22))
    cert = rll_flow_bound_certified(thresholds=(0.55, 0.65))
    assert cert == pytest.approx(h11, abs=1e-9)
    fb = flow_bound_result(rll, thresholds=(0.55, 0.65))
    assert fb.certified and fb.value == pytest.approx(h11, abs=1e-9)
    assert flow_bound(rll, thresholds=(0.55, 0.65)) == pytest.approx(h11, abs=1e-6)


def test_coverage_certified_uses_mu1_only_on_full_support() -> None:
    from sofic.generators.generative_search import _coverage_lower_bound

    eps = 9 / 20
    d = np.array([[1 - eps, eps], [eps, 1 - eps]]) / 2
    joint = {
        ("0", "0"): float(d[0, 0]),
        ("0", "1"): float(d[0, 1]),
        ("1", "0"): float(d[1, 0]),
        ("1", "1"): float(d[1, 1]),
    }
    cov, meta, ok = _coverage_lower_bound(joint, np.random.default_rng(0), mu_starts=8, certified=True, k_max=4)
    assert ok
    assert meta["certified"] is True
    assert cov == pytest.approx(_h(10 / 11), abs=1e-3)
    caps = meta["caps"]
    assert caps[1] == pytest.approx(min(1.0, 2 * meta["mu_1"]), rel=1e-9)


def test_mu_k_certified_trivial_cap_and_lp_on_dsbs_product() -> None:
    from sofic.generators.coverage_bound import (
        lp_rank_k_weights,
        mu1_optimize,
        mu_k_certified_trivial_cap,
        mu_k_lp_simplex,
        mu_k_optimize,
    )

    eps = 9 / 20
    d = np.array([[1 - eps, eps], [eps, 1 - eps]]) / 2
    m = np.kron(d, d)
    mu1_big = mu1_optimize(m)
    assert mu1_big.certified
    cap2 = mu_k_certified_trivial_cap(2, mu1_big.value)
    assert cap2 == pytest.approx(min(1.0, 2 * mu1_big.value), rel=1e-12)
    opt2 = mu_k_optimize(m, 2, seed=0)
    assert opt2.certified and opt2.value < cap2
    assert opt2.value == pytest.approx(1 - 11 / 400, rel=1e-6)
    p00 = (1 - eps) / 2
    a = p00 - (eps / 2) ** 2 / p00
    q = np.array([eps, 1 - eps])
    u = np.vstack([np.kron(q, q), np.kron(np.array([1.0, 0.0]), q)])
    v = np.vstack([np.kron(q, q), np.kron(np.array([1.0, 0.0]), q)])
    w1, w2 = (1 - a) ** 2, a * (1 - a)
    lp = lp_rank_k_weights(m, u, v)
    assert lp == pytest.approx(w1 + w2, rel=0.02)
    assert lp <= opt2.value + 1e-9
    lp_search = mu_k_lp_simplex(m, 2, starts=8, seed=1)
    assert (lp_search.feasible or 0.0) <= opt2.value + 1e-9


def test_separating_optima_finds_the_generating_flags() -> None:
    """The interleaved process has several optima of G[S+ : S-]; exactly two of them are generator states."""
    problem = cg._GeneratorProblem(_interleaved(0.1))
    best, optima = cg.separating_optima(problem, starts=60, rng=np.random.default_rng(0))
    assert best == pytest.approx(2 * _h(4 / 9), abs=1e-7)
    generating = [
        c for c in optima if cg.generator_kernel(problem, cg.polish_separating_channel(problem, c))[0] <= 1e-9
    ]
    assert len(optima) > len(generating)
    assert len(generating) == 2


def test_separating_optima_of_rll_are_not_generator_states() -> None:
    problem = cg._GeneratorProblem(_rll())
    expected = 2 - 3 / 8 * np.log2(3)
    best, _ = cg.separating_optima(problem, starts=60, sizes=(3, 4), rng=np.random.default_rng(0))
    assert best == pytest.approx(expected, abs=1e-7)
    channels = _rll_ci_optima_channels(problem)
    assert len(_dedupe_channels(channels)) == 2
    assert all(cg.generator_kernel(problem, c)[0] > 1e-3 for c in channels)


def test_flow_bound_reproduces_the_rll_certificates() -> None:
    from sofic.generators.generator_bounds import flow_bound

    rll = _rll()
    h5 = -(2 * (5 / 12) * np.log2(5 / 12) + (1 / 6) * np.log2(1 / 6))
    h11 = -(0.5 * np.log2(0.5) + (3 / 11) * np.log2(3 / 11) + (5 / 22) * np.log2(5 / 22))
    assert flow_bound(rll, thresholds=(0.6,)) == pytest.approx(h5, abs=1e-6)
    assert flow_bound(rll, thresholds=(0.55, 0.65)) == pytest.approx(h11, abs=1e-6)
    assert flow_bound(golden_mean_bidirectional(0.5)) == pytest.approx(_h(1 / 3), abs=1e-6)


def test_mu1_brute_on_dsbs_matches_ten_elevenths() -> None:
    from sofic.generators.coverage_bound import mu1_brute_2x2, mu1_product_lower

    eps = 9 / 20
    d = np.array([[1 - eps, eps], [eps, 1 - eps]]) / 2
    w = mu1_brute_2x2(d, grid=8000)
    assert w.value == pytest.approx(10 / 11, abs=2e-3)
    assert w.u0 == pytest.approx(w.v0, abs=0.05)
    prod = mu1_product_lower(d, w)
    assert prod == pytest.approx((10 / 11) ** 2, abs=1e-4)


def test_mu1_vertex_matches_brute_on_dsbs() -> None:
    from sofic.generators.coverage_bound import mu1_brute_2x2, mu1_optimize

    eps = 9 / 20
    d = np.array([[1 - eps, eps], [eps, 1 - eps]]) / 2
    brute = mu1_brute_2x2(d, grid=6000)
    opt = mu1_optimize(d)
    assert opt.certified
    assert opt.value == pytest.approx(10 / 11, rel=1e-6)
    assert opt.value >= brute.value


def test_mu_caps_certified_on_full_support_joint() -> None:
    from sofic.generators.coverage_bound import joint_matrix, mu_caps

    machine = _interleaved(9 / 20)
    joint = {k: float(v) for k, v in machine.joint_distribution().items() if float(v) > 0}
    m, _, _ = joint_matrix(joint)
    assert np.all(m > 0)
    caps, certified = mu_caps(m, k=2, seed=0)
    assert len(caps) == 2
    assert certified
    assert caps[1] <= min(1.0, 2 * caps[0]) + 1e-12
    assert 0.5 < caps[0] < 1.0


def test_coverage_entropy_lower_is_sensible_on_interleaved() -> None:
    from sofic.generators.coverage_bound import coverage_entropy_lower, joint_matrix, mu1_search, mu_k_bilinear

    machine = _interleaved(9 / 20)
    joint = {k: float(v) for k, v in machine.joint_distribution().items() if float(v) > 0}
    m, _, _ = joint_matrix(joint)
    mu1 = mu1_search(m, maxiter=200)
    mu2 = mu_k_bilinear(m, 2, starts=30)
    assert mu1 >= mu2 or mu2 >= mu1 * 0.9
    g_lo = coverage_entropy_lower((mu1, mu2))
    assert 0.5 < g_lo < 1.5


def test_separating_decomposition_channels_are_sorted_by_entropy() -> None:
    problem = cg._GeneratorProblem(golden_mean_bidirectional(0.4))
    rows = cg.separating_decomposition_channels(problem, starts=15, sizes=(2, 3), rng=np.random.default_rng(1))
    entropies = [h for h, _ in rows]
    assert entropies == sorted(entropies)
    assert entropies[0] <= cg.separating_optima(problem, starts=15, rng=np.random.default_rng(1))[0] + 1e-6


def test_first_generating_separator_on_golden_mean() -> None:
    problem = cg._GeneratorProblem(golden_mean_bidirectional(0.4))
    channels = cg.separating_decomposition_channels(problem, starts=30, rng=np.random.default_rng(0))
    hit = cg.first_generating_separator(problem, channels, max_entropy=1.0)
    assert hit is not None
    entropy, _c, _K = hit
    g, _ = cg.separating_optima(problem, starts=30, rng=np.random.default_rng(0))
    assert entropy >= g - 1e-5
    assert cg.generator_kernel(problem, hit[1])[0] <= 1e-9
