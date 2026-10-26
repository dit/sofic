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
    from sofic.generators.generative_search import generative_model

    result = generative_model(golden_mean_bidirectional(0.5), starts=3, max_states=3, rng=np.random.default_rng(0))
    assert result.certified
    assert result.upper == pytest.approx(_h(1 / 3), abs=1e-6)
    assert result.separating_variables_generate


def test_generative_model_brackets_the_rll_process() -> None:
    from sofic.generators.generative_search import generative_model

    result = generative_model(
        _rll(), stages=("machines", "coupled"), starts=6, max_states=3, rng=np.random.default_rng(1)
    )
    assert result.lower == pytest.approx(2 - 3 / 8 * np.log2(3), abs=1e-9)
    assert result.lower_source == "support-type bound"
    assert result.upper == pytest.approx(1.5, abs=1e-5)
    assert not result.certified and not result.separating_variables_generate


def test_separating_optima_finds_the_generating_flags() -> None:
    """The interleaved process has several optima of G[S+ : S-]; exactly two of them are generator states."""
    problem = cg._GeneratorProblem(_interleaved(0.1))
    best, optima = cg.separating_optima(problem, starts=30, sizes=(4,), rng=np.random.default_rng(0))
    assert best == pytest.approx(2 * _h(4 / 9), abs=1e-7)
    generating = [c for c in optima if cg.generator_kernel(problem, c)[0] <= 1e-9]
    assert len(optima) > len(generating) == 2


def test_separating_optima_of_rll_are_not_generator_states() -> None:
    problem = cg._GeneratorProblem(_rll())
    best, optima = cg.separating_optima(problem, starts=20, sizes=(3, 4), rng=np.random.default_rng(0))
    assert best == pytest.approx(2 - 3 / 8 * np.log2(3), abs=1e-7)
    assert len(optima) == 2
    assert all(cg.generator_kernel(problem, c)[0] > 1e-3 for c in optima)
