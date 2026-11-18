"""Belief-free generator search over HMM symbol matrices."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from sofic.generators import hmm_search as hs


def _forward(mats):
    from sofic.examples.epsilon_machines import from_symbol_matrices
    from sofic.generators.epsilon_machine import EpsilonMachine

    n = mats[0].shape[0]
    m = from_symbol_matrices(tuple(range(n)), (0, 1), mats)
    return m if isinstance(m, EpsilonMachine) else EpsilonMachine.from_hmm(m)


def _problem(forward):
    from sofic.generators.bidirectional_epsilon_machine import BidirectionalEpsilonMachine
    from sofic.generators.coupled_generators import _GeneratorProblem

    return _GeneratorProblem(BidirectionalEpsilonMachine.from_forward(forward))


def _two_state(p=0.6115387860108349, q=0.2662431875066565):
    return _forward({0: np.array([[p, 0], [q, 0]]), 1: np.array([[0, 1 - p], [1 - q, 0]])})


def _dataset_n3_t122_d2():
    """Dataset process ``n3-t122-d2``: C_mu = 0.9037; a 4-state generator of entropy 0.79555 exists."""
    a, b, c = 0.17230706472197183, 0.45789302678502125, 0.557898211837216
    mats = {
        0: np.array([[0, a, 0], [0, 0, b], [c, 0, 0]]),
        1: np.array([[1 - a, 0, 0], [1 - b, 0, 0], [1 - c, 0, 0]]),
    }
    return _forward(mats)


@settings(max_examples=10, deadline=None)
@given(st.integers(0, 2**31 - 1), st.sampled_from([0.0, 1.0]))
def test_objective_gradient_matches_finite_differences(seed, weight) -> None:
    rng = np.random.default_rng(seed)
    k, ns = 3, 2
    tree = hs._WordTree(ns, 4)
    target = rng.random(len(tree.words))
    lam = rng.normal(size=len(target))
    theta = rng.normal(size=k * ns * k)
    value, grad = hs._objective(theta, k, ns, tree, target, lam, 7.0, weight)
    eps = 1e-6
    numeric = np.array(
        [
            (hs._objective(theta + eps * np.eye(len(theta))[i], k, ns, tree, target, lam, 7.0, weight)[0] - value) / eps
            for i in range(len(theta))
        ]
    )
    assert np.abs(numeric - grad).max() < 1e-4 * max(1.0, np.abs(grad).max())


def test_search_returns_exact_generators() -> None:
    from sofic.generators.coupled_generators import CoupledSearchResult, _model_from_kernel
    from sofic.generators.minimal_generative_model import _equivalence_error

    forward = _two_state()
    problem = _problem(forward)
    results = hs.hmm_generator_search(forward, 2, starts=3, rng=np.random.default_rng(0))
    assert results

    def h(p):
        p = p[p > 0]
        return float(-(p * np.log2(p)).sum())

    excess = h(problem.p_plus) + h(problem.p_minus) - h(problem.p_pair)
    for res in results:
        seed = hs.coupled_seed_from_hmm(problem, res.matrices)
        assert problem.residual(*seed) < 1e-6
        model = _model_from_kernel(problem, CoupledSearchResult(*seed, res.entropy, 0.0, "t"), 1e-12)
        assert _equivalence_error(model, forward) < 1e-6
        assert res.entropy >= excess - 1e-9  # every generator carries at least the excess entropy


@pytest.mark.slow
def test_search_beats_the_epsilon_machine_on_n3_t122_d2() -> None:
    forward = _dataset_n3_t122_d2()
    results = hs.hmm_generator_search(forward, 4, starts=8, rng=np.random.default_rng(0))
    assert results and results[0].entropy < 0.80  # C_mu = 0.9037


def test_pipeline_stage_runs_and_can_be_disabled() -> None:
    from sofic.generators.bidirectional_epsilon_machine import BidirectionalEpsilonMachine
    from sofic.generators.generative_search import generative_model_from_bidirectional

    bidir = BidirectionalEpsilonMachine.from_forward(_two_state())
    on = generative_model_from_bidirectional(bidir, time_budget=0.0, rng=np.random.default_rng(0))
    assert "belief-free HMM search" in {s["stage"] for s in on.stages}
    off = generative_model_from_bidirectional(bidir, time_budget=0.0, hmm_search=False, rng=np.random.default_rng(0))
    assert "belief-free HMM search" not in {s["stage"] for s in off.stages}
