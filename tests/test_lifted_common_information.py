"""Tests for lifted (generator-constrained) Wyner and exact common information."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from sofic.examples.epsilon_machines import bernoulli, from_symbol_matrices, golden_mean_bidirectional
from sofic.examples.processes import odd
from sofic.generators import lifted_common_information as lifted
from sofic.generators.bidirectional_epsilon_machine import BidirectionalEpsilonMachine
from sofic.generators.lifted_common_information import _LiftedProblem, _set_partitions, partition_residual
from sofic.generators.minimal_generative_model import (
    MinimalGenerativeModel,
    WynerGenerativeModel,
    _auxiliary_state_channel,
    _normalized_joint_distribution,
    _optimize_wyner_common_information,
    _reproduction_error,
    _reproduction_length,
    functional_generative_model,
    minimal_generative_model,
    wyner_generative_model,
)
from tests.test_minimal_generative_model import _analytic_binary_markov_cg, _binary_entropy, _binary_markov


def _reproduces(model, bidir) -> bool:
    reference = bidir.forward_machine
    return _reproduction_error(model, reference, max_length=_reproduction_length(reference)) < 1e-3


def test_golden_mean_matches_unconstrained_values() -> None:
    bidir = golden_mean_bidirectional(0.5)
    wyner = wyner_generative_model(bidir, rng=np.random.default_rng(0))
    exact = minimal_generative_model(bidir, rng=np.random.default_rng(0))
    assert isinstance(wyner, WynerGenerativeModel)
    assert isinstance(exact, MinimalGenerativeModel)
    assert wyner.wyner_common_information == pytest.approx(0.6667, abs=1e-3)
    assert exact.exact_common_information == pytest.approx(_binary_entropy(1 / 3), abs=1e-6)
    assert exact.generative_complexity() == pytest.approx(exact.exact_common_information, abs=1e-6)
    for model in (wyner, exact):
        assert model.markov_residual <= 1e-7
        assert _reproduces(model, bidir)


@pytest.mark.parametrize(("p", "q"), [(0.3, 0.6), (0.2, 0.7), (0.4, 0.9), (0.15, 0.35)])
def test_binary_markov_exact_equals_generative_complexity(p: float, q: float) -> None:
    bidir = BidirectionalEpsilonMachine.from_forward(_binary_markov(p, q))
    exact = minimal_generative_model(bidir, rng=np.random.default_rng(1))
    assert exact.exact_common_information == pytest.approx(_analytic_binary_markov_cg(p, q), abs=5e-4)
    assert _reproduces(exact, bidir)


def test_odd_process_wyner_optimum_is_an_interior_generator() -> None:
    """The unconstrained Wyner optimum of the Odd process lies inside the rim and already lifts."""
    bidir = BidirectionalEpsilonMachine.from_forward(odd())
    joint = _normalized_joint_distribution(bidir, cutoff=1e-10)
    channel, value = _auxiliary_state_channel(
        joint,
        optimizer=_optimize_wyner_common_information,
        optimizer_name="Wyner common information",
        bound=None,
        niter=10,
        maxiter=1000,
        polish=1e-8,
        backend="numpy",
        cutoff=1e-10,
        rng=np.random.default_rng(0),
    )
    complexity = min(
        float(bidir.forward_machine.statistical_complexity()), float(bidir.reverse_machine.statistical_complexity())
    )
    assert float(bidir.excess_entropy()) < value < complexity - 0.1
    assert partition_residual(bidir, joint, channel) < 1e-4


def test_odd_process_wyner_model_generates() -> None:
    bidir = BidirectionalEpsilonMachine.from_forward(odd())
    wyner = wyner_generative_model(bidir, rng=np.random.default_rng(0))
    assert float(bidir.excess_entropy()) < wyner.wyner_common_information
    assert wyner.wyner_common_information <= float(bidir.forward_machine.statistical_complexity()) + 1e-9
    assert _reproduces(wyner, bidir)


def _rll_two_infinity() -> BidirectionalEpsilonMachine:
    """Every 1 is followed by at least two 0s."""
    process = from_symbol_matrices(
        ("A", "B", "C"),
        (0, 1),
        {
            0: np.array([[0.5, 0.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]]),
            1: np.array([[0.0, 0.5, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
        },
    )
    return BidirectionalEpsilonMachine.from_forward(process)


def test_rll_two_infinity_constraints_bind() -> None:
    """The unconstrained exact optimum (a deterministic partition) is not a generator state."""
    bidir = _rll_two_infinity()
    model = minimal_generative_model(bidir, rng=np.random.default_rng(0))
    assert bidir.exact_common_information(rng=np.random.default_rng(0)) == pytest.approx(1.4056, abs=2e-3)
    assert model.exact_common_information == pytest.approx(1.5, abs=1e-4)
    assert _reproduces(model, bidir)


def test_functional_model_always_generates() -> None:
    """dit's functional partition fails the lifting conditions here; the lifted partition search does not."""
    bidir = _rll_two_infinity()
    joint = _normalized_joint_distribution(bidir, cutoff=1e-10)
    model = functional_generative_model(bidir)
    assert partition_residual(bidir, joint, model.pair_state_channel) <= 1e-10
    assert model.functional_common_information == pytest.approx(1.5, abs=1e-9)
    assert model.generative_complexity() == pytest.approx(model.functional_common_information, abs=1e-9)
    assert _reproduces(model, bidir)


def test_functional_greedy_search_generates(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(lifted, "_EXHAUSTIVE_PAIRS", 0)
    bidir = _rll_two_infinity()
    model = functional_generative_model(bidir)
    assert model.functional_common_information <= float(bidir.statistical_complexity()) + 1e-9
    assert _reproduces(model, bidir)


@pytest.mark.parametrize(("n", "bell"), [(0, 0), (1, 1), (3, 5), (5, 52)])
def test_set_partitions_enumerates_each_partition_once(n: int, bell: int) -> None:
    partitions = {tuple(labels) for labels in _set_partitions(n)}
    assert len(partitions) == bell == len(list(_set_partitions(n)))


def test_single_causal_state_is_trivial() -> None:
    bidir = BidirectionalEpsilonMachine.from_forward(bernoulli(0.3, symbols=(0, 1)))
    model = wyner_generative_model(bidir)
    assert model.wyner_common_information == 0.0
    assert len(list(model.states())) == 1


def test_generator_seeds_satisfy_the_lifted_constraints() -> None:
    bidir = BidirectionalEpsilonMachine.from_forward(odd())
    problem = _LiftedProblem(bidir, _normalized_joint_distribution(bidir, cutoff=1e-10))
    for seed in problem.seeds(len(problem.pairs)):
        assert problem.markov_residual(seed) < 1e-12


def test_constraint_jacobian_matches_finite_differences() -> None:
    bidir = BidirectionalEpsilonMachine.from_forward(odd())
    problem = _LiftedProblem(bidir, _normalized_joint_distribution(bidir, cutoff=1e-10))
    c = np.random.default_rng(0).dirichlet(np.ones(3), size=len(problem.pairs))
    jac = problem.constraint_jacobian(c)
    h = 1e-7
    for i, j in np.ndindex(c.shape):
        step = np.zeros_like(c)
        step[i, j] = h
        fd = (problem.constraint_informations(c + step) - problem.constraint_informations(c - step)) / (2 * h)
        np.testing.assert_allclose(jac[:, i, j], fd, atol=1e-6)


@given(p=st.floats(min_value=0.05, max_value=0.95), q=st.floats(min_value=0.05, max_value=0.95))
@settings(max_examples=10, deadline=None)
def test_binary_markov_lifted_wyner_bounds(p: float, q: float) -> None:
    process = _binary_markov(p, q)
    bidir = BidirectionalEpsilonMachine.from_forward(process)
    wyner = wyner_generative_model(bidir, starts=2, rng=np.random.default_rng(0))
    excess = float(bidir.excess_entropy())
    assert excess - 1e-6 <= wyner.wyner_common_information <= float(process.statistical_complexity()) + 1e-6
    assert wyner.markov_residual <= 1e-7
    assert _reproduces(wyner, bidir)
