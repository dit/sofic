"""Tests for lifted (generator-constrained) Wyner and exact common information."""

from __future__ import annotations

import time

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


def _light_pair_bidirectional() -> BidirectionalEpsilonMachine:
    """Binary 3-state process (cpfci dataset ``n3-t131-d0``) with a causal-state pair of mass 4e-11."""
    mats = {
        0: np.array([[0.0, 0.10369669321827639, 0.0], [0.0, 0.0, 0.67213678972452], [0.0, 0.9987816335358809, 0.0]]),
        1: np.array([[0.8963033067817237, 0.0, 0.0], [0.0, 0.0, 0.32786321027548004], [0.00121836646411911, 0.0, 0.0]]),
    }
    return from_symbol_matrices((0, 1, 2), (0, 1), mats).to_bidirectional()


def test_unconstrained_channel_gives_pairs_left_out_of_the_joint_the_state_marginal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bidir = _light_pair_bidirectional()
    problem = _LiftedProblem(bidir, _normalized_joint_distribution(bidir, cutoff=0.0))
    joint = _normalized_joint_distribution(bidir, cutoff=1e-10)
    (missing,) = set(problem.pairs) - set(joint)
    rows = {pair: {"G0": 0.25, "G1": 0.75} if i % 2 else {"G1": 1.0} for i, pair in enumerate(joint)}
    monkeypatch.setattr(lifted, "_auxiliary_state_channel", lambda joint, **kwargs: (rows, 0.0))
    c = lifted._unconstrained_channel(
        problem, joint, "exact", niter=1, maxiter=1, polish=False, backend="numpy", cutoff=1e-10, rng=None
    )
    assert c.shape == (len(problem.pairs), 2)
    assert np.allclose(c.sum(axis=1), 1.0)
    weights = np.array([joint[pair] for pair in rows])
    marginal = weights @ np.array([[row.get("G0", 0.0), row.get("G1", 0.0)] for row in rows.values()]) / weights.sum()
    assert np.allclose(c[problem.pairs.index(missing)], marginal)


def test_unconstrained_channel_survives_dit_dropping_a_light_pair() -> None:
    # The pipeline's dit fallback hands dit the joint above 1e-10, which drops the pair of mass 4e-11.
    bidir = _light_pair_bidirectional()
    problem = _LiftedProblem(bidir, _normalized_joint_distribution(bidir, cutoff=0.0))
    joint = _normalized_joint_distribution(bidir, cutoff=1e-10)
    assert len(joint) == len(problem.pairs) - 1
    c = lifted._unconstrained_channel(
        problem,
        joint,
        "exact",
        niter=2,
        maxiter=200,
        polish=False,
        backend="numpy",
        cutoff=1e-10,
        rng=np.random.default_rng(0),
    )
    if c is None:
        # On Windows CI dit's optimum fails sofic's channel validation, so the missing pair is never looked up.
        pytest.skip("dit's exact common information optimum failed channel validation on this platform")
    assert c.shape[0] == len(problem.pairs)
    assert np.allclose(c.sum(axis=1), 1.0)


def test_a_past_deadline_runs_no_solve_and_returns_an_epsilon_machine_channel(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail(*args, **kwargs):
        raise AssertionError("nothing runs after the deadline")

    monkeypatch.setattr(lifted, "_unconstrained_channel", fail)
    monkeypatch.setattr(lifted, "minimize", fail)
    bidir = golden_mean_bidirectional(0.5)
    model = minimal_generative_model(bidir, rng=np.random.default_rng(0), deadline=time.monotonic())
    assert _reproduces(model, bidir)
    assert model.exact_common_information == pytest.approx(_binary_entropy(1 / 3), abs=1e-9)


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


def test_generative_gap_rll_two_infinity() -> None:
    bidir = _rll_two_infinity()
    exact = bidir.generative_gap(rng=np.random.default_rng(0))
    assert exact.measure == "exact"
    assert exact.lower == pytest.approx(1.4056, abs=2e-3)
    assert exact.upper == pytest.approx(1.5, abs=1e-4)
    assert exact.gap == pytest.approx(exact.upper - exact.lower)
    assert exact.gap > 0.09
    assert exact.upper == pytest.approx(exact.model.generative_complexity(), abs=1e-6)
    wyner = bidir.generative_gap(measure="wyner", rng=np.random.default_rng(0))
    assert wyner.lower == pytest.approx(1.129, abs=5e-3)
    assert wyner.gap > 0.1


def test_generative_gap_vanishes_for_markov_chains() -> None:
    bidir = golden_mean_bidirectional(0.5)
    for measure in ("exact", "wyner"):
        result = bidir.generative_gap(measure=measure, rng=np.random.default_rng(0))
        assert 0.0 <= result.gap < 1e-3


def test_generative_gap_rejects_unknown_measure() -> None:
    with pytest.raises(ValueError, match="measure"):
        golden_mean_bidirectional(0.5).generative_gap(measure="functional")


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


@settings(max_examples=25, deadline=None)
@given(st.sampled_from(["golden", "odd", "rll"]), st.integers(1, 6), st.integers(0, 2**32 - 1))
def test_lifted_tensors_and_jacobian_match_the_einsum_reference(name: str, k: int, seed: int) -> None:
    from tests import oracles

    bidir = {
        "golden": lambda: golden_mean_bidirectional(0.4),
        "odd": lambda: BidirectionalEpsilonMachine.from_forward(odd()),
        "rll": _rll_two_infinity,
    }[name]()
    problem = _LiftedProblem(bidir, _normalized_joint_distribution(bidir, cutoff=0.0))
    rng = np.random.default_rng(seed)
    c = rng.dirichlet(np.ones(k), size=len(problem.pairs))
    c[rng.random(c.shape) < 0.2] = 0.0
    for got, expected in zip(problem.tensors(c), oracles.lifted_tensors(problem, c), strict=True):
        np.testing.assert_allclose(got, expected, rtol=1e-12, atol=1e-15)
    np.testing.assert_allclose(
        problem.constraint_informations(c), oracles.lifted_constraint_informations(problem, c), atol=1e-12
    )
    np.testing.assert_allclose(
        problem.constraint_jacobian(c), oracles.lifted_constraint_jacobian(problem, c), rtol=1e-10, atol=1e-9
    )
    # A second channel must not be served from the cache of the first.
    other = rng.dirichlet(np.ones(k), size=len(problem.pairs))
    np.testing.assert_allclose(problem.tensors(other)[1], oracles.lifted_tensors(problem, other)[1], rtol=1e-12)
