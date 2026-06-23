"""Tests for minimal generative models."""

from __future__ import annotations

import math
from itertools import product

import numpy as np
import pytest

from pensive.examples.epsilon_machines import bernoulli, from_symbol_matrices, golden_mean_bidirectional
from pensive.generators.epsilon_machine import EpsilonMachine
from pensive.graph import ATTR_EMISSION, ATTR_PROB

pytest.importorskip("dit")
pytestmark = pytest.mark.measures


def _binary_markov(p: float, q: float) -> EpsilonMachine:
    if np.isclose(p, 0.0):
        return _constant_binary_symbol(0)
    if np.isclose(q, 0.0):
        return _constant_binary_symbol(1)
    if np.isclose(p, 1.0 - q):
        return bernoulli(p, symbols=(0, 1))

    return from_symbol_matrices(
        ("A", "B"),
        (0, 1),
        {
            0: np.array([[1.0 - p, 0.0], [q, 0.0]]),
            1: np.array([[0.0, p], [0.0, 1.0 - q]]),
        },
    )


def _constant_binary_symbol(symbol: int) -> EpsilonMachine:
    eps = EpsilonMachine(
        initial_distribution={"A": 1.0},
        observation_alphabet=frozenset({0, 1}),
    )
    eps.graph.add_state("A")
    eps.graph.add_transition("A", "A", **{ATTR_EMISSION: symbol, ATTR_PROB: 1.0})
    eps.validate()
    return eps


def _binary_entropy(prob: float) -> float:
    if prob <= 0.0 or prob >= 1.0:
        return 0.0
    return float(-(prob * math.log2(prob) + (1.0 - prob) * math.log2(1.0 - prob)))


def _analytic_binary_markov_cg(p: float, q: float) -> float:
    if np.isclose(p, 0.0) or np.isclose(q, 0.0) or np.isclose(p, 1.0 - q):
        return 0.0

    lower_alpha = min(q, 1.0 - p)
    lower_beta = max(q, 1.0 - p)
    values: list[float] = []
    for alpha, beta in product((0.0, lower_alpha), (lower_beta, 1.0)):
        if beta <= alpha:
            continue
        gamma = (beta + p - 1.0) / (beta - alpha)
        delta = (beta - q) / (beta - alpha)
        if not (-1e-12 <= gamma <= 1.0 + 1e-12 and -1e-12 <= delta <= 1.0 + 1e-12):
            continue
        pr_a = (q * gamma + p * delta) / (p + q)
        values.append(_binary_entropy(pr_a))

    if not values:
        raise AssertionError(f"no feasible Fig. 5 generator corners for p={p}, q={q}")
    return min(values)


def _assert_same_words(left: EpsilonMachine, right: EpsilonMachine, *, max_length: int = 5, atol: float = 5e-5) -> None:
    for length in range(max_length + 1):
        left_words = left.word_probabilities(length, sparse=False)
        right_words = right.word_probabilities(length, sparse=False)
        assert left_words.keys() == right_words.keys()
        for word, probability in left_words.items():
            assert right_words[word] == pytest.approx(probability, abs=atol)


def _mgm(process: EpsilonMachine):
    return process.minimal_generative_model(
        bound=2,
        niter=8,
        maxiter=400,
        polish=1e-8,
        cutoff=1e-8,
        rng=np.random.default_rng(1234),
    )


def _wgm(process: EpsilonMachine):
    return process.wyner_generative_model(
        bound=2,
        niter=8,
        maxiter=400,
        polish=1e-8,
        cutoff=1e-8,
        rng=np.random.default_rng(4321),
    )


@pytest.mark.parametrize(
    ("p", "q"),
    [
        (1.0 / 4.0, 1.0 / 2.0),
        (1.0 / 2.0, 1.0 / 4.0),
        (3.0 / 4.0, 3.0 / 4.0),
        (0.45, 0.45),
    ],
)
def test_binary_markov_minimal_generative_model_matches_paper_examples(p: float, q: float):
    process = _binary_markov(p, q)
    bidir = process.to_bidirectional()
    mgm = _mgm(process)

    _assert_same_words(process, mgm)
    assert mgm.state_entropy() == pytest.approx(_analytic_binary_markov_cg(p, q), abs=5e-4)
    assert bidir.excess_entropy() <= mgm.generative_complexity() + 1e-8
    assert mgm.generative_complexity() <= process.statistical_complexity() + 1e-8
    assert len([state for state, mass in mgm.initial_distribution.items() if mass > 1e-8]) == 2


@pytest.mark.parametrize(("p", "q"), [(0.25, 0.75), (0.45, 0.55), (0.0, 0.5), (0.5, 0.0)])
def test_binary_markov_iid_and_constant_boundaries_collapse_to_one_state(p: float, q: float):
    process = _binary_markov(p, q)
    mgm = process.minimal_generative_model(cutoff=1e-10, rng=np.random.default_rng(5678))

    _assert_same_words(process, mgm)
    assert mgm.generative_complexity() == pytest.approx(0.0, abs=1e-9)
    assert len(list(mgm.states())) == 1


def test_binary_markov_period_two_boundary_uses_matching_support():
    for _ in range(5):
        process = _binary_markov(1.0, 1.0)
        mgm = process.minimal_generative_model()

        _assert_same_words(process, mgm)
        assert mgm.generative_complexity() == pytest.approx(1.0, abs=1e-12)
        assert len(list(mgm.states())) == 2


def test_bidirectional_and_epsilon_machine_apis_agree():
    process = _binary_markov(1.0 / 4.0, 1.0 / 2.0)
    bidir = process.to_bidirectional()

    from_forward = process.minimal_generative_model(
        bound=2,
        niter=8,
        maxiter=400,
        polish=1e-8,
        cutoff=1e-8,
        rng=np.random.default_rng(9),
    )
    from_bidir = bidir.minimal_generative_model(
        bound=2,
        niter=8,
        maxiter=400,
        polish=1e-8,
        cutoff=1e-8,
        rng=np.random.default_rng(9),
    )

    _assert_same_words(process, from_forward)
    _assert_same_words(process, from_bidir)
    assert process.generative_complexity(
        bound=2,
        niter=8,
        maxiter=400,
        polish=1e-8,
        cutoff=1e-8,
        rng=np.random.default_rng(9),
    ) == pytest.approx(from_bidir.generative_complexity(), abs=1e-9)


def test_golden_mean_wyner_generative_model_matches_triangle_value():
    bidir = golden_mean_bidirectional(0.5)
    wgm = bidir.wyner_generative_model(
        bound=2,
        niter=12,
        maxiter=500,
        polish=1e-8,
        cutoff=1e-8,
        rng=np.random.default_rng(2468),
    )

    _assert_same_words(bidir.forward_machine, wgm)
    assert wgm.wyner_common_information == pytest.approx(2.0 / 3.0, abs=2e-3)
    assert wgm.wyner_common_information <= wgm.generative_complexity() + 1e-8


@pytest.mark.parametrize(("p", "q"), [(0.25, 0.75), (0.45, 0.55), (0.0, 0.5), (0.5, 0.0)])
def test_binary_markov_wyner_iid_and_constant_boundaries_collapse_to_one_state(p: float, q: float):
    process = _binary_markov(p, q)
    wgm = process.wyner_generative_model(cutoff=1e-10, rng=np.random.default_rng(8765))

    _assert_same_words(process, wgm)
    assert wgm.wyner_common_information == pytest.approx(0.0, abs=1e-9)
    assert wgm.generative_complexity() == pytest.approx(0.0, abs=1e-9)
    assert len(list(wgm.states())) == 1


def test_binary_markov_wyner_period_two_boundary_uses_matching_support():
    process = _binary_markov(1.0, 1.0)
    wgm = process.wyner_generative_model()

    _assert_same_words(process, wgm)
    assert wgm.wyner_common_information == pytest.approx(1.0, abs=1e-12)
    assert wgm.generative_complexity() == pytest.approx(1.0, abs=1e-12)
    assert len(list(wgm.states())) == 2


def test_wyner_bidirectional_and_epsilon_machine_apis_agree():
    process = _binary_markov(1.0 / 4.0, 1.0 / 2.0)
    bidir = process.to_bidirectional()

    from_forward = _wgm(process)
    from_bidir = bidir.wyner_generative_model(
        bound=2,
        niter=8,
        maxiter=400,
        polish=1e-8,
        cutoff=1e-8,
        rng=np.random.default_rng(4321),
    )

    _assert_same_words(process, from_forward)
    _assert_same_words(process, from_bidir)
    assert from_forward.wyner_common_information == pytest.approx(from_bidir.wyner_common_information, abs=1e-9)
