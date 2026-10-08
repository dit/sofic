"""Relative entropy rate between stationary hidden Markov processes."""

from __future__ import annotations

import math
from itertools import product

import numpy as np
import pytest
from hypothesis import assume, given, settings

from sofic.examples import bernoulli, binary_markov_chain, even_process, golden_mean
from sofic.generators.mealy import MealyHMM
from sofic.generators.relative_entropy_rate import (
    _support_includes,
    _support_nfa,
    relative_entropy_rate,
    relative_entropy_rate_bounds,
)
from sofic.testing.strategies import epsilon_machines, mealy_hmms

TOL = 1e-9


def _kl(p, q) -> float:
    return sum(a * math.log2(a / b) for a, b in zip(p, q, strict=True) if a > 0)


def _noisy_hmm(eps: float = 0.2) -> MealyHMM:
    """Non-unifilar two-state HMM in which every state can emit every symbol."""
    hmm = MealyHMM(observation_alphabet=frozenset({"0", "1"}), initial_distribution={"A": 0.5, "B": 0.5})
    for state in ("A", "B"):
        hmm.graph.add_state(state)
    hmm.add_transition("A", "A", "0", 0.6 * (1 - eps))
    hmm.add_transition("A", "B", "0", 0.4 * (1 - eps))
    hmm.add_transition("A", "A", "1", 0.6 * eps)
    hmm.add_transition("A", "B", "1", 0.4 * eps)
    hmm.add_transition("B", "A", "1", 0.3 * (1 - eps))
    hmm.add_transition("B", "B", "1", 0.7 * (1 - eps))
    hmm.add_transition("B", "A", "0", 0.3 * eps)
    hmm.add_transition("B", "B", "0", 0.7 * eps)
    return hmm


@pytest.mark.parametrize(("a", "b"), [(0.3, 0.6), (0.5, 0.1), (0.9, 0.5)])
def test_iid_equals_per_symbol_kl(a, b):
    expected = _kl([1 - a, a], [1 - b, b])
    assert relative_entropy_rate(bernoulli(a), bernoulli(b)) == pytest.approx(expected, abs=TOL)


@pytest.mark.parametrize(("p", "q"), [((0.4, 0.3), (0.2, 0.6)), ((0.1, 0.8), (0.5, 0.45))])
def test_markov_chain_is_stationary_weighted_row_kl(p, q):
    """Gray (1990), Lemma 3.10 with k = 1: ``sum_i pi_i D(P(.|i) || Q(.|i))``."""
    (p01, p10), (q01, q10) = p, q
    pi = np.array([p10, p01]) / (p01 + p10)
    rows_p = [[1 - p01, p01], [p10, 1 - p10]]
    rows_q = [[1 - q01, q01], [q10, 1 - q10]]
    expected = sum(pi[i] * _kl(rows_p[i], rows_q[i]) for i in range(2))
    actual = relative_entropy_rate(binary_markov_chain(*p), binary_markov_chain(*q))
    assert actual == pytest.approx(expected, abs=TOL)


@pytest.mark.parametrize("model", [bernoulli(0.3), golden_mean(0.4), even_process(0.6), binary_markov_chain(0.2, 0.7)])
def test_self_divergence_is_zero(model):
    assert relative_entropy_rate(model, model) == pytest.approx(0.0, abs=TOL)


def test_golden_mean_against_fair_coin():
    # Fair coin cross entropy is 1 bit; golden mean at p = 1/2 has h = 2/3.
    assert relative_entropy_rate(golden_mean(0.5), bernoulli(0.5)) == pytest.approx(1 / 3, abs=TOL)


def test_infinite_when_support_not_included():
    assert math.isfinite(relative_entropy_rate(golden_mean(0.5), bernoulli()))
    assert relative_entropy_rate(bernoulli(), golden_mean(0.5)) == math.inf
    assert relative_entropy_rate(even_process(), golden_mean(0.5)) == math.inf
    assert relative_entropy_rate(golden_mean(0.5), even_process()) == math.inf
    bounds = relative_entropy_rate_bounds(bernoulli(), golden_mean(0.5), 3)
    assert bounds.lower == bounds.upper == math.inf


def test_support_nfa_matches_positive_probability_words():
    for model in (golden_mean(0.4), even_process(0.3), _noisy_hmm()):
        nfa = _support_nfa(model)
        mealy = model.to_mealy()
        pi = mealy.stationary_distribution()
        for length in range(6):
            for word in product(("0", "1"), repeat=length):
                positive = model.word_probability(word, start=pi) > 1e-12
                assert nfa.recognizes(word) == positive


def test_support_includes_orders_golden_mean_below_full_shift():
    assert _support_includes(golden_mean(0.5), bernoulli())
    assert not _support_includes(bernoulli(), golden_mean(0.5))
    assert _support_includes(even_process(), bernoulli())


def test_non_unifilar_reference_requires_bounds():
    with pytest.raises(NotImplementedError):
        relative_entropy_rate(golden_mean(0.5), _noisy_hmm())


@pytest.mark.parametrize(
    ("p", "q"), [(golden_mean(0.4), binary_markov_chain(0.3, 0.6)), (even_process(), bernoulli(0.3))]
)
def test_bounds_bracket_exact_value_for_unifilar_reference(p, q):
    exact = relative_entropy_rate(p, q)
    for n in (1, 3, 6):
        bounds = relative_entropy_rate_bounds(p, q, n)
        assert bounds.lower - TOL <= exact <= bounds.upper + TOL


def test_bounds_converge_for_non_unifilar_reference():
    p, q = even_process(0.4), _noisy_hmm()
    gaps = []
    for n in (2, 5, 9):
        bounds = relative_entropy_rate_bounds(p, q, n)
        assert bounds.lower <= bounds.upper + TOL
        assert bounds.lower - TOL <= bounds.conditional_estimate
        gaps.append(bounds.upper - bounds.lower)
    assert gaps[0] > gaps[1] > gaps[2]


def test_self_divergence_bounds_contain_zero():
    q = _noisy_hmm()
    bounds = relative_entropy_rate_bounds(q, q, 6)
    assert bounds.lower == 0.0
    assert bounds.upper >= -TOL


def test_block_estimate_converges():
    p, q = golden_mean(0.4), binary_markov_chain(0.3, 0.6)
    exact = relative_entropy_rate(p, q)
    errors = [abs(relative_entropy_rate_bounds(p, q, n).block_estimate - exact) for n in (2, 4, 8)]
    assert errors[0] > errors[1] > errors[2]
    # A 1-step Markov reference makes the conditional estimate exact from n = 2 on.
    assert relative_entropy_rate_bounds(p, q, 4).conditional_estimate == pytest.approx(exact, abs=1e-9)


def test_block_length_must_be_positive():
    with pytest.raises(ValueError, match="block_length"):
        relative_entropy_rate_bounds(bernoulli(), bernoulli(), 0)


@settings(max_examples=60, deadline=None)
@given(epsilon_machines(max_states=3), epsilon_machines(max_states=3))
def test_nonnegative_and_bracketed(p, q):
    try:
        exact = relative_entropy_rate(p, q)
    except NotImplementedError:
        assume(False)
    assert exact >= 0.0
    if math.isfinite(exact):
        bounds = relative_entropy_rate_bounds(p, q, 4)
        assert bounds.lower - 1e-9 <= exact <= bounds.upper + 1e-9


@settings(max_examples=40, deadline=None)
@given(mealy_hmms(max_states=2), mealy_hmms(max_states=2))
def test_general_bounds_are_ordered(p, q):
    bounds = relative_entropy_rate_bounds(p, q, 4)
    assert 0.0 <= bounds.lower <= bounds.upper + 1e-9
