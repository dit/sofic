"""Rényi entropy rates, pressure and the large-deviation rate function."""

from __future__ import annotations

import math

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from sofic.examples import bernoulli, even_process, golden_mean
from sofic.examples.epsilon_machines import from_symbol_matrices
from sofic.generators.epsilon_machine import EpsilonMachine
from sofic.generators.matrices import symbol_matrices
from sofic.generators.mealy import MealyHMM
from sofic.generators.renyi import pressure, rate_function, renyi_entropy_rate
from sofic.testing.strategies import epsilon_machines

TOL = 1e-9


@st.composite
def weighted_machines(draw, max_states: int = 3) -> EpsilonMachine:
    """Topological ε-machine skeleton with random positive edge probabilities."""
    skeleton = draw(epsilon_machines(max_states=max_states))
    states = list(skeleton.reindex().states)
    matrices = {symbol: np.array(m, dtype=float) for symbol, m in symbol_matrices(skeleton).items()}
    for i in range(len(states)):
        edges = [(symbol, j) for symbol, m in matrices.items() for j in np.flatnonzero(m[i])]
        weights = np.array([draw(st.floats(0.1, 1.0)) for _ in edges])
        for (symbol, j), weight in zip(edges, weights / weights.sum(), strict=True):
            matrices[symbol][i, j] = weight
    return from_symbol_matrices(states, tuple(matrices), matrices)


def _block_power_sum(model, n: int, alpha: float) -> float:
    probs = np.array([p for p in model.words_of_length(n).values() if p > 0.0])
    return float(np.log2(np.sum(probs**alpha)))


def _support_entropy(model) -> float:
    adjacency = sum((np.asarray(m, dtype=float) > 0).astype(float) for m in symbol_matrices(model).values())
    return float(np.log2(np.max(np.abs(np.linalg.eigvals(adjacency)))))


def _noisy_hmm() -> MealyHMM:
    hmm = MealyHMM(observation_alphabet=frozenset({"0", "1"}), initial_distribution={"A": 0.5, "B": 0.5})
    for state in ("A", "B"):
        hmm.graph.add_state(state)
    hmm.add_transition("A", "A", "0", 0.5)
    hmm.add_transition("A", "B", "1", 0.5)
    hmm.add_transition("B", "A", "1", 0.5)
    hmm.add_transition("B", "B", "1", 0.5)
    return hmm


@pytest.mark.parametrize("p", [0.1, 0.25, 0.5])
@pytest.mark.parametrize("alpha", [0.0, 0.5, 2.0, 7.0])
def test_iid_closed_form(p: float, alpha: float) -> None:
    expected = math.log2((1 - p) ** alpha + p**alpha) / (1 - alpha)
    assert renyi_entropy_rate(bernoulli(p), alpha) == pytest.approx(expected, abs=TOL)
    assert pressure(bernoulli(p), alpha) == pytest.approx((1 - alpha) * expected, abs=TOL)


def test_iid_min_entropy() -> None:
    assert renyi_entropy_rate(bernoulli(0.2), np.inf) == pytest.approx(-math.log2(0.8), abs=TOL)


def test_golden_mean_limits() -> None:
    model = golden_mean(0.5)
    assert renyi_entropy_rate(model, 0.0) == pytest.approx(math.log2((1 + math.sqrt(5)) / 2), abs=TOL)
    assert renyi_entropy_rate(model, 1.0) == pytest.approx(2 / 3, abs=TOL)
    assert renyi_entropy_rate(model, np.inf) == pytest.approx(0.5, abs=TOL)


def test_non_unifilar_input_is_converted() -> None:
    hmm = _noisy_hmm()
    eps = EpsilonMachine.from_hmm(hmm)
    for alpha in (0.0, 0.5, 1.0, 2.0, np.inf):
        assert renyi_entropy_rate(hmm, alpha) == pytest.approx(renyi_entropy_rate(eps, alpha), abs=1e-9)


@pytest.mark.parametrize("bad", [-0.5, float("nan")])
def test_rejects_negative_orders(bad: float) -> None:
    with pytest.raises(ValueError, match="non-negative"):
        renyi_entropy_rate(golden_mean(0.5), bad)
    with pytest.raises(ValueError, match="non-negative"):
        pressure(golden_mean(0.5), bad)


def test_brute_force_even_process() -> None:
    model = even_process(0.5)
    for alpha in (0.5, 2.0):
        estimate = (_block_power_sum(model, 20, alpha) - _block_power_sum(model, 10, alpha)) / (10 * (1 - alpha))
        assert estimate == pytest.approx(renyi_entropy_rate(model, alpha), abs=0.01)


@pytest.mark.parametrize("p", [0.2, 0.4])
def test_iid_rate_function_is_cramer(p: float) -> None:
    low, high = -math.log2(1 - p), -math.log2(p)
    values = np.linspace(low + 0.05, (low + high) / 2, 5)
    q = (values - low) / (high - low)
    expected = q * np.log2(q / p) + (1 - q) * np.log2((1 - q) / (1 - p))
    np.testing.assert_allclose(rate_function(bernoulli(p), values), expected, atol=1e-7)


def test_rate_function_domain() -> None:
    model = golden_mean(0.5)
    below, above = rate_function(model, [0.4, 5.0])
    assert below == np.inf
    assert np.isnan(above)
    assert rate_function(model, 2 / 3) == pytest.approx(0.0, abs=1e-9)


@settings(max_examples=40)
@given(weighted_machines(), st.floats(0.05, 5.0))
def test_properties(model: EpsilonMachine, alpha: float) -> None:
    h_mu = float(model.entropy_rate())
    assert renyi_entropy_rate(model, 1.0) == pytest.approx(h_mu, abs=TOL)
    assert renyi_entropy_rate(model, 1.0 + 1e-6) == pytest.approx(h_mu, abs=1e-5)
    assert renyi_entropy_rate(model, 0.0) == pytest.approx(_support_entropy(model), abs=TOL)
    rates = [renyi_entropy_rate(model, a) for a in (0.0, alpha, alpha + 0.5, 2 * alpha + 1.0, np.inf)]
    assert all(a >= b - 1e-9 for a, b in zip(rates, rates[1:], strict=False))


@settings(max_examples=20)
@given(weighted_machines(), st.sampled_from([0.5, 2.0, 3.0]), st.integers(1, 10))
def test_brute_force_sandwich(model: EpsilonMachine, alpha: float, n: int) -> None:
    pi = np.asarray(model.stationary_distribution(), dtype=float)
    keep = pi > 1e-12
    powered = sum(
        np.where(m > 0, m, 1.0) ** alpha * (m > 0)
        for m in (np.asarray(m, dtype=float)[np.ix_(keep, keep)] for m in symbol_matrices(model).values())
    )
    middle = math.log2(pi[keep] ** alpha @ np.linalg.matrix_power(powered, n) @ np.ones(keep.sum()))
    size = math.log2(keep.sum())
    brute = _block_power_sum(model, n, alpha)
    assert middle - size - 1e-9 <= brute <= middle + alpha * size + 1e-9
    assert renyi_entropy_rate(model, alpha) == pytest.approx(
        math.log2(np.max(np.abs(np.linalg.eigvals(powered)))) / (1 - alpha), abs=1e-9
    )


@settings(max_examples=20)
@given(weighted_machines())
def test_rate_function_nonnegative_with_zero_at_entropy_rate(model: EpsilonMachine) -> None:
    h_mu = float(model.entropy_rate())
    h_inf = renyi_entropy_rate(model, np.inf)
    h_0 = renyi_entropy_rate(model, 0.0)
    values = np.linspace(h_inf, h_0, 9)
    rates = rate_function(model, values)
    finite = rates[np.isfinite(rates)]
    assert np.all(finite >= 0.0)
    assert rate_function(model, h_mu) == pytest.approx(0.0, abs=1e-7)
