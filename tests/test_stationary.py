"""Tests for numeric stationary distributions."""

from __future__ import annotations

import warnings

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

import sofic.generators.stationary as stationary
from sofic.generators.mealy import MealyHMM
from sofic.generators.stationary import stationary_distribution_from_transition


def test_reducible_chain_warns_that_stationary_distribution_is_not_unique():
    transition = np.array([[0.5, 0.5, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    with pytest.warns(RuntimeWarning, match="not unique"):
        pi = stationary_distribution_from_transition(transition)
    assert pi @ transition == pytest.approx(pi)
    assert pi.sum() == pytest.approx(1.0)


@pytest.mark.parametrize(
    "transition",
    [
        np.array([[0.5, 0.5], [0.25, 0.75]]),
        np.array([[0.0, 1.0], [1.0, 0.0]]),
        np.array([[0.5, 0.5, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]]),
    ],
)
def test_unique_stationary_distribution_does_not_warn(transition):
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        pi = stationary_distribution_from_transition(transition)
    assert pi @ transition == pytest.approx(pi)


def _chain(n: int, edges: list[tuple[int, int, float]]) -> MealyHMM:
    hmm = MealyHMM(observation_alphabet=frozenset({0}))
    for i in range(n):
        hmm.graph.add_state(i)
    for i, j, p in edges:
        hmm.add_transition(i, j, 0, p)
    return hmm


def _edges(T: np.ndarray) -> list[tuple[int, int, float]]:
    return [(i, j, float(T[i, j])) for i, j in zip(*np.nonzero(T), strict=True)]


@settings(max_examples=40, deadline=None)
@given(st.integers(2, 12), st.integers(0, 2**32 - 1), st.floats(0.0, 0.7))
def test_sparse_stationary_matches_dense(n: int, seed: int, sparsity: float) -> None:
    rng = np.random.default_rng(seed)
    T = rng.random((n, n)) * (rng.random((n, n)) >= sparsity)
    T[np.arange(n), (np.arange(n) + 1) % n] += 0.1  # a cycle keeps the chain irreducible
    T /= T.sum(axis=1, keepdims=True)
    pi = stationary._stationary_distribution_sparse(_chain(n, _edges(T)), list(range(n)))
    assert pi is not None
    assert np.allclose(pi, stationary_distribution_from_transition(T), atol=1e-10)


def test_sparse_stationary_declines_two_closed_classes() -> None:
    T = np.array([[0.5, 0.5, 0, 0], [0.5, 0.5, 0, 0], [0, 0, 0.5, 0.5], [0, 0, 0.5, 0.5]])
    assert stationary._stationary_distribution_sparse(_chain(4, _edges(T)), [0, 1, 2, 3]) is None


def test_large_chain_uses_the_sparse_solve() -> None:
    import time

    n = 20_000
    edges = [(i, (i + 1) % n, 0.75) for i in range(n)] + [(i, 0, 0.25) for i in range(n)]
    start = time.monotonic()
    pi = stationary.stationary_distribution_hmm(_chain(n, edges))
    assert time.monotonic() - start < 30
    expected = 0.25 * 0.75 ** np.arange(n) / (1 - 0.75**n)
    assert np.allclose(pi, expected, atol=1e-12)
