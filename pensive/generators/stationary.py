"""Stationary distribution for hidden Markov models."""

from __future__ import annotations

import numpy as np

from pensive.exceptions import StochasticValidationError
from pensive.generators.base import HiddenMarkovModel
from pensive.generators.prob import (
    as_prob,
    has_symbolic,
    is_symbolic,
    simplify_prob,
    zeros,
)
from pensive.graph import ATTR_PROB


def stationary_distribution_hmm(hmm: HiddenMarkovModel) -> np.ndarray:
    from pensive.properties import transition_matrix

    idx = hmm.reindex()
    if len(idx) == 0:
        return np.array([], dtype=float)
    transition, _states = transition_matrix(hmm, attr=ATTR_PROB, states=idx.states)
    return stationary_distribution_from_transition(transition)


def stationary_distribution_from_transition(transition: np.ndarray) -> np.ndarray:
    """Return a normalized left eigenvector of ``transition`` for eigenvalue one.

    When ``transition`` contains sympy expressions, solve ``π P = π`` and
    ``sum(π) = 1`` exactly via sympy linear algebra.
    """
    matrix = np.asarray(transition)
    if matrix.dtype == object or has_symbolic(matrix.ravel()):
        return _stationary_distribution_symbolic(matrix)
    return _stationary_distribution_numeric(np.asarray(matrix, dtype=float))


def _stationary_distribution_numeric(matrix: np.ndarray) -> np.ndarray:
    """Return a normalized left eigenvector of ``transition`` for eigenvalue one."""
    n = matrix.shape[0]
    if matrix.shape != (n, n):
        raise ValueError("transition matrix must be square")
    if n == 0:
        return np.array([], dtype=float)

    eigenvalues, eigenvectors = np.linalg.eig(matrix.T)
    candidates = sorted(range(n), key=lambda i: abs(eigenvalues[i] - 1.0))
    for index in candidates:
        if not np.isclose(eigenvalues[index], 1.0, rtol=1e-9, atol=1e-10):
            continue
        vector = np.real_if_close(eigenvectors[:, index], tol=1000)
        if np.iscomplexobj(vector):
            continue
        pi = np.asarray(vector, dtype=float)
        if pi.sum() < 0.0:
            pi = -pi
        pi[np.isclose(pi, 0.0, atol=1e-12)] = 0.0
        if np.any(pi < -1e-10):
            continue
        pi = np.maximum(pi, 0.0)
        total = float(pi.sum())
        if total <= 0.0:
            continue
        pi = _clean_stationary_distribution(pi / total)
        if np.allclose(pi @ matrix, pi, rtol=1e-8, atol=1e-10):
            return pi

    augmented = np.vstack([matrix.T - np.eye(n), np.ones(n)])
    target = np.zeros(n + 1, dtype=float)
    target[-1] = 1.0
    solution, *_ = np.linalg.lstsq(augmented, target, rcond=None)
    solution[np.isclose(solution, 0.0, atol=1e-12)] = 0.0
    solution = np.maximum(solution, 0.0)
    total = float(solution.sum())
    if total <= 0.0:
        raise StochasticValidationError("failed to compute a positive stationary distribution")
    pi = _clean_stationary_distribution(solution / total)
    if not np.allclose(pi @ matrix, pi, rtol=1e-8, atol=1e-10):
        raise StochasticValidationError("failed to compute an invariant stationary distribution")
    return pi


def _stationary_distribution_symbolic(matrix: np.ndarray) -> np.ndarray:
    """Solve π P = π, sum π = 1 over a sympy-valued transition matrix."""
    import sympy as sp

    n = matrix.shape[0]
    if matrix.shape != (n, n):
        raise ValueError("transition matrix must be square")
    if n == 0:
        return zeros((0,), symbolic=True)

    symbols = sp.symbols(f"pi0:{n}", real=True, nonnegative=True)
    eqs = []
    for j in range(n):
        # (π P)_j = π_j
        lhs = sum(symbols[i] * sp.sympify(matrix[i, j]) for i in range(n))
        eqs.append(sp.Eq(sp.simplify(lhs - symbols[j]), 0))
    eqs.append(sp.Eq(sum(symbols), 1))

    solution = sp.solve(eqs, symbols, dict=True)
    if not solution:
        # Fall back to nullspace of (P^T - I) with normalization.
        p = sp.Matrix([[sp.sympify(matrix[i, j]) for j in range(n)] for i in range(n)])
        null = (p.T - sp.eye(n)).nullspace()
        if not null:
            raise StochasticValidationError("failed to compute a symbolic stationary distribution")
        vec = null[0]
        total = sum(vec)
        if total == 0:
            raise StochasticValidationError("failed to compute a symbolic stationary distribution")
        pi = zeros((n,), symbolic=True)
        for i in range(n):
            pi[i] = simplify_prob(vec[i] / total)
        return pi

    best = solution[0]
    pi = zeros((n,), symbolic=True)
    for i, symbol in enumerate(symbols):
        pi[i] = simplify_prob(best[symbol])
    return pi


def _clean_stationary_distribution(distribution: np.ndarray) -> np.ndarray:
    n = len(distribution)
    if n == 0:
        return distribution
    uniform = np.full(n, 1.0 / n, dtype=float)
    if np.allclose(distribution, uniform, rtol=1e-12, atol=1e-12):
        return uniform
    cleaned = distribution.copy()
    cleaned[np.isclose(cleaned, 0.0, atol=1e-15)] = 0.0
    return cleaned / cleaned.sum()
