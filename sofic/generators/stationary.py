"""Stationary distribution for hidden Markov models."""

from __future__ import annotations

import warnings

import numpy as np

from sofic.exceptions import StochasticValidationError
from sofic.generators.base import HiddenMarkovModel
from sofic.generators.prob import (
    has_symbolic,
    simplify_prob,
    zeros,
)
from sofic.graph import ATTR_PROB

# Above this many states a numeric chain is solved with sparse matrices: the dense path builds the full
# matrix and computes its rank and eigendecomposition (O(n^2) memory, O(n^3) time).
_SPARSE_STATES = 2000


def stationary_distribution_hmm(hmm: HiddenMarkovModel) -> np.ndarray:
    from sofic.properties import transition_matrix

    idx = hmm.reindex()
    if len(idx) == 0:
        return np.array([], dtype=float)
    if len(idx) > _SPARSE_STATES:
        pi = _stationary_distribution_sparse(hmm, idx.states)
        if pi is not None:
            return pi
    transition, _states = transition_matrix(hmm, attr=ATTR_PROB, states=idx.states)
    return stationary_distribution_from_transition(transition)


def _stationary_distribution_sparse(hmm: HiddenMarkovModel, states: list) -> np.ndarray | None:
    """Sparse solve of ``π P = π``, ``sum(π) = 1`` for a large numeric chain.

    One balance equation is replaced by the normalization, which leaves a nonsingular system exactly when
    the stationary distribution is unique. Returns ``None`` (so the caller falls back to the dense path,
    with its warnings) for symbolic weights, a singular system, or a solution that is not a stationary
    distribution.
    """
    import scipy.sparse as sp
    from scipy.sparse.linalg import MatrixRankWarning, spsolve

    index = {state: i for i, state in enumerate(states)}
    rows, cols, values = [], [], []
    for state in states:
        for transition in hmm.graph.out_transitions(state):
            j = index.get(transition.target)
            if j is None:
                continue
            value = transition.data.get(ATTR_PROB, 0.0)
            if has_symbolic([value]):
                return None
            rows.append(index[state])
            cols.append(j)
            values.append(float(value))
    n = len(states)
    matrix = sp.csr_matrix((values, (rows, cols)), shape=(n, n))
    system = (matrix.T - sp.identity(n, format="csr")).tolil()
    system[n - 1, :] = np.ones(n)
    target = np.zeros(n)
    target[-1] = 1.0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", MatrixRankWarning)
        pi = spsolve(system.tocsc(), target)
    if not np.all(np.isfinite(pi)) or np.any(pi < -1e-10):
        return None
    pi = np.maximum(pi, 0.0)
    total = float(pi.sum())
    if total <= 0.0:
        return None
    pi = _clean_stationary_distribution(pi / total)
    if not np.allclose(matrix.T @ pi, pi, rtol=1e-8, atol=1e-10):
        return None
    return pi


def stationary_distribution_from_transition(transition: np.ndarray) -> np.ndarray:
    """Return a normalized left eigenvector of ``transition`` for eigenvalue one.

    When ``transition`` contains sympy expressions, solve ``π P = π`` and
    ``sum(π) = 1`` exactly via sympy linear algebra. A numeric chain with more
    than one closed recurrent class has no unique stationary distribution; a
    :class:`RuntimeWarning` is emitted and one stationary vector is returned.
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

    multiplicity = n - int(np.linalg.matrix_rank(matrix.T - np.eye(n), tol=1e-9))
    if multiplicity > 1:
        warnings.warn(
            f"stationary distribution is not unique: the chain has {multiplicity} closed recurrent classes; "
            "returning an arbitrary stationary vector",
            RuntimeWarning,
            stacklevel=3,
        )

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
