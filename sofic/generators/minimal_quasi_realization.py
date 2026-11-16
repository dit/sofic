"""Minimal linear (quasi-)realizations of finite-state processes.

A hidden Markov model with symbol-labeled matrices :math:`T^{(x)}`, start vector
:math:`\\pi`, and final vector :math:`\\mathbf{1}` is a *linear representation* of
its word function :math:`f(w) = \\pi T^{(w)} \\mathbf{1}`. Schützenberger's
minimization of weighted automata :cite:`Schutzenberger1961` (equivalently,
Fliess' Hankel-rank theorem) reduces any such representation to one of minimal
dimension: the dimension of the subspace that is both reachable (spanned by
:math:`\\pi T^{(w)}`) and observable (spanned by :math:`T^{(w)} \\mathbf{1}`). That
dimension is the *process rank*, the rank of the Hankel matrix
:math:`H_{u, v} = f(uv)`, and the finite dimension of Upper's generalized-state
space :cite:`Upper1997`.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Callable, Mapping
from typing import Any

import numpy as np

from sofic.generators.base import HiddenMarkovModel
from sofic.generators.matrices import emission_tensors
from sofic.generators.prob import _sympy
from sofic.generators.quasi_realization import QuasiRealization

_DEFAULT_TOL = 1e-9


def minimal_quasi_realization(
    model: HiddenMarkovModel | QuasiRealization, *, tol: float = _DEFAULT_TOL
) -> QuasiRealization:
    """Return a minimal-dimension quasi-realization of ``model``'s word process.

    The representation :math:`(\\pi, T^{(x)}, \\tau)` of ``model`` (for an HMM the
    model start vector, its joint symbol matrices, and :math:`\\tau = \\mathbf{1}`)
    is reduced by Schützenberger/Fliess minimization :cite:`Schutzenberger1961`:
    bases :math:`R` of the reachable row space :math:`\\{\\pi T^{(w)}\\}` and
    :math:`O` of the observable column space :math:`\\{T^{(w)} \\tau\\}` are grown
    breadth-first, the finite Hankel core :math:`M = R O` is factored as
    :math:`M = C F` with :math:`k = \\operatorname{rank} M`, and the projected maps

    .. math:: D^{(x)} = C^{+} R \\, T^{(x)} \\, O F^{+}

    reproduce every word probability with :math:`k` equal to the process rank
    :cite:`Upper1997`. A final change of basis makes ``tau`` the all-ones vector,
    so ``pi`` sums to :math:`f(\\varepsilon)` (one for a normalized process).

    Symbolic (sympy) input is reduced exactly and returns object-dtype arrays of
    sympy expressions; numeric input uses orthonormal Krylov bases and truncates
    the singular values of :math:`M` below ``tol`` times the largest.

    Parameters
    ----------
    model
        A hidden Markov model (any presentation with ``to_mealy()``) or an
        existing :class:`~sofic.generators.quasi_realization.QuasiRealization`.
    tol
        Relative numeric rank tolerance; ignored for exact input.
    """
    pi, matrices, tau = _representation(model)
    symbols = list(matrices)
    if _is_exact(pi, tau, *matrices.values()):
        start, maps, final = _minimize_exact(pi, [matrices[x] for x in symbols], tau)
    else:
        start, maps, final = _minimize_numeric(pi, [matrices[x] for x in symbols], tau, tol=tol)
    return _quasi_realization(start, dict(zip(symbols, maps, strict=True)), final)


def process_rank(model: HiddenMarkovModel | QuasiRealization, *, tol: float = _DEFAULT_TOL) -> int:
    """Return the process rank: the dimension of :func:`minimal_quasi_realization`.

    This equals the rank of the Hankel matrix :math:`H_{u, v} = P(uv)` of word
    probabilities :cite:`Upper1997`, and lower-bounds the state count of every
    HMM presentation of the process, including its ε-machine.
    """
    return int(minimal_quasi_realization(model, tol=tol).pi.size)


def _representation(
    model: HiddenMarkovModel | QuasiRealization,
) -> tuple[np.ndarray, dict[Any, np.ndarray], np.ndarray]:
    if isinstance(model, QuasiRealization):
        return model.pi, dict(model.symbol_maps), model.tau
    pi, matrices = emission_tensors(model)
    symbolic = pi.dtype == object or any(matrix.dtype == object for matrix in matrices.values())
    tau = np.ones(len(pi), dtype=object if symbolic else float)
    if symbolic:
        sp = _sympy()
        tau.fill(sp.Integer(1))
    return pi, matrices, tau


def _is_exact(*arrays: np.ndarray) -> bool:
    return any(np.asarray(array).dtype == object for array in arrays)


def _minimize_numeric(
    pi: np.ndarray, matrices: list[np.ndarray], tau: np.ndarray, *, tol: float
) -> tuple[np.ndarray, list[np.ndarray], np.ndarray]:
    pi = np.asarray(pi, dtype=float)
    tau = np.asarray(tau, dtype=float)
    matrices = [np.asarray(matrix, dtype=float) for matrix in matrices]
    n = pi.size

    reachable = _orthonormal_krylov(pi, [lambda v, m=m: v @ m for m in matrices], tol=tol, n=n)
    observable = _orthonormal_krylov(tau, [lambda v, m=m: m @ v for m in matrices], tol=tol, n=n).T
    core = reachable @ observable
    if core.size == 0:
        return _empty_numeric(len(matrices))
    u, s, vt = np.linalg.svd(core)
    k = int(np.sum(s > tol * s[0])) if s.size and s[0] > 0.0 else 0
    if k == 0:
        return _empty_numeric(len(matrices))
    root = np.sqrt(s[:k])
    left = (u[:, :k] / root).T @ reachable
    right = observable @ (vt[:k].T / root)

    start = pi @ right
    maps = [left @ matrix @ right for matrix in matrices]
    final = left @ tau
    pivot = int(np.argmax(np.abs(final)))
    shift, unshift = _unit_final_basis(final, pivot, identity=np.eye(k))
    return start @ unshift, [shift @ matrix @ unshift for matrix in maps], np.ones(k)


def _orthonormal_krylov(
    seed: np.ndarray, steps: list[Callable[[np.ndarray], np.ndarray]], *, tol: float, n: int
) -> np.ndarray:
    basis: list[np.ndarray] = []
    queue: deque[np.ndarray] = deque([seed])
    while queue and len(basis) < n:
        vector = queue.popleft()
        scale = float(np.linalg.norm(vector))
        residual = vector
        for _ in range(2):
            for q in basis:
                residual = residual - (q @ residual) * q
        norm = float(np.linalg.norm(residual))
        if scale == 0.0 or norm <= tol * max(1.0, scale):
            continue
        direction = residual / norm
        basis.append(direction)
        queue.extend(step(direction) for step in steps)
    return np.asarray(basis, dtype=float).reshape(len(basis), n)


def _empty_numeric(num_symbols: int) -> tuple[np.ndarray, list[np.ndarray], np.ndarray]:
    return np.zeros(0), [np.zeros((0, 0)) for _ in range(num_symbols)], np.zeros(0)


def _minimize_exact(
    pi: np.ndarray, matrices: list[np.ndarray], tau: np.ndarray
) -> tuple[np.ndarray, list[np.ndarray], np.ndarray]:
    sp = _sympy()
    n = len(pi)
    start_row = sp.Matrix([list(pi)]).applyfunc(sp.sympify)
    final_col = sp.Matrix(list(tau)).applyfunc(sp.sympify)
    maps = [sp.Matrix(np.asarray(matrix, dtype=object).tolist()).applyfunc(sp.sympify) for matrix in matrices]

    reachable = _exact_krylov(start_row, [lambda v, m=m: v * m for m in maps], n=n)
    observable = _exact_krylov(final_col.T, [lambda v, m=m: (m * v.T).T for m in maps], n=n).T
    core = (reachable * observable).applyfunc(sp.simplify)
    if core.rank(simplify=True) == 0:
        return _empty_exact(len(maps))
    c, f = core.rank_decomposition(simplify=True)
    k = c.shape[1]
    left = (c.T * c).inv() * c.T * reachable
    right = observable * f.T * (f * f.T).inv()

    start = (start_row * right).applyfunc(sp.simplify)
    reduced = [(left * m * right).applyfunc(sp.simplify) for m in maps]
    final = (left * final_col).applyfunc(sp.simplify)
    pivot = next(i for i in range(k) if sp.simplify(final[i]) != 0)
    shift, unshift = _unit_final_basis(final, pivot, identity=sp.eye(k))
    return (
        _to_object_vector(start * unshift),
        [_to_object(shift * m * unshift) for m in reduced],
        _to_object_vector(sp.ones(k, 1)),
    )


def _exact_krylov(seed: Any, steps: list[Callable[[Any], Any]], *, n: int) -> Any:
    sp = _sympy()
    rows: list[Any] = []
    queue: deque[Any] = deque([seed])
    while queue and len(rows) < n:
        row = queue.popleft().applyfunc(sp.simplify)
        if sp.Matrix.vstack(*rows, row).rank(simplify=True) <= len(rows):
            continue
        rows.append(row)
        queue.extend(step(row) for step in steps)
    return sp.Matrix.vstack(*rows) if rows else sp.zeros(0, n)


def _empty_exact(num_symbols: int) -> tuple[np.ndarray, list[np.ndarray], np.ndarray]:
    return (
        np.zeros(0, dtype=object),
        [np.zeros((0, 0), dtype=object) for _ in range(num_symbols)],
        np.zeros(0, dtype=object),
    )


def _unit_final_basis(final: Any, pivot: int, *, identity: Any) -> tuple[Any, Any]:
    """Return ``(S, S^{-1})`` with ``S @ final`` the all-ones vector.

    ``S = I + (1 - final) e_pivot^T / final[pivot]`` has determinant
    ``1 / final[pivot]`` and inverse ``I - (1 - final) e_pivot^T``.
    """
    k = identity.shape[0]
    deficit = [1 - final[i] for i in range(k)]
    shift = identity.copy()
    unshift = identity.copy()
    for i in range(k):
        shift[i, pivot] += deficit[i] / final[pivot]
        unshift[i, pivot] -= deficit[i]
    return shift, unshift


def _to_object(matrix: Any) -> np.ndarray:
    sp = _sympy()
    return np.asarray(matrix.applyfunc(sp.simplify).tolist(), dtype=object).reshape(matrix.shape)


def _to_object_vector(matrix: Any) -> np.ndarray:
    return _to_object(matrix).ravel()


def _quasi_realization(pi: np.ndarray, symbol_maps: Mapping[Any, np.ndarray], tau: np.ndarray) -> QuasiRealization:
    if pi.dtype != object:
        return QuasiRealization(pi=pi, tau=tau, symbol_maps=dict(symbol_maps))
    realization = QuasiRealization(pi=np.zeros(0), tau=np.zeros(0), symbol_maps={})
    realization.pi = pi
    realization.tau = tau
    realization.symbol_maps = dict(symbol_maps)
    return realization
