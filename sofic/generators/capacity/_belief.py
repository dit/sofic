"""Probability-simplex grids and average-reward value iteration.

Grid points are ``k / n`` for nonnegative integer vectors ``k`` summing to the
resolution ``n``. In cumulative coordinates ``y_i = n * sum_{j >= i} p_j``
(``i = 1, ..., d``) the grid is the set of integer points with
``n >= y_1 >= ... >= y_d >= 0``, a union of cells of the Freudenthal (Kuhn)
triangulation of the cube, so piecewise-linear interpolation needs only a sort
per query point.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations_with_replacement
from math import comb

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix


@dataclass(frozen=True, eq=False)
class SimplexGrid:
    """Resolution-``n`` grid on the probability simplex with ``n_vertices`` corners."""

    n_vertices: int
    resolution: int
    points: np.ndarray
    _codes: np.ndarray
    _order: np.ndarray

    @classmethod
    def build(cls, n_vertices: int, resolution: int, *, max_points: int = 100_000) -> SimplexGrid:
        if n_vertices < 1 or resolution < 1:
            raise ValueError("need at least one vertex and resolution >= 1")
        size = comb(resolution + n_vertices - 1, n_vertices - 1)
        if size > max_points:
            raise ValueError(f"belief grid would have {size} points (> max_points={max_points}); lower the resolution")
        d = n_vertices - 1
        if d == 0:
            cum = np.zeros((1, 0), dtype=int)
        else:
            cum = np.array(
                [sorted(c, reverse=True) for c in combinations_with_replacement(range(resolution + 1), d)],
                dtype=int,
            )
        points = _cumulative_to_counts(cum, resolution) / resolution
        codes = _encode(cum, resolution)
        order = np.argsort(codes)
        return cls(n_vertices, resolution, points, codes[order], order)

    @property
    def size(self) -> int:
        return self.points.shape[0]

    def interpolation(self, queries: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(indices, weights)``, each ``(m, d + 1)``, interpolating at ``queries``."""
        queries = np.asarray(queries, dtype=float)
        m, d, n = queries.shape[0], self.n_vertices - 1, self.resolution
        if d == 0:
            return np.zeros((m, 1), dtype=int), np.ones((m, 1))
        tail = np.cumsum(queries[:, ::-1], axis=1)[:, ::-1][:, 1:]
        y = np.clip(n * tail, 0.0, n)
        y = np.minimum.accumulate(y, axis=1)
        base = np.floor(y).astype(int)
        frac = y - base
        order = np.argsort(-frac, axis=1, kind="stable")
        f_sorted = np.take_along_axis(frac, order, axis=1)

        vertices = np.empty((m, d + 1, d), dtype=int)
        vertices[:, 0] = base
        current = base.copy()
        rows = np.arange(m)
        for k in range(d):
            current[rows, order[:, k]] += 1
            vertices[:, k + 1] = current
        weights = np.empty((m, d + 1))
        weights[:, 0] = 1.0 - f_sorted[:, 0]
        weights[:, 1:d] = f_sorted[:, :-1] - f_sorted[:, 1:]
        weights[:, d] = f_sorted[:, -1]

        codes = _encode(vertices.reshape(-1, d), n)
        position = np.searchsorted(self._codes, codes)
        position = np.minimum(position, len(self._codes) - 1)
        found = self._codes[position] == codes
        weights = weights.reshape(-1)
        if not np.all(found | (weights <= 1e-12)):
            raise AssertionError("interpolation vertex outside the simplex grid")
        indices = np.where(found, self._order[position], 0).reshape(m, d + 1)
        return indices, np.where(found, weights, 0.0).reshape(m, d + 1)


def _cumulative_to_counts(cum: np.ndarray, n: int) -> np.ndarray:
    padded = np.hstack([np.full((cum.shape[0], 1), n), cum, np.zeros((cum.shape[0], 1), dtype=int)])
    return padded[:, :-1] - padded[:, 1:]


def _encode(cum: np.ndarray, n: int) -> np.ndarray:
    radix = (n + 1) ** np.arange(cum.shape[1], dtype=np.int64)
    return cum.astype(np.int64) @ radix


def transition_operator(
    n_rows: int, n_cols: int, rows: np.ndarray, indices: np.ndarray, weights: np.ndarray, scale: np.ndarray
) -> csr_matrix:
    """Assemble ``W[row, index] += scale * weight`` from per-query interpolation data."""
    data = (weights * scale[:, None]).ravel()
    row_ids = np.repeat(rows, indices.shape[1])
    matrix = coo_matrix((data, (row_ids, indices.ravel())), shape=(n_rows, n_cols))
    return matrix.tocsr()


@dataclass(frozen=True)
class ValueIterationResult:
    """Span bounds ``gain_lower <= rho* <= gain_upper`` and the greedy policy.

    For any bias vector ``h`` and a monotone, translation-invariant operator
    ``T``, every state's optimal average reward lies in
    ``[min(Th - h), max(Th - h)]``.
    """

    gain_lower: float
    gain_upper: float
    bias: np.ndarray
    policy: np.ndarray
    iterations: int


def relative_value_iteration(
    rewards: np.ndarray,
    operator: csr_matrix,
    *,
    tol: float = 1e-9,
    max_iter: int = 20_000,
    damping: float = 0.5,
    bias: np.ndarray | None = None,
) -> ValueIterationResult:
    """Maximize average reward ``rewards[i, a] + (operator h)[a * n + i]``.

    ``operator`` stacks one ``(n, n)`` transition block per action; disallowed
    actions carry reward ``-inf``. The damped update ``h <- (1 - t) h + t T h``
    removes periodicity without moving the fixed point.
    """
    n, n_actions = rewards.shape
    h = np.zeros(n) if bias is None else np.array(bias, dtype=float)
    iterations = 0
    while True:
        iterations += 1
        q = rewards + (operator @ h).reshape(n_actions, n).T
        th = q.max(axis=1)
        diff = th - h
        lower, upper = float(diff.min()), float(diff.max())
        if upper - lower < tol or iterations >= max_iter:
            break
        h = h + damping * diff
        h -= h[0]
    policy = q.argmax(axis=1)
    return ValueIterationResult(lower, upper, h, policy, iterations)
