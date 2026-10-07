"""Algebraic conjugacy invariants of edge shifts (Lind & Marcus §7.4)."""

from __future__ import annotations

from typing import Any, NamedTuple

import numpy as np

from sofic.shifts.algorithms import integer_adjacency_matrix, reciprocal_characteristic_polynomial
from sofic.shifts.base import SymbolicModel


class BowenFranksGroup(NamedTuple):
    """``BF(A) = Z^r / Z^r (I - A)`` together with the sign of ``det(I - A)``.

    ``invariant_factors`` are the Smith-form diagonal entries of ``I - A``
    other than ``1``, in divisibility order, with ``0`` standing for a free
    ``Z`` summand; ``()`` is the trivial group.
    """

    invariant_factors: tuple[int, ...]
    det_sign: int


def bowen_franks_group(matrix: np.ndarray | SymbolicModel) -> BowenFranksGroup:
    """Return the Bowen-Franks group of ``A`` and the sign of ``det(I - A)``.

    ``BF(A) ≅ Z_{d_1} ⊕ ... ⊕ Z_{d_r}`` where ``d_j`` are the diagonal entries
    of the Smith form of ``I - A`` over ``Z`` :cite:`LindMarcus1995`
    (Definition 7.4.15); it is an invariant of shift equivalence over ``Z``
    and hence of conjugacy (Theorem 7.4.17). ``det(I - A) = 1/zeta(1)`` is a
    conjugacy invariant too, and its sign together with ``BF(A)`` is the
    Parry-Sullivan / Bowen-Franks flow-equivalence invariant
    :cite:`LindMarcus1995` (§13.6). Arithmetic is exact.

    Parameters
    ----------
    matrix : array_like or TopologicalMarkovChain
        Square integer matrix, or a shift whose adjacency matrix (with edge
        multiplicities) is used.

    Returns
    -------
    BowenFranksGroup
        ``(invariant_factors, det_sign)``.
    """
    a = _integer_matrix(matrix)
    size = a.shape[0]
    difference = [[int(i == j) - int(a[i, j]) for j in range(size)] for i in range(size)]
    diagonal = _smith_diagonal(difference)
    det = sum(reciprocal_characteristic_polynomial(a))
    return BowenFranksGroup(tuple(d for d in diagonal if d != 1), (det > 0) - (det < 0))


def jordan_form_away_from_zero(matrix: np.ndarray | SymbolicModel) -> tuple[tuple[Any, int], ...]:
    """Return ``J^x(A)``, the Jordan blocks of ``A`` with nonzero eigenvalue.

    ``J^x(A)`` is the Jordan form of the invertible part of ``A``, an invariant
    of shift equivalence over ``Z`` :cite:`LindMarcus1995` (Definition 7.4.9,
    Theorem 7.4.10). Block sizes come from exact ranks: for each irreducible
    factor ``q != t`` of ``chi_A`` over ``Q``, ``d_k = nullity(q(A)^k) / deg q``
    and the number of blocks of size ``k`` is ``2 d_k - d_(k-1) - d_(k+1)``
    for each root of ``q`` (Exercise 7.4.5). Requires sympy
    (``sofic[symbolic]``).

    Parameters
    ----------
    matrix : array_like or TopologicalMarkovChain
        Square integer matrix, or a shift whose adjacency matrix is used.

    Returns
    -------
    tuple of (sympy.Expr, int)
        ``(eigenvalue, block size)`` pairs, one per block, sorted canonically.
    """
    import sympy as sp

    a = _integer_matrix(matrix)
    size = a.shape[0]
    m = sp.Matrix(size, size, lambda i, j: int(a[i, j]))
    x = sp.Symbol("x")
    _, factors = sp.Poly(m.charpoly(x).as_expr(), x).factor_list()
    blocks: list[tuple[Any, int]] = []
    for q, multiplicity in factors:
        if q.eval(0) == 0:
            continue
        q_at_a = sp.zeros(size, size)
        for coefficient in q.all_coeffs():
            q_at_a = q_at_a * m + coefficient * sp.eye(size)
        nullities = [0]
        power = sp.eye(size)
        while nullities[-1] < multiplicity:
            power = power * q_at_a
            nullities.append((size - power.rank()) // q.degree())
        nullities.append(nullities[-1])
        counts = {k: 2 * nullities[k] - nullities[k - 1] - nullities[k + 1] for k in range(1, len(nullities) - 1)}
        for root in q.all_roots():
            blocks.extend((root, k) for k, count in counts.items() for _ in range(count))
    return tuple(sorted(blocks, key=lambda block: (sp.default_sort_key(block[0]), block[1])))


def _integer_matrix(matrix: np.ndarray | SymbolicModel) -> np.ndarray:
    if isinstance(matrix, SymbolicModel):
        return integer_adjacency_matrix(matrix)[0]
    a = np.asarray(matrix)
    if a.size == 0:
        return np.zeros((0, 0), dtype=object)
    if a.ndim != 2 or a.shape[0] != a.shape[1]:
        raise ValueError("matrix must be square")
    if any(int(v) != v for v in a.flat):
        raise ValueError("matrix must have integer entries")
    return np.vectorize(int, otypes=[object])(a)


def _smith_diagonal(matrix: list[list[int]]) -> list[int]:
    """Smith-form diagonal ``d_1 | d_2 | ... | d_r`` (``d_j >= 0``) of a square integer matrix."""
    m = [row[:] for row in matrix]
    size = len(m)
    diagonal: list[int] = []
    for t in range(size):
        entries = [(abs(m[i][j]), i, j) for i in range(t, size) for j in range(t, size) if m[i][j]]
        if not entries:
            break
        while True:
            _, i, j = min(entries)
            m[t], m[i] = m[i], m[t]
            for row in m:
                row[t], row[j] = row[j], row[t]
            pivot = m[t][t]
            for i in range(t + 1, size):
                q = m[i][t] // pivot
                m[i] = [a - q * b for a, b in zip(m[i], m[t], strict=True)]
            for j in range(t + 1, size):
                q = m[t][j] // pivot
                for row in m:
                    row[j] -= q * row[t]
            residue = [(i, j) for i in range(t + 1, size) for j in range(t + 1, size) if m[i][j] % pivot]
            if any(m[i][t] for i in range(t + 1, size)) or any(m[t][j] for j in range(t + 1, size)):
                pass
            elif residue:
                i = residue[0][0]
                m[t] = [a + b for a, b in zip(m[t], m[i], strict=True)]
            else:
                diagonal.append(abs(pivot))
                break
            entries = [(abs(m[i][j]), i, j) for i in range(t, size) for j in range(t, size) if m[i][j]]
    return diagonal + [0] * (size - len(diagonal))
