r"""Rigorous lower bounds for LPs and SDPs from approximate dual solutions, in exact rational arithmetic.

For the problem

.. math::

    \min\ c^\top x \quad\text{s.t.}\quad A_{eq} x = b_{eq},\ A_{ub} x \le b_{ub},\
    F_0^{(j)} + \textstyle\sum_i x_i F_i^{(j)} \succeq 0,\ \ell \le x \le h,

any multipliers :math:`y`, :math:`u \le 0` and :math:`Z_j \succeq 0` give, for every feasible :math:`x`,

.. math::

    c^\top x \ \ge\ b_{eq}^\top y + b_{ub}^\top u - \textstyle\sum_j \langle Z_j, F_0^{(j)} \rangle
    + \textstyle\sum_i \min(r_i \ell_i, r_i h_i),
    \qquad r = c - A_{eq}^\top y - A_{ub}^\top u - \textstyle\sum_j F^{(j)*}(Z_j),

whatever the solver error in :math:`(y, u, Z)` :cite:`neumaier2004safe,jansson2007rigorous`. The
multipliers are rationalized, each :math:`Z_j` is shifted by a multiple of the identity until an exact
:math:`LDL^\top` factorization proves it positive definite, and the right-hand side is evaluated in
:class:`fractions.Fraction`. Floats in the problem data are converted exactly, so the bound is rigorous
for the data as given. With :math:`c = 0`, a positive bound certifies infeasibility (a Farkas
certificate); :func:`certified_lp_minimum` obtains one from an elastic LP.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from fractions import Fraction
from typing import Any

import numpy as np
import scipy.sparse as sp
from scipy.optimize import linprog

Vector = Sequence[Any] | np.ndarray

_DENOMINATORS: tuple[int | None, ...] = (None, 10**12, 10**9, 10**6, 10**4)


def _fraction(value: Any, denominator: int | None = None) -> Fraction:
    exact = value if isinstance(value, Fraction) else Fraction(float(value))
    return exact if denominator is None else exact.limit_denominator(denominator)


def _coo(A: np.ndarray | sp.spmatrix | None, n: int) -> tuple[int, list[tuple[int, int, Fraction]]]:
    if A is None:
        return 0, []
    M = sp.coo_matrix(A)
    if M.shape[1] != n:
        raise ValueError(f"matrix has {M.shape[1]} columns, expected {n}")
    return M.shape[0], [(int(i), int(j), _fraction(v)) for i, j, v in zip(M.row, M.col, M.data, strict=True) if v != 0]


def _is_positive_definite(Z: list[list[Fraction]]) -> bool:
    """Exact test by Gaussian elimination without pivoting: every pivot of an SPD matrix is positive."""
    n = len(Z)
    A = [row[:] for row in Z]
    for k in range(n):
        if A[k][k] <= 0:
            return False
        for i in range(k + 1, n):
            if A[i][k] == 0:
                continue
            f = A[i][k] / A[k][k]
            for j in range(k, n):
                A[i][j] -= f * A[k][j]
    return True


def certified_psd(Z: np.ndarray, denominator: int | None = None) -> list[list[Fraction]]:
    """A rational symmetric matrix near ``Z`` that is provably positive definite.

    Rationalizes the symmetric part of ``Z`` and adds :math:`\\delta I`, starting from the float estimate
    of :math:`-\\lambda_{\\min}` and doubling until the exact test passes.
    """
    Z = 0.5 * (np.asarray(Z, dtype=float) + np.asarray(Z, dtype=float).T)
    n = Z.shape[0]
    base = [[_fraction(Z[i, j], denominator) for j in range(n)] for i in range(n)]
    for i in range(n):
        for j in range(i):
            base[j][i] = base[i][j]
    scale = max(1.0, float(np.max(np.abs(Z)))) if n else 1.0
    lam = float(np.linalg.eigvalsh(Z)[0]) if n else 0.0
    delta = Fraction(max(0.0, -lam) + 1e-12 * scale).limit_denominator(10**15) + Fraction(1, 10**15)
    while True:
        shifted = [[base[i][j] + (delta if i == j else 0) for j in range(n)] for i in range(n)]
        if _is_positive_definite(shifted):
            return shifted
        delta *= 2


@dataclass(frozen=True)
class PSDBlock:
    r"""Linear matrix inequality :math:`F_0 + \sum_i x_i F_i \succeq 0`, with ``F`` mapping ``i`` to :math:`F_i`."""

    F0: np.ndarray
    F: dict[int, np.ndarray]


def dual_bound(
    c: Vector,
    lo: Vector,
    hi: Vector,
    *,
    A_eq: np.ndarray | sp.spmatrix | None = None,
    b_eq: Vector | None = None,
    y_eq: Vector | None = None,
    A_ub: np.ndarray | sp.spmatrix | None = None,
    b_ub: Vector | None = None,
    y_ub: Vector | None = None,
    psd: Sequence[PSDBlock] = (),
    Z: Sequence[np.ndarray] = (),
    denominator: int | None = None,
) -> Fraction | None:
    """Exact lower bound on the minimum of the problem in the module docstring, or ``None`` if unbounded below.

    ``y_ub`` entries are clipped to ``<= 0`` (scipy's sign for ``<=`` rows of a minimization), and each
    ``Z`` is replaced by :func:`certified_psd`; both keep the bound valid.
    """
    n = len(c)
    r = [_fraction(v) for v in c]
    value = Fraction(0)
    for A, b, y, clip in ((A_eq, b_eq, y_eq, False), (A_ub, b_ub, y_ub, True)):
        m, entries = _coo(A, n)
        if m == 0:
            continue
        if b is None or y is None:
            raise ValueError("each constraint matrix needs its right-hand side and multipliers")
        ys = _multipliers(y, denominator, clip)
        value += sum((_fraction(bi) * yi for bi, yi in zip(list(b), ys, strict=True)), Fraction(0))
        for i, j, a in entries:
            if ys[i]:
                r[j] -= a * ys[i]
    for block, Zj in zip(psd, Z, strict=True):
        Zr = certified_psd(Zj, denominator)
        size = len(Zr)

        def inner(F: np.ndarray, Zr: list[list[Fraction]] = Zr, size: int = size) -> Fraction:
            F = np.asarray(F, dtype=float)
            return sum(
                (Zr[a][b] * _fraction(F[a, b]) for a in range(size) for b in range(size) if F[a, b] != 0),
                Fraction(0),
            )

        value -= inner(block.F0)
        for i, Fi in block.F.items():
            r[i] -= inner(Fi)
    for ri, l, h in zip(r, lo, hi, strict=True):
        if ri > 0:
            if not np.isfinite(l):
                return None
            value += ri * _fraction(l)
        elif ri < 0:
            if not np.isfinite(h):
                return None
            value += ri * _fraction(h)
    return value


def _multipliers(y: Vector, denominator: int | None, clip: bool) -> list[Fraction]:
    ys = [_fraction(v, denominator) for v in y]
    return [min(v, Fraction(0)) for v in ys] if clip else ys


@dataclass(frozen=True)
class CertifiedLPResult:
    """Outcome of :func:`certified_lp_minimum`.

    ``status`` is ``"optimal"`` (``bound`` is a rigorous lower bound on the minimum), ``"infeasible"``
    (a Farkas certificate was verified exactly) or ``"failed"`` (nothing certified). ``y_eq`` and
    ``y_ub`` are the exact multipliers behind the bound or the Farkas certificate, so
    :func:`dual_bound` can recheck them without a solver.
    """

    status: str
    bound: Fraction | None
    float_value: float | None
    y_eq: tuple[Fraction, ...] | None = None
    y_ub: tuple[Fraction, ...] | None = None


def _best_bound(c, lo, hi, denominators, *, y_eq, y_ub, **kwargs):
    best, best_d = None, None
    for d in denominators:
        value = dual_bound(c, lo, hi, denominator=d, y_eq=y_eq, y_ub=y_ub, **kwargs)
        if value is not None and (best is None or value > best):
            best, best_d = value, d
    if best is None:
        return None, None, None
    exact_eq = None if y_eq is None else tuple(_multipliers(y_eq, best_d, False))
    exact_ub = None if y_ub is None else tuple(_multipliers(y_ub, best_d, True))
    return best, exact_eq, exact_ub


def certified_lp_minimum(
    c: Vector,
    lo: Vector,
    hi: Vector,
    *,
    A_eq: np.ndarray | sp.spmatrix | None = None,
    b_eq: Vector | None = None,
    A_ub: np.ndarray | sp.spmatrix | None = None,
    b_ub: Vector | None = None,
    denominators: Sequence[int | None] = _DENOMINATORS,
) -> CertifiedLPResult:
    """Solve an LP with HiGHS and return a rigorous lower bound on its minimum (or an infeasibility proof).

    Right-hand sides may be :class:`~fractions.Fraction`: HiGHS sees their float values and the
    certificate is checked against the exact ones. The HiGHS duals are rationalized at each of
    ``denominators`` and the best exact bound is kept. If HiGHS reports infeasibility, the duals of the
    elastic LP (minimize the total violation of the equalities and inequalities) are tested as a Farkas
    certificate.
    """
    c = np.asarray(c, dtype=float)
    lo, hi = np.asarray(lo, dtype=float), np.asarray(hi, dtype=float)
    data = {"A_eq": A_eq, "b_eq": b_eq, "A_ub": A_ub, "b_ub": b_ub}
    b_eq_f = None if b_eq is None else np.array([float(v) for v in b_eq])
    b_ub_f = None if b_ub is None else np.array([float(v) for v in b_ub])
    res = linprog(
        c, A_ub=A_ub, b_ub=b_ub_f, A_eq=A_eq, b_eq=b_eq_f, bounds=list(zip(lo, hi, strict=True)), method="highs"
    )
    if res.status == 0:
        duals = {
            "y_eq": res.eqlin.marginals if A_eq is not None else None,
            "y_ub": res.ineqlin.marginals if A_ub is not None else None,
        }
        bound, exact_eq, exact_ub = _best_bound(c, lo, hi, denominators, **data, **duals)
        status = "optimal" if bound is not None else "failed"
        return CertifiedLPResult(status, bound, float(res.fun), exact_eq, exact_ub)
    if res.status == 2:
        y_eq, y_ub = _elastic_duals(lo, hi, A_eq, b_eq_f, A_ub, b_ub_f)
        if y_eq is not None or y_ub is not None:
            proof, exact_eq, exact_ub = _best_bound(
                np.zeros_like(c), lo, hi, denominators, **data, y_eq=y_eq, y_ub=y_ub
            )
            if proof is not None and proof > 0:
                return CertifiedLPResult("infeasible", None, None, exact_eq, exact_ub)
    return CertifiedLPResult("failed", None, None)


def _elastic_duals(lo, hi, A_eq, b_eq, A_ub, b_ub) -> tuple[np.ndarray | None, np.ndarray | None]:
    n = len(lo)
    m_eq = 0 if A_eq is None else A_eq.shape[0]
    m_ub = 0 if A_ub is None else A_ub.shape[0]
    A_eq_e = A_ub_e = None
    if m_eq:
        A_eq_e = sp.hstack(
            [sp.csr_matrix(A_eq), sp.identity(m_eq), -sp.identity(m_eq), sp.csr_matrix((m_eq, m_ub))]
        ).tocsr()
    if m_ub:
        A_ub_e = sp.hstack([sp.csr_matrix(A_ub), sp.csr_matrix((m_ub, 2 * m_eq)), -sp.identity(m_ub)]).tocsr()
    cost = np.concatenate([np.zeros(n), np.ones(2 * m_eq + m_ub)])
    bounds = list(zip(lo, hi, strict=True)) + [(0, None)] * (2 * m_eq + m_ub)
    res = linprog(cost, A_ub=A_ub_e, b_ub=b_ub, A_eq=A_eq_e, b_eq=b_eq, bounds=bounds, method="highs")
    if res.status != 0:
        return None, None
    return (res.eqlin.marginals if m_eq else None), (res.ineqlin.marginals if m_ub else None)


__all__ = ["CertifiedLPResult", "PSDBlock", "certified_lp_minimum", "certified_psd", "dual_bound"]
