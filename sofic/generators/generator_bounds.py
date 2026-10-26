r"""Lower bounds on exact common information and generative complexity from the support of p(s+, s-).

Every value :math:`w` of a variable separating the forward and reverse causal states has
:math:`p(s^+, s^- \mid w) = q_w(s^+)\, r_w(s^-)`, so its support is a rectangle inside the support of the
causal-state joint. Assign each value to a maximal rectangle containing its support (a function of
:math:`W`); the type masses :math:`m_T` then satisfy, for every family :math:`\mathcal T'` of maximal rectangles,

.. math::

   p\Bigl(\text{cells covered only by } \mathcal T'\Bigr) \;\le\; \sum_{T \in \mathcal T'} m_T
   \;\le\; p\Bigl(\textstyle\bigcup_{T \in \mathcal T'} T\Bigr),

and :math:`\H{W} \ge \H{m}`. Entropy is concave, so its minimum over this polytope is attained at a vertex.
The bound applies to the exact common information :math:`\op{G}{S^+ : S^-}` (Kumar, Li & El Gamal
:cite:`Kumar2014`) and therefore to the generative complexity :math:`C_g` (Löhr & Ay :cite:`LohrAy2009`),
whose generator states separate the causal states. It is informative only when the support is sparse; for
the :math:`(2, \infty)` run-length-limited process it equals the exact common information.
"""

from __future__ import annotations

import itertools
from collections.abc import Hashable
from fractions import Fraction
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from sofic.generators.bidirectional_epsilon_machine import BidirectionalEpsilonMachine


def maximal_rectangles(support: set[tuple[Hashable, Hashable]]) -> list[tuple[frozenset, frozenset]]:
    """Maximal rectangles ``U x V`` contained in a set of cells (maximal bicliques), by closure of row sets."""
    rows = sorted({a for a, _ in support}, key=repr)
    cols_of = {a: frozenset(b for a2, b in support if a2 == a) for a in rows}
    found = set()
    for size in range(1, len(rows) + 1):
        for subset in itertools.combinations(rows, size):
            V = frozenset.intersection(*(cols_of[a] for a in subset))
            if not V:
                continue
            U = frozenset(a for a in rows if cols_of[a] >= V)
            found.add((U, V))
    return sorted(found, key=lambda t: (sorted(map(repr, t[0])), sorted(map(repr, t[1]))))


def _solve(M: list[list[Fraction]], r: list[Fraction]) -> list[Fraction] | None:
    n = len(M)
    A = [row[:] + [r[i]] for i, row in enumerate(M)]
    for col in range(n):
        piv = next((i for i in range(col, n) if A[i][col] != 0), None)
        if piv is None:
            return None
        A[col], A[piv] = A[piv], A[col]
        p = A[col][col]
        A[col] = [v / p for v in A[col]]
        for i in range(n):
            if i != col and A[i][col] != 0:
                f = A[i][col]
                A[i] = [a - f * b for a, b in zip(A[i], A[col], strict=False)]
    return [A[i][n] for i in range(n)]


def support_type_bound(
    joint: dict[tuple[Hashable, Hashable], float], *, max_denominator: int = 10**6
) -> tuple[float, list[tuple[frozenset, frozenset]], tuple[Fraction, ...]]:
    """Lower bound on ``G[S+ : S-]`` (hence on ``C_g``) from the rectangle types of the support.

    ``joint`` maps causal-state pairs to probabilities (rationalized with ``max_denominator``). Returns the bound
    (bits), the maximal rectangles, and a minimizing vertex of type masses.
    """
    p = {cell: Fraction(v).limit_denominator(max_denominator) for cell, v in joint.items() if v > 0}
    total = sum(p.values())
    p = {cell: v / total for cell, v in p.items()}
    types = maximal_rectangles(set(p))
    n = len(types)
    contains = [{cell for cell in p if cell[0] in U and cell[1] in V} for U, V in types]
    ineq = [([-Fraction(int(j == i)) for j in range(n)], Fraction(0)) for i in range(n)]
    for size in range(1, n + 1):
        for fam in itertools.combinations(range(n), size):
            union = set().union(*(contains[i] for i in fam))
            only = {cell for cell in union if all(i in fam for i in range(n) if cell in contains[i])}
            row = [Fraction(int(j in fam)) for j in range(n)]
            ineq.append((row, sum((p[c] for c in union), Fraction(0))))
            ineq.append(([-x for x in row], -sum((p[c] for c in only), Fraction(0))))
    best = (np.inf, None)
    for S in itertools.combinations(range(len(ineq)), n - 1):
        M = [[Fraction(1)] * n] + [ineq[i][0] for i in S]
        r = [Fraction(1)] + [ineq[i][1] for i in S]
        x = _solve(M, r)
        if x is None or not all(sum(a * xi for a, xi in zip(row, x, strict=False)) <= v for row, v in ineq):
            continue
        h = -sum(float(m) * np.log2(float(m)) for m in x if m > 0)
        if h < best[0]:
            best = (h, tuple(x))
    if best[1] is None:
        raise ValueError("support_type_bound: the type polytope has no vertex")
    return float(best[0]), types, best[1]


def generator_support_bound(bidir: BidirectionalEpsilonMachine) -> float:
    """:func:`support_type_bound` of a bidirectional ε-machine's causal-state joint distribution."""
    return support_type_bound({k: float(v) for k, v in bidir.joint_distribution().items()})[0]


__all__ = ["generator_support_bound", "maximal_rectangles", "support_type_bound"]
