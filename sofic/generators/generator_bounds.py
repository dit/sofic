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
from collections.abc import Set as AbstractSet
from dataclasses import dataclass
from fractions import Fraction
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from sofic.generators.bidirectional_epsilon_machine import BidirectionalEpsilonMachine


def maximal_rectangles(support: AbstractSet[tuple[Hashable, Hashable]]) -> list[tuple[frozenset, frozenset]]:
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


def _polytope_vertices(ineq: list[tuple[list[Fraction], Fraction]], n: int) -> set[tuple[Fraction, ...]]:
    """Vertices of ``{x >= 0 : A x <= b}`` intersected with ``sum x = 1`` (first row normalization)."""
    out: set[tuple[Fraction, ...]] = set()
    for subset in itertools.combinations(range(len(ineq)), n - 1):
        M = [[Fraction(1)] * n] + [ineq[i][0] for i in subset]
        r = [Fraction(1)] + [ineq[i][1] for i in subset]
        x = _solve(M, r)
        if x is None:
            continue
        if all(sum(a * xi for a, xi in zip(row, x, strict=False)) <= v for row, v in ineq):
            out.add(tuple(x))
    return out


def _entropy_type_masses(masses: tuple[Fraction, ...]) -> float:
    return float(-sum(float(m) * np.log2(float(m)) for m in masses if m > 0))


def _min_entropy_over_inequalities(
    ineq: list[tuple[list[Fraction], Fraction]], n: int
) -> tuple[float, tuple[Fraction, ...] | None]:
    best = (np.inf, None)
    for vertex in _polytope_vertices(ineq, n):
        h = _entropy_type_masses(vertex)
        if h < best[0]:
            best = (h, vertex)
    if best[1] is None:
        raise ValueError("polytope has no vertex")
    return float(best[0]), best[1]


@dataclass(frozen=True)
class FlowBoundResult:
    """Flow lower bound on :math:`C_g` with optional polytope vertex certification."""

    value: float
    certified: bool
    method: str


def _le(ineq: list[tuple[list[Fraction], Fraction]], row: list[Fraction], v: Fraction) -> None:
    ineq.append((row, v))


def _rll_generator_polytope(
    thresholds: tuple[float, ...],
) -> tuple[list[tuple[list[Fraction], Fraction]], int]:
    """Inequalities for the (2, inf) generator-class polytope (cpfci ``rll_certificate``)."""
    ta = tuple(Fraction(t).limit_denominator(10**6) for t in thresholds)
    tb = ta
    lba, uba = [Fraction(0), *ta], [*ta, Fraction(1)]
    lbb, ubb = [Fraction(0), *tb], [*tb, Fraction(1)]
    na, nb = len(uba), len(ubb)
    n = na + nb + 1
    ineq: list[tuple[list[Fraction], Fraction]] = []
    for i in range(n):
        _le(ineq, [-Fraction(int(j == i)) for j in range(n)], Fraction(0))
    ra = [Fraction(1)] * na + [Fraction(0)] * nb + [Fraction(0)]
    rb = [Fraction(0)] * na + [Fraction(1)] * nb + [Fraction(0)]
    _le(ineq, [-x for x in ra], Fraction(-1, 4))
    _le(ineq, ra, Fraction(1, 2))
    _le(ineq, [-x for x in rb], Fraction(-1, 4))
    _le(ineq, rb, Fraction(1, 2))
    _le(ineq, [Fraction(0)] * (n - 1) + [Fraction(-1)], Fraction(-1, 8))
    for lam in list(ta) + [Fraction(1)]:
        _le(
            ineq,
            [-max(u - lam, Fraction(0)) for u in uba] + [Fraction(0)] * nb + [Fraction(-lam)],
            Fraction(-1, 8),
        )
    for lam in list(tb) + [Fraction(1)]:
        _le(
            ineq,
            [Fraction(0)] * na + [-max(u - lam, Fraction(0)) for u in ubb] + [Fraction(-lam)],
            Fraction(-1, 8),
        )
    _le(ineq, [-u for u in uba] + [Fraction(0)] * nb + [Fraction(0)], Fraction(-1, 4))
    _le(ineq, lba + [Fraction(0)] * nb + [Fraction(0)], Fraction(1, 4))
    _le(ineq, [Fraction(0)] * na + [-u for u in ubb] + [Fraction(0)], Fraction(-1, 4))
    _le(ineq, [Fraction(0)] * na + lbb + [Fraction(0)], Fraction(1, 4))
    return ineq, n


def rll_flow_bound_certified(*, thresholds: tuple[float, ...] = (0.55, 0.65)) -> float:
    """Exact-minimum entropy over the (2, inf) generator-class polytope (rational vertex enumeration)."""
    ineq, n = _rll_generator_polytope(thresholds)
    value, _ = _min_entropy_over_inequalities(ineq, n)
    return value


def is_rll_three_state_machine(bidir: BidirectionalEpsilonMachine, *, atol: float = 1e-9) -> bool:
    """Whether ``bidir`` is the length-3 bidirectional machine for the (2, inf) process."""
    joint = {pair: float(p) for pair, p in bidir.joint_distribution().items() if float(p) > atol}
    expected = {
        ("A", "D"): 1 / 8,
        ("A", "E"): 1 / 4,
        ("A", "F"): 1 / 8,
        ("B", "D"): 1 / 4,
        ("C", "D"): 1 / 8,
        ("C", "F"): 1 / 8,
    }
    if set(joint) != set(expected):
        return False
    return all(abs(joint[k] - v) <= atol for k, v in expected.items())


def flow_bound_result(
    bidir: BidirectionalEpsilonMachine,
    *,
    thresholds: tuple[float, ...] = (0.55, 0.65),
) -> FlowBoundResult:
    """Flow lower bound; certified on the canonical (2, inf) machine via vertex enumeration."""
    if is_rll_three_state_machine(bidir):
        value = rll_flow_bound_certified(thresholds=thresholds)
        return FlowBoundResult(value, True, "rll generator-class polytope (exact vertices)")
    value = flow_bound(bidir, thresholds=thresholds)
    return FlowBoundResult(value, False, "branch-and-bound on bin masses")


@dataclass(frozen=True)
class SupportTypeBoundResult:
    """Support-type lower bound on :math:`\\op{G}{S^+ : S^-}` with certification metadata."""

    value: float
    certified: bool
    types: list[tuple[frozenset, frozenset]]
    masses: tuple[Fraction, ...] | None
    method: str


def support_exact_cover_bound(
    joint: dict[tuple[Hashable, Hashable], float], *, max_denominator: int = 10**6
) -> SupportTypeBoundResult | None:
    r"""When each support cell lies in exactly one maximal rectangle, :math:`\\H{W}` equals :math:`\\H{m}` for fixed type masses.

    This is an **exact** (certified) lower bound, often tighter than :func:`support_type_bound` when rectangles overlap in the
    relaxation but admit a disjoint assignment.
    """
    p = {cell: Fraction(v).limit_denominator(max_denominator) for cell, v in joint.items() if v > 0}
    if not p:
        return None
    total = sum(p.values())
    p = {cell: v / total for cell, v in p.items()}
    types = maximal_rectangles(set(p))
    contains = [{cell for cell in p if cell[0] in U and cell[1] in V} for U, V in types]
    owner: dict[tuple, int] = {}
    for cell in p:
        holders = [i for i, cells in enumerate(contains) if cell in cells]
        if len(holders) != 1:
            return None
        owner[cell] = holders[0]
    masses = [Fraction(0) for _ in types]
    for cell, mass in p.items():
        masses[owner[cell]] += mass
    masses_t = tuple(masses)
    value = _entropy_type_masses(masses_t)
    return SupportTypeBoundResult(value, True, types, masses_t, "exact rectangle partition")


def support_type_bound_result(
    joint: dict[tuple[Hashable, Hashable], float], *, max_denominator: int = 10**6
) -> SupportTypeBoundResult:
    """Support-type bound with ``certified=True`` (exact rational vertex enumeration on the relaxation polytope)."""
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
    value, vertex = _min_entropy_over_inequalities(ineq, n)
    return SupportTypeBoundResult(value, True, types, vertex, "type-mass polytope vertices")


def support_type_bound(
    joint: dict[tuple[Hashable, Hashable], float], *, max_denominator: int = 10**6
) -> tuple[float, list[tuple[frozenset, frozenset]], tuple[Fraction, ...]]:
    """Lower bound on ``G[S+ : S-]`` (hence on ``C_g``) from the rectangle types of the support."""
    res = support_type_bound_result(joint, max_denominator=max_denominator)
    return res.value, res.types, res.masses or ()


def _concave_entropy_min(
    A_ub: np.ndarray, b_ub: np.ndarray, A_eq: np.ndarray, b_eq: np.ndarray, *, tol: float = 1e-7, max_nodes: int = 20000
) -> float:
    """Minimum of the (concave) entropy of x over a polytope, by branch-and-bound with secant LP bounds.

    On a box ``lo <= x <= hi`` each term ``-x_i log2 x_i`` is at least its secant, so the LP with the secant
    objective bounds the minimum from below; the entropy at the LP solution bounds it from above.
    """
    from scipy.optimize import linprog

    n = A_eq.shape[1]

    def f(t: np.ndarray) -> np.ndarray:
        t = np.clip(t, 0.0, 1.0)
        return np.where(t > 0, -t * np.log2(np.where(t > 0, t, 1.0)), 0.0)

    def solve(lo: np.ndarray, hi: np.ndarray):
        width = np.maximum(hi - lo, 1e-15)
        slope = (f(hi) - f(lo)) / width
        res = linprog(
            slope, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=list(zip(lo, hi, strict=True)), method="highs"
        )
        if not res.success:
            return None
        x = res.x
        return float((f(lo) + slope * (x - lo)).sum()), x

    lo, hi = np.zeros(n), np.ones(n)
    root = solve(lo, hi)
    if root is None:
        return np.inf
    best_upper = float(f(root[1]).sum())
    heap = [(root[0], 0, lo, hi, root[1])]
    import heapq

    counter = 1
    nodes = 0
    while heap:
        bound, _, lo, hi, x = heapq.heappop(heap)
        if bound >= best_upper - tol or nodes > max_nodes:
            return min(bound, best_upper) if nodes > max_nodes else best_upper
        nodes += 1
        width = np.maximum(hi - lo, 1e-15)
        gap = f(x) - (f(lo) + (f(hi) - f(lo)) / width * (x - lo))
        i = int(np.argmax(gap))
        split = x[i] if lo[i] + 1e-12 < x[i] < hi[i] - 1e-12 else (lo[i] + hi[i]) / 2
        for new_lo, new_hi in (
            (lo, np.where(np.arange(n) == i, split, hi)),
            (np.where(np.arange(n) == i, split, lo), hi),
        ):
            child = solve(new_lo, new_hi)
            if child is None:
                continue
            best_upper = min(best_upper, float(f(child[1]).sum()))
            if child[0] < best_upper - tol:
                heapq.heappush(heap, (child[0], counter, new_lo, new_hi, child[1]))
                counter += 1
    return best_upper


def flow_bound(bidir: BidirectionalEpsilonMachine, *, thresholds: tuple[float, ...] = (0.55, 0.65)) -> float:
    r"""Lower bound on the generative complexity :math:`C_g` from support types plus generator flows.

    Generators satisfy, beyond separation, the flow identities of their Markov property: if the forward causal
    state :math:`a'` follows only :math:`a` (on symbol :math:`x`), then
    :math:`p(w') q_{w'}(a') = \sum_w p(w)\, q_w(a)\, K(x, w' \mid w)`, and symmetrically for reverse causal states
    with a unique successor. States of a type that must hold a *forced* cell (one no other type covers) in row
    :math:`a'` are therefore entered from states that give :math:`a` high probability. Binning each type by such a
    belief at ``thresholds`` turns these facts into linear constraints on the bin masses (the inflow bound in its
    LP-dual form); since the bin label is a function of the state, :math:`\H{W}` is at least the minimum entropy of
    the bin masses over the resulting polytope, computed by branch-and-bound. This automates the certificate
    :math:`C_g \ge 1.49702` for the :math:`(2, \infty)` process.
    """
    from sofic.generators.coupled_generators import _GeneratorProblem

    problem = _GeneratorProblem(bidir)
    lifted = problem.lifted
    P = lifted.plus.T @ (lifted.p[:, None] * lifted.minus)
    na, nb = P.shape
    support = {(a, b) for a in range(na) for b in range(nb) if P[a, b] > 1e-14}
    types = maximal_rectangles(support)
    contains = [{c for c in support if c[0] in U and c[1] in V} for U, V in types]
    forced = [{c for c in contains[t] if sum(c in s for s in contains) == 1} for t in range(len(types))]

    # unique-predecessor forward sources (a, a') and unique-successor reverse sources (b, b')
    fwd = [(int(np.argwhere(problem.F[:, :, a2])[0][0]), a2) for a2 in range(na) if problem.F[:, :, a2].sum() == 1]
    rev = [(b, int(np.argwhere(problem.G[:, :, b])[0][1])) for b in range(nb) if problem.G[:, :, b].sum() == 1]
    # candidate binning variables: ("+", a) for forward sources whose target row has forced cells, ("-", b') mirror
    cands = [("+", a) for a, a2 in fwd if any(c[0] == a2 for t in forced for c in t)]
    cands += [("-", b2) for b, b2 in rev if any(c[1] == b for t in forced for c in t)]

    def holds(t: int, var: tuple[str, int]) -> bool:
        U, V = types[t]
        return var[1] in (U if var[0] == "+" else V)

    binned: dict[int, tuple[str, int]] = {}
    for var in sorted(set(cands), key=lambda v: sum(holds(t, v) for t in range(len(types)))):
        for t in range(len(types)):
            if t not in binned and holds(t, var):
                binned[t] = var
    edges = [0.0, *thresholds, 1.0]
    columns = []  # (type, lower bound, upper bound) of the belief binned on, or (type, 0, 1) when unbinned
    for t in range(len(types)):
        if t in binned:
            columns += [(t, edges[i], edges[i + 1]) for i in range(len(edges) - 1)]
        else:
            columns.append((t, 0.0, 1.0))
    n = len(columns)
    A_ub, b_ub = [], []

    def type_row(fam):
        return np.array([1.0 if col[0] in fam else 0.0 for col in columns])

    for size in range(1, len(types) + 1):
        for fam in itertools.combinations(range(len(types)), size):
            union = set().union(*(contains[t] for t in fam))
            only = {c for c in union if all(t in fam for t in range(len(types)) if c in contains[t])}
            A_ub.append(type_row(fam))
            b_ub.append(sum(P[c] for c in union))
            A_ub.append(-type_row(fam))
            b_ub.append(-sum(P[c] for c in only))

    def coeffs(var: tuple[str, int], which: int) -> np.ndarray:
        """Per column: bin bound (which = 1 upper, 0 lower) of the belief var, 1/0 if the type holds var unbinned."""
        out = np.zeros(n)
        for j, (t, lb, ub) in enumerate(columns):
            if not holds(t, var):
                continue
            if binned.get(t) == var:
                out[j] = ub if which else lb
            else:
                out[j] = 1.0 if which else 0.0
        return out

    for var in set(binned.values()):
        marginal = float(P[var[1], :].sum()) if var[0] == "+" else float(P[:, var[1]].sum())
        A_ub.append(coeffs(var, 0))
        b_ub.append(marginal)
        A_ub.append(-coeffs(var, 1))
        b_ub.append(-marginal)

    def flow(var: tuple[str, int], t0: int, need: float) -> None:
        u = coeffs(var, 1)
        own = type_row({t0})
        for lam in sorted(set(u) | {1.0}):
            A_ub.append(-(lam * own + np.maximum(u - lam, 0.0)))
            b_ub.append(-need)

    for a, a2 in fwd:
        for t0 in range(len(types)):
            need = sum(P[c] for c in forced[t0] if c[0] == a2)
            if need > 0:
                flow(("+", a), t0, need)
    for b, b2 in rev:
        for t0 in range(len(types)):
            need = sum(P[c] for c in forced[t0] if c[1] == b)
            if need > 0:
                flow(("-", b2), t0, need)

    return _concave_entropy_min(np.array(A_ub), np.array(b_ub), np.ones((1, n)), np.array([1.0]))


def generator_support_bound(bidir: BidirectionalEpsilonMachine) -> float:
    """:func:`support_type_bound` of a bidirectional ε-machine's causal-state joint distribution."""
    return support_type_bound({k: float(v) for k, v in bidir.joint_distribution().items()})[0]


__all__ = [
    "FlowBoundResult",
    "SupportTypeBoundResult",
    "flow_bound",
    "flow_bound_result",
    "generator_support_bound",
    "is_rll_three_state_machine",
    "maximal_rectangles",
    "rll_flow_bound_certified",
    "support_exact_cover_bound",
    "support_type_bound",
    "support_type_bound_result",
]
