r"""Certified caps on the weights of a generator's heaviest states, and the coverage bound they give.

A generator's states :math:`w` carry rank-1 terms :math:`\pi_w Q_w R_w^\top` of the causal-state joint
:math:`p(s^+, s^-)`, so the :math:`k` heaviest weigh at most :math:`\mu_k`
(:mod:`~sofic.generators.coverage_bound`). The generator equations of
:mod:`~sofic.generators.coupled_generators` (edge law, forward stationarity, reverse consistency)
constrain them further. :func:`generator_weight_cap` keeps ``k`` states explicit, aggregates every
other state into one group, and writes those equations for per-group and per-group-pair moments
(masses, belief moments, flows). Products of an explicit state's beliefs with flows are relaxed by
McCormick envelopes :cite:`mccormick1976computability` over a box of beliefs, and the boxes are
branched. Every generator gives a feasible point (its ``k`` heaviest states explicit, the rest
aggregated), so the largest open bound caps the total weight of the ``k`` heaviest states at any
node budget. Unlike :math:`\mu_k` these caps use the dynamics, so the coverage bound they feed
(:func:`generator_coverage_bound`) can exceed :math:`\op{G}{S^+ : S^-}`.
"""

from __future__ import annotations

import heapq
import itertools

import numpy as np
import scipy.sparse as sp
from scipy.optimize import linprog

from sofic.generators.bidirectional_epsilon_machine import BidirectionalEpsilonMachine
from sofic.generators.coupled_generators import _GeneratorProblem
from sofic.generators.coverage_bound import CoverageLowerResult, coverage_entropy_lower, mu_caps

_LP_SLACK = 1e-7
_HIGHS_INFEASIBLE = 2


class _HeaviestStatesRelaxation:
    """LP over moments of ``k`` explicit states (groups ``0 .. k-1``) and an aggregate (group ``k``)."""

    def __init__(self, problem: _GeneratorProblem, k: int) -> None:
        na, nb, ns, G = problem.n_plus, problem.n_minus, problem.n_sym, k + 1
        self.k, self.na, self.nb = k, na, nb
        P = problem.lifted.plus.T @ (problem.p_pair[:, None] * problem.lifted.minus)
        sizes = {
            "m": (G,),
            "y": (G, na),
            "z": (G, nb),
            "T": (G, na, nb),
            "phi": (ns, G, G),
            "Fl": (ns, G, G, na),
            "Hr": (ns, G, G, nb),
            "Gt": (ns, G, G, na, nb),
        }
        offset, var = 0, {}
        for name, shape in sizes.items():
            var[name] = np.arange(offset, offset + int(np.prod(shape))).reshape(shape)
            offset += int(np.prod(shape))
        self.n = offset
        m, y, z, T = var["m"], var["y"], var["z"], var["T"]
        phi, Fl, Hr, Gt = var["phi"], var["Fl"], var["Hr"], var["Gt"]

        eq_rows: list[list[tuple[int, float]]] = []
        eq_rhs: list[float] = []

        def eq(terms: list[tuple[int, float]], rhs: float) -> None:
            eq_rows.append(terms)
            eq_rhs.append(rhs)

        def total(indices, minus) -> list[tuple[int, float]]:
            return [(int(i), 1.0) for i in np.ravel(indices)] + [(int(minus), -1.0)]

        groups, symbols = range(G), range(ns)
        eq([(int(i), 1.0) for i in m], 1.0)
        for a, b in itertools.product(range(na), range(nb)):
            eq([(int(T[g, a, b]), 1.0) for g in groups], float(P[a, b]))
        for g in groups:
            eq(total(y[g], m[g]), 0.0)
            eq(total(z[g], m[g]), 0.0)
            for a in range(na):
                eq(total(T[g, a], y[g, a]), 0.0)
            for b in range(nb):
                eq(total(T[g, :, b], z[g, b]), 0.0)
            eq(total(phi[:, g, :], m[g]), 0.0)
            eq(total(phi[:, :, g], m[g]), 0.0)
            for a in range(na):
                eq(total(Fl[:, g, :, a], y[g, a]), 0.0)
            for b in range(nb):
                eq(total(Hr[:, :, g, b], z[g, b]), 0.0)
        for x, g, h in itertools.product(symbols, groups, groups):
            eq(total(Fl[x, g, h], phi[x, g, h]), 0.0)
            eq(total(Hr[x, g, h], phi[x, g, h]), 0.0)
            for a in range(na):
                eq(total(Gt[x, g, h, a], Fl[x, g, h, a]), 0.0)
            for b in range(nb):
                eq(total(Gt[x, g, h, :, b], Hr[x, g, h, b]), 0.0)
        for a, x, b in itertools.product(range(na), symbols, range(nb)):
            eq([(int(Gt[x, g, h, a, b]), 1.0) for g in groups for h in groups], float(problem.Pe[a, x, b]))
        for h, a2 in itertools.product(groups, range(na)):
            sources = [(a, x) for a in range(na) for x in symbols if problem.F[a, x, a2]]
            eq([(int(Fl[x, g, h, a]), 1.0) for a, x in sources for g in groups] + [(int(y[h, a2]), -1.0)], 0.0)
        for g, b in itertools.product(groups, range(nb)):
            targets = [(x, b2) for x in symbols for b2 in range(nb) if problem.G[x, b2, b]]
            eq([(int(Hr[x, g, h, b2]), 1.0) for x, b2 in targets for h in groups] + [(int(z[g, b]), -1.0)], 0.0)
            for a in range(na):
                terms = [(int(Gt[x, g, h, a, b2]), 1.0) for x, b2 in targets for h in groups]
                eq(terms + [(int(T[g, a, b]), -1.0)], 0.0)
        self.A_eq = _sparse(eq_rows, self.n)
        self.b_eq = np.array(eq_rhs)

        # Inequalities: explicit states ordered by weight, then McCormick pairs whose coefficient is a
        # box bound of one explicit state's belief, recorded per row so a node only rewrites those.
        rows, cols, data = [], [], []
        slots: list[tuple[int, int, bool, int, float]] = []  # (row, factor column, is_q, index, sign)
        r = 0
        for e in range(k - 1):
            rows += [r, r]
            cols += [int(m[e + 1]), int(m[e])]
            data += [1.0, -1.0]
            r += 1

        def envelope(prod: int, factor: int, e: int, is_q: bool, i: int) -> None:
            nonlocal r
            for sign in (-1.0, 1.0):
                rows.append(r)
                cols.append(int(prod))
                data.append(sign)
                slots.append((r, int(factor), is_q, e * max(na, nb) + i, sign))
                r += 1

        for e in range(k):
            for a in range(na):
                envelope(y[e, a], m[e], e, True, a)
                for b in range(nb):
                    envelope(T[e, a, b], z[e, b], e, True, a)
            for b in range(nb):
                envelope(z[e, b], m[e], e, False, b)
                for a in range(na):
                    envelope(T[e, a, b], y[e, a], e, False, b)
            for x, h in itertools.product(symbols, groups):
                for a in range(na):
                    envelope(Fl[x, e, h, a], phi[x, e, h], e, True, a)
                    for b in range(nb):
                        envelope(Gt[x, e, h, a, b], Hr[x, e, h, b], e, True, a)
            for x, g in itertools.product(symbols, groups):
                for b in range(nb):
                    envelope(Hr[x, g, e, b], phi[x, g, e], e, False, b)
                    for a in range(na):
                        envelope(Gt[x, g, e, a, b], Fl[x, g, e, a], e, False, b)
        self.n_ub = r
        self.fixed = (np.array(rows), np.array(cols), np.array(data))
        self.slot_rows = np.array([s[0] for s in slots])
        self.slot_cols = np.array([s[1] for s in slots])
        self.slot_is_q = np.array([s[2] for s in slots])
        self.slot_index = np.array([s[3] for s in slots])
        self.slot_sign = np.array([s[4] for s in slots])
        self.c = np.zeros(self.n)
        self.c[m[:k]] = -1.0

    def bound(self, boxes: tuple[tuple[np.ndarray, ...], ...]) -> float | None:
        """Max total weight of the explicit states with beliefs in ``boxes``; ``None`` if the box is infeasible."""
        width = max(self.na, self.nb)

        def flat(part: int, n: int) -> np.ndarray:
            out = np.zeros(self.k * width)
            for e, box in enumerate(boxes):
                out[e * width : e * width + n] = box[part]
            return out

        lo = np.where(self.slot_is_q, flat(0, self.na)[self.slot_index], flat(2, self.nb)[self.slot_index])
        hi = np.where(self.slot_is_q, flat(1, self.na)[self.slot_index], flat(3, self.nb)[self.slot_index])
        # sign -1: lo * factor - prod <= 0;  sign +1: prod - hi * factor <= 0
        coef = np.where(self.slot_sign < 0, lo, -hi)
        fr, fc, fd = self.fixed
        A_ub = sp.csr_matrix(
            (np.concatenate([fd, coef]), (np.concatenate([fr, self.slot_rows]), np.concatenate([fc, self.slot_cols]))),
            shape=(self.n_ub, self.n),
        )
        res = linprog(
            self.c, A_ub=A_ub, b_ub=np.zeros(self.n_ub), A_eq=self.A_eq, b_eq=self.b_eq, bounds=(0, 1), method="highs"
        )
        if res.status == _HIGHS_INFEASIBLE:
            return None
        if not res.success:
            return 1.0
        return float(-res.fun)


def _sparse(rows: list[list[tuple[int, float]]], n: int) -> sp.csr_matrix:
    r = [i for i, terms in enumerate(rows) for _ in terms]
    c = [j for terms in rows for j, _ in terms]
    d = [v for terms in rows for _, v in terms]
    return sp.csr_matrix((d, (r, c)), shape=(len(rows), n))


def _tighten(lo: np.ndarray, hi: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    """Shrink a box to its intersection with the probability simplex (``None`` if empty)."""
    for _ in range(3):
        hi = np.minimum(hi, 1.0 - (lo.sum() - lo))
        lo = np.maximum(lo, 1.0 - (hi.sum() - hi))
    return (lo, hi) if np.all(lo <= hi + 1e-12) else None


def generator_weight_cap(
    bidir: BidirectionalEpsilonMachine | _GeneratorProblem,
    k: int,
    *,
    max_nodes: int = 1000,
) -> float:
    """Certified cap on the total weight of the ``k`` heaviest states of any generator; see the module docstring.

    Branches on the widest belief coordinate of any explicit state; the result is valid at any
    ``max_nodes`` and decreases as it grows.
    """
    problem = bidir if isinstance(bidir, _GeneratorProblem) else _GeneratorProblem(bidir)
    if k <= 0:
        return 0.0
    relaxation = _HeaviestStatesRelaxation(problem, k)
    na, nb = relaxation.na, relaxation.nb
    root = tuple((np.zeros(na), np.ones(na), np.zeros(nb), np.ones(nb)) for _ in range(k))
    first = relaxation.bound(root)
    if first is None:
        return 0.0
    heap = [(-first, 0, root)]
    order = itertools.count(1)
    nodes = 0
    while heap and nodes < max_nodes:
        bound, _, boxes = heapq.heappop(heap)
        nodes += 1
        widths = [(boxes[e][1] - boxes[e][0], boxes[e][3] - boxes[e][2]) for e in range(k)]
        e, part, i = max(
            ((e, part, i) for e in range(k) for part in (0, 1) for i in range(len(widths[e][part]))),
            key=lambda epi: widths[epi[0]][epi[1]][epi[2]],
        )
        for side in (0, 1):
            lq, hq, lr, hr = (v.copy() for v in boxes[e])
            lo, hi = (lq, hq) if part == 0 else (lr, hr)
            mid = 0.5 * (lo[i] + hi[i])
            (hi if side == 0 else lo)[i] = mid
            tightened = _tighten(lo, hi)
            if tightened is None:
                continue
            box = (*tightened, lr, hr) if part == 0 else (lq, hq, *tightened)
            child = (*boxes[:e], box, *boxes[e + 1 :])
            value = relaxation.bound(child)
            if value is not None:
                heapq.heappush(heap, (-min(-bound, value), next(order), child))
    cap = -heap[0][0] if heap else 0.0
    return min(1.0, cap + _LP_SLACK)


def generator_coverage_bound(
    bidir: BidirectionalEpsilonMachine | _GeneratorProblem,
    *,
    k_max: int = 8,
    explicit: int = 2,
    max_nodes: tuple[int, ...] = (1000, 2000),
) -> CoverageLowerResult:
    """Coverage lower bound on :math:`C_g` from :func:`~sofic.generators.coverage_bound.mu_caps` tightened by
    :func:`generator_weight_cap` for the ``explicit`` heaviest states.

    The caps on the ``j`` heaviest states for ``j > explicit`` use subadditivity, ``cap_j <= cap_{j-1} + cap_1``.
    """
    problem = bidir if isinstance(bidir, _GeneratorProblem) else _GeneratorProblem(bidir)
    P = problem.lifted.plus.T @ (problem.p_pair[:, None] * problem.lifted.minus)
    k_used = max(1, min(int(k_max), *P.shape))
    caps = list(mu_caps(P, k=k_used)[0])
    for j in range(1, min(explicit, k_used) + 1):
        budget = max_nodes[min(j, len(max_nodes)) - 1]
        caps[j - 1] = min(caps[j - 1], generator_weight_cap(problem, j, max_nodes=budget))
    for j in range(1, k_used):
        caps[j] = min(caps[j], caps[j - 1] + caps[0])
    for j in range(k_used - 2, -1, -1):
        caps[j] = min(caps[j], caps[j + 1])
    return CoverageLowerResult(coverage_entropy_lower(caps), tuple(caps), True, caps[0], k_used)


__all__ = ["generator_coverage_bound", "generator_weight_cap"]
