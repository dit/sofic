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
import math
import time
from dataclasses import dataclass
from fractions import Fraction

import numpy as np
import scipy.sparse as sp
from scipy.optimize import linprog

from sofic.generators.bidirectional_epsilon_machine import BidirectionalEpsilonMachine
from sofic.generators.certificates import CertifiedLPResult, certified_lp_minimum, dual_bound
from sofic.generators.coupled_generators import _GeneratorProblem
from sofic.generators.coverage_bound import CoverageLowerResult, coverage_entropy_lower, mu_caps

_LP_SLACK = 1e-7
_HIGHS_INFEASIBLE = 2

Boxes = tuple[tuple[np.ndarray, ...], ...]
Path = tuple[tuple[int, int, int, int], ...]


class _HeaviestStatesRelaxation:
    """LP over moments of ``k`` explicit states (groups ``0 .. k-1``) and an aggregate (group ``k``)."""

    def __init__(self, problem: _GeneratorProblem, k: int, *, data_denominator: int | None = None) -> None:
        na, nb, ns, G = problem.n_plus, problem.n_minus, problem.n_sym, k + 1
        self.k, self.na, self.nb = k, na, nb
        P = problem.lifted.plus.T @ (problem.p_pair[:, None] * problem.lifted.minus)
        Pe = problem.Pe
        if data_denominator is not None:
            P, Pe = (_rationalize(A, data_denominator) for A in (P, problem.Pe))
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
        self.var = var
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
            eq([(int(T[g, a, b]), 1.0) for g in groups], P[a, b])
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
            eq([(int(Gt[x, g, h, a, b]), 1.0) for g in groups for h in groups], Pe[a, x, b])
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
        self.b_eq = np.array([float(v) for v in eq_rhs])
        self.b_eq_exact = [v if isinstance(v, Fraction) else Fraction(float(v)) for v in eq_rhs]

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
        res = linprog(
            self.c,
            A_ub=self.inequalities(boxes),
            b_ub=np.zeros(self.n_ub),
            A_eq=self.A_eq,
            b_eq=self.b_eq,
            bounds=(0, 1),
            method="highs",
        )
        if res.status == _HIGHS_INFEASIBLE:
            return None
        if not res.success:
            return 1.0
        return float(-res.fun)

    def certified_bound(self, boxes: tuple[tuple[np.ndarray, ...], ...]) -> CertifiedLPResult:
        """:meth:`bound` with an exact dual certificate (the LP minimizes minus the weight)."""
        return certified_lp_minimum(
            self.c,
            np.zeros(self.n),
            np.ones(self.n),
            A_eq=self.A_eq,
            b_eq=self.b_eq_exact,
            A_ub=self.inequalities(boxes),
            b_ub=np.zeros(self.n_ub),
        )

    def inequalities(self, boxes: tuple[tuple[np.ndarray, ...], ...]) -> sp.csr_matrix:
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
        return sp.csr_matrix(
            (np.concatenate([fd, coef]), (np.concatenate([fr, self.slot_rows]), np.concatenate([fc, self.slot_cols]))),
            shape=(self.n_ub, self.n),
        )


def _rationalize(A: np.ndarray, denominator: int) -> np.ndarray:
    """Nearest rationals with denominator at most ``denominator`` (object array of :class:`~fractions.Fraction`)."""
    out = np.empty(A.shape, dtype=object)
    for idx, v in np.ndenumerate(A):
        out[idx] = Fraction(float(v)).limit_denominator(denominator)
    return out


def _sparse(rows: list[list[tuple[int, float]]], n: int) -> sp.csr_matrix:
    r = [i for i, terms in enumerate(rows) for _ in terms]
    c = [j for terms in rows for j, _ in terms]
    d = [v for terms in rows for _, v in terms]
    return sp.csr_matrix((d, (r, c)), shape=(len(rows), n))


def _outward(value: Fraction, down: bool) -> float:
    """``value`` rounded to a float in the given direction."""
    f = float(value)
    if down and Fraction(f) > value:
        return math.nextafter(f, -math.inf)
    if not down and Fraction(f) < value:
        return math.nextafter(f, math.inf)
    return f


def _tighten(lo: np.ndarray, hi: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    """Shrink a box to its intersection with the probability simplex (``None`` if empty).

    The new bounds are computed exactly and rounded outward, so no point of the simplex in the box is lost.
    """
    for _ in range(3):
        exact_lo = [Fraction(float(v)) for v in lo]
        sum_lo = sum(exact_lo, Fraction(0))
        hi = np.minimum(hi, [_outward(1 - (sum_lo - v), down=False) for v in exact_lo])
        exact_hi = [Fraction(float(v)) for v in hi]
        sum_hi = sum(exact_hi, Fraction(0))
        lo = np.maximum(lo, [_outward(1 - (sum_hi - v), down=True) for v in exact_hi])
    return (lo, hi) if np.all(lo <= hi) else None


def _branch_and_bound(
    relaxation: _HeaviestStatesRelaxation, max_nodes: int, deadline: float | None = None
) -> tuple[list[tuple[float, Boxes, Path]], list[tuple[Boxes, Path]]]:
    """Best-first branching over belief boxes.

    Returns the open leaves ``(bound, boxes, path)`` and the ``(boxes, path)`` dropped because their LP is
    infeasible; a path lists the ``(state, part, coordinate, side)`` splits from the root. Boxes dropped
    by :func:`_tighten` miss the simplex, so the open and infeasible leaves cover every generator.
    """
    k = relaxation.k
    root = _root(relaxation)
    first = relaxation.bound(root)
    if first is None:
        return [], [(root, ())]
    heap: list[tuple[float, int, Boxes, Path]] = [(-first, 0, root, ())]
    infeasible: list[tuple[Boxes, Path]] = []
    order = itertools.count(1)
    nodes = 0
    while heap and nodes < max_nodes and (deadline is None or time.monotonic() < deadline):
        bound, _, boxes, path = heapq.heappop(heap)
        nodes += 1
        widths = [(boxes[e][1] - boxes[e][0], boxes[e][3] - boxes[e][2]) for e in range(k)]
        e, part, i = max(
            ((e, part, i) for e in range(k) for part in (0, 1) for i in range(len(widths[e][part]))),
            key=lambda epi: widths[epi[0]][epi[1]][epi[2]],
        )
        for side in (0, 1):
            child = _split(boxes, e, part, i, side)
            if child is None:
                continue
            child_path = (*path, (e, part, i, side))
            value = relaxation.bound(child)
            if value is None:
                infeasible.append((child, child_path))
            else:
                heapq.heappush(heap, (-min(-bound, value), next(order), child, child_path))
    return [(-b, boxes, path) for b, _, boxes, path in heap], infeasible


def generator_weight_cap(
    bidir: BidirectionalEpsilonMachine | _GeneratorProblem,
    k: int,
    *,
    max_nodes: int = 1000,
    deadline: float | None = None,
) -> float:
    """Certified cap on the total weight of the ``k`` heaviest states of any generator; see the module docstring.

    Branches on the widest belief coordinate of any explicit state; the result is valid at any
    ``max_nodes`` and decreases as it grows. Branching also stops at ``deadline`` (a
    :func:`time.monotonic` value), which leaves a valid but looser cap. It is valid up to the LP tolerance (``1e-7`` is added);
    :func:`certified_generator_weight_cap` removes that caveat.
    """
    problem = bidir if isinstance(bidir, _GeneratorProblem) else _GeneratorProblem(bidir)
    if k <= 0:
        return 0.0
    leaves, _ = _branch_and_bound(_HeaviestStatesRelaxation(problem, k), max_nodes, deadline)
    cap = max((leaf[0] for leaf in leaves), default=0.0)
    return min(1.0, cap + _LP_SLACK)


@dataclass(frozen=True)
class LeafCertificate:
    """One branch-and-bound leaf: its branching ``path`` from the root and exact multipliers.

    ``kind`` is ``"bound"`` (the multipliers bound the leaf's weight by ``value``), ``"farkas"`` (they prove
    the leaf infeasible) or ``"failed"``.
    """

    path: tuple[tuple[int, int, int, int], ...]
    kind: str
    value: Fraction | None
    y_eq: tuple[Fraction, ...] | None
    y_ub: tuple[Fraction, ...] | None


@dataclass(frozen=True)
class WeightCapCertificate:
    """Exact cap from :func:`certified_generator_weight_cap`.

    ``cap`` bounds the total weight of the ``k`` heaviest states of any generator of the process whose
    causal-state joint and edge law are the rationals the certificate used: the floats themselves, or with
    ``data_denominator`` the nearest rationals of bounded denominator, which differ from the floats by at
    most ``data_error``. ``leaves`` open boxes carry exact dual bounds and ``infeasible`` boxes exact
    Farkas certificates; ``failed`` boxes could not be certified, and then ``cap`` is the trivial 1.
    ``records`` lets :func:`verify_weight_cap_certificate` recheck everything without an LP solver.
    """

    k: int
    cap: Fraction
    leaves: int
    infeasible: int
    failed: int
    data_error: float
    data_denominator: int | None = None
    records: tuple[LeafCertificate, ...] = ()


def certified_generator_weight_cap(
    bidir: BidirectionalEpsilonMachine | _GeneratorProblem,
    k: int,
    *,
    max_nodes: int = 1000,
    data_denominator: int | None = None,
) -> WeightCapCertificate:
    """:func:`generator_weight_cap` with every branch-and-bound leaf verified in exact arithmetic.

    Each open leaf's LP gets a rigorous bound from rationalized HiGHS duals and each infeasible leaf an
    exact Farkas certificate (:mod:`~sofic.generators.certificates`, :cite:`neumaier2004safe`). The box
    tightening rounds outward, so the leaves cover every generator and the maximum leaf bound is a cap
    with no floating-point caveat.

    When the true edge law is rational with denominators at most ``D``, pass ``data_denominator=D``: two
    such rationals differ by at least ``1 / D**2``, so if ``data_error < 1 / (2 D**2)`` the rationalized
    data are the true data.
    """
    problem = bidir if isinstance(bidir, _GeneratorProblem) else _GeneratorProblem(bidir)
    P = problem.lifted.plus.T @ (problem.p_pair[:, None] * problem.lifted.minus)
    data_error = 0.0
    if data_denominator is not None:
        data_error = max(
            float(np.max(np.abs(np.asarray(_rationalize(A, data_denominator), dtype=float) - A)))
            for A in (P, problem.Pe)
        )
    if k <= 0:
        return WeightCapCertificate(k, Fraction(0), 0, 0, 0, data_error, data_denominator)
    relaxation = _HeaviestStatesRelaxation(problem, k, data_denominator=data_denominator)
    leaves, infeasible = _branch_and_bound(relaxation, max_nodes)
    cap, records = Fraction(0), []
    for boxes, path in [*((b, p) for _, b, p in leaves), *infeasible]:
        result = relaxation.certified_bound(boxes)
        if result.status == "optimal" and result.bound is not None:
            cap = max(cap, -result.bound)
            records.append(LeafCertificate(path, "bound", -result.bound, result.y_eq, result.y_ub))
        elif result.status == "infeasible":
            records.append(LeafCertificate(path, "farkas", None, result.y_eq, result.y_ub))
        else:
            records.append(LeafCertificate(path, "failed", None, None, None))
    proved = sum(r.kind == "farkas" for r in records)
    failed = sum(r.kind == "failed" for r in records)
    if failed:
        cap = Fraction(1)
    return WeightCapCertificate(
        k, min(cap, Fraction(1)), len(leaves), proved, failed, data_error, data_denominator, tuple(records)
    )


def _root(relaxation: _HeaviestStatesRelaxation) -> Boxes:
    na, nb = relaxation.na, relaxation.nb
    return tuple((np.zeros(na), np.ones(na), np.zeros(nb), np.ones(nb)) for _ in range(relaxation.k))


def _split(boxes: Boxes, e: int, part: int, i: int, side: int) -> Boxes | None:
    """Child ``side`` of ``boxes`` split at the midpoint of coordinate ``i`` of explicit state ``e``, tightened."""
    lq, hq, lr, hr = (v.copy() for v in boxes[e])
    lo, hi = (lq, hq) if part == 0 else (lr, hr)
    mid = 0.5 * (lo[i] + hi[i])
    (hi if side == 0 else lo)[i] = mid
    tightened = _tighten(lo, hi)
    if tightened is None:
        return None
    box = (*tightened, lr, hr) if part == 0 else (lq, hq, *tightened)
    return (*boxes[:e], box, *boxes[e + 1 :])


def verify_weight_cap_certificate(
    bidir: BidirectionalEpsilonMachine | _GeneratorProblem, certificate: WeightCapCertificate
) -> bool:
    """Recheck a :class:`WeightCapCertificate` in exact arithmetic, without an LP solver.

    Replays each leaf's path from the root box, checks that the leaves tile the root (a missing sibling
    must miss the simplex after exact tightening), and re-evaluates every stored dual bound and Farkas
    certificate with :func:`~sofic.generators.certificates.dual_bound`.
    """
    if certificate.failed or not certificate.records:
        return certificate.k <= 0
    problem = bidir if isinstance(bidir, _GeneratorProblem) else _GeneratorProblem(bidir)
    relaxation = _HeaviestStatesRelaxation(problem, certificate.k, data_denominator=certificate.data_denominator)
    zeros, ones = np.zeros(relaxation.n), np.ones(relaxation.n)
    paths = {r.path for r in certificate.records}
    splits: dict[Path, tuple[int, int, int]] = {}
    for path in paths:
        for depth, (e, part, i, _) in enumerate(path):
            if splits.setdefault(path[:depth], (e, part, i)) != (e, part, i):
                return False
    if any(path[:depth] in paths for path in paths for depth in range(len(path))):
        return False
    boxes_at: dict[Path, Boxes | None] = {(): _root(relaxation)}

    def box(path: Path) -> Boxes | None:
        if path not in boxes_at:
            parent = box(path[:-1])
            boxes_at[path] = None if parent is None else _split(parent, *path[-1])
        return boxes_at[path]

    for prefix, (e, part, i) in splits.items():
        for side in (0, 1):
            child = (*prefix, (e, part, i, side))
            if child not in paths and child not in splits and box(child) is not None:
                return False
    for record in certificate.records:
        boxes = box(record.path)
        if boxes is None:
            continue
        farkas = record.kind != "bound"
        value = dual_bound(
            np.zeros(relaxation.n) if farkas else relaxation.c,
            zeros,
            ones,
            A_eq=relaxation.A_eq,
            b_eq=relaxation.b_eq_exact,
            y_eq=record.y_eq,
            A_ub=relaxation.inequalities(boxes),
            b_ub=np.zeros(relaxation.n_ub),
            y_ub=record.y_ub,
        )
        if value is None or (value <= 0 if farkas else -value > certificate.cap):
            return False
    return True


def generator_coverage_bound(
    bidir: BidirectionalEpsilonMachine | _GeneratorProblem,
    *,
    k_max: int = 8,
    explicit: int = 2,
    max_nodes: tuple[int, ...] = (1000, 2000),
    deadline: float | None = None,
) -> CoverageLowerResult:
    """Coverage lower bound on :math:`C_g` from :func:`~sofic.generators.coverage_bound.mu_caps` tightened by
    :func:`generator_weight_cap` for the ``explicit`` heaviest states.

    The caps on the ``j`` heaviest states for ``j > explicit`` use subadditivity, ``cap_j <= cap_{j-1} + cap_1``.
    ``deadline`` (a :func:`time.monotonic` value) is split evenly between the explicit caps still to run;
    every cap stays valid, only looser.
    """
    problem = bidir if isinstance(bidir, _GeneratorProblem) else _GeneratorProblem(bidir)
    P = problem.lifted.plus.T @ (problem.p_pair[:, None] * problem.lifted.minus)
    k_used = max(1, min(int(k_max), *P.shape))
    caps = list(mu_caps(P, k=k_used)[0])
    n_explicit = min(explicit, k_used)
    for j in range(1, n_explicit + 1):
        budget = max_nodes[min(j, len(max_nodes)) - 1]
        stop = None
        if deadline is not None:
            now = time.monotonic()
            stop = now + max(0.0, deadline - now) / (n_explicit - j + 1)
        caps[j - 1] = min(caps[j - 1], generator_weight_cap(problem, j, max_nodes=budget, deadline=stop))
    for j in range(1, k_used):
        caps[j] = min(caps[j], caps[j - 1] + caps[0])
    for j in range(k_used - 2, -1, -1):
        caps[j] = min(caps[j], caps[j + 1])
    return CoverageLowerResult(coverage_entropy_lower(caps), tuple(caps), True, caps[0], k_used)


__all__ = [
    "LeafCertificate",
    "WeightCapCertificate",
    "certified_generator_weight_cap",
    "generator_coverage_bound",
    "generator_weight_cap",
    "verify_weight_cap_certificate",
]
