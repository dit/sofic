r"""Certified lower bound on the exact common information by branching on the heaviest term.

Any exact decomposition :math:`P = \sum_w \lambda_w q_w r_w^\top` with heaviest term
:math:`T = t\, q r^\top` has

.. math:: H(\lambda) = h(t) + (1 - t)\, H(\lambda'),

where :math:`h` is the binary entropy and :math:`\lambda'` an exact decomposition of the remainder
:math:`(P - T)/(1 - t)` whose atoms are at most :math:`t / (1 - t)`. Over a box of :math:`(q, r, t)` the
remainder lies entrywise below :math:`\bar R = P - t_{lo}\, q_{lo} r_{lo}^\top`, so (``mu_k`` being monotone)
its ``k`` heaviest atoms weigh at most :math:`\mu_k(\bar R)/(1 - t_{hi})`; the coverage bound with those caps
and the atom cap :math:`t_{hi}/(1 - t_{hi})` bounds :math:`H(\lambda')`, and :math:`h(t) + (1 - t) C` is
concave in :math:`t`, so its minimum over the box is at an endpoint. An optimum needs at most
:math:`2^{\min(|X|, |Y|)} - 1` terms :cite:`Kumar2014`, so :math:`t \ge 1/(2^{\min} - 1)`. Best-first
branch and bound over the boxes gives a lower bound valid at any node budget; it is tight wherever the
remainder of an optimum is covered exactly by its caps (e.g. two remaining terms).
"""

from __future__ import annotations

import heapq
import itertools
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from sofic.generators.coverage_bound import mu1_vertex, mu_k_spectral_cap
from sofic.generators.generator_caps import _tighten

if TYPE_CHECKING:
    from sofic.generators.coupled_generators import _GeneratorProblem

Box = tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float]


@dataclass(frozen=True)
class HeaviestTermBoundResult:
    """``value`` bounds :math:`\\op{G}{X : Y}` from below; ``reached`` says whether it met ``target``."""

    value: float
    nodes: int
    open_boxes: int
    reached: bool


def _entropy(p: np.ndarray) -> float:
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def capped_coverage_entropy(caps: list[float], atom: float, n_max: int = 64) -> float:
    """Lower bound on :math:`H(\\pi)` when the ``k`` heaviest atoms weigh at most ``caps[k-1]`` and every atom
    at most ``atom``.

    The ``k``-th partial sum of the sorted vector is at most :math:`S_k = \\min(C_k, k\\,\\mathrm{atom}, 1)`, with
    :math:`C` the least concave majorant of :math:`(0, \\mathrm{caps}, 1, 1, \\ldots)`. :math:`S` is concave, so its
    increments form a sorted vector that majorizes every feasible one, and entropy is Schur concave.
    """
    pts = [0.0] + [min(1.0, float(c)) for c in caps] + [1.0] * max(1, n_max - len(caps))
    pts = np.maximum.accumulate(np.array(pts))
    hull = [0]
    for k in range(1, len(pts)):
        while len(hull) >= 2:
            i, j = hull[-2], hull[-1]
            if (pts[j] - pts[i]) * (k - i) <= (pts[k] - pts[i]) * (j - i):
                hull.pop()
            else:
                break
        hull.append(k)
    C = np.interp(np.arange(len(pts)), hull, pts[hull])
    S = np.minimum(np.minimum(C, atom * np.arange(len(pts))), 1.0)
    return _entropy(np.diff(S))


def _box_bound(P: np.ndarray, box: Box) -> float | None:
    qlo, _qhi, rlo, _rhi, tlo, thi = box
    T_lo = tlo * np.outer(qlo, rlo)
    if np.any(T_lo > P + 1e-15):
        return None
    scale = 1.0 - thi
    if scale <= 0.0:
        return 0.0
    Rbar = np.maximum(P - T_lo, 0.0)
    mu1 = mu1_vertex(Rbar)
    caps = [min(1.0, (mu1.value if mu1 is not None else float(Rbar.sum())) / scale)]
    caps += [min(1.0, mu_k_spectral_cap(Rbar, j) / scale) for j in range(2, min(P.shape))]
    cover = capped_coverage_entropy(caps, thi / scale)
    h = [_entropy(np.array([t, 1.0 - t])) + (1.0 - t) * cover for t in (tlo, thi)]
    return min(h)


def _split(box: Box, dim: int) -> list[Box]:
    qlo, qhi, rlo, rhi, tlo, thi = box
    na = len(qlo)
    out: list[Box] = []
    for side in (0, 1):
        q0, q1, r0, r1, t0, t1 = qlo.copy(), qhi.copy(), rlo.copy(), rhi.copy(), tlo, thi
        if dim < na:
            mid = 0.5 * (q0[dim] + q1[dim])
            if side == 0:
                q1[dim] = mid
            else:
                q0[dim] = mid
        elif dim < na + len(rlo):
            j = dim - na
            mid = 0.5 * (r0[j] + r1[j])
            if side == 0:
                r1[j] = mid
            else:
                r0[j] = mid
        else:
            mid = 0.5 * (t0 + t1)
            t0, t1 = (t0, mid) if side == 0 else (mid, t1)
        tq, tr = _tighten(q0, q1), _tighten(r0, r1)
        if tq is not None and tr is not None:
            out.append((tq[0], tq[1], tr[0], tr[1], t0, t1))
    return out


def heaviest_term_bound(
    P: np.ndarray,
    *,
    target: float | None = None,
    max_nodes: int = 200_000,
    deadline: float | None = None,
    weight_cap: Callable[[Box], float | None] | None = None,
) -> HeaviestTermBoundResult:
    """Certified lower bound on :math:`\\op{G}{X : Y}` for the joint ``P``; see the module docstring.

    Branches best first on the widest coordinate of the heaviest term's box and stops once every open box
    is at least ``target``, after ``max_nodes`` boxes, or at ``deadline`` (a :func:`time.monotonic` value).
    The caps are :func:`~sofic.generators.coverage_bound.mu1_vertex` and the spectral caps, as in the
    certified coverage bound. The work grows quickly with ``|X| + |Y|``; it is meant for small joints.

    ``weight_cap(box)``, if given, is an extra upper bound on the heavy term's weight over a box (``None``
    when no admissible term lies in it, which prunes the box);
    :func:`generator_heaviest_term_bound` uses it to restrict the heavy term to generator states.
    """
    P = np.asarray(P, dtype=float)
    P = P / P.sum()
    na, nb = P.shape
    mu1 = mu1_vertex(P)
    t_hi = min(1.0, mu1.value if mu1 is not None else 1.0)
    t_lo = 1.0 / (2 ** min(na, nb) - 1)

    def bound(box: Box) -> tuple[float | None, Box]:
        if weight_cap is not None:
            cap = weight_cap(box)
            if cap is None or cap < box[4]:
                return None, box
            box = (*box[:5], min(box[5], cap))
        return _box_bound(P, box), box

    first, root = bound((np.zeros(na), np.ones(na), np.zeros(nb), np.ones(nb), t_lo, t_hi))
    heap: list[tuple[float, int, Box]] = [] if first is None else [(first, 0, root)]
    order = itertools.count(1)
    nodes = 0
    goal = np.inf if target is None else target
    while heap and nodes < max_nodes and (deadline is None or time.monotonic() < deadline):
        lowest, _, box = heap[0]
        if lowest >= goal:
            break
        heapq.heappop(heap)
        nodes += 1
        qlo, qhi, rlo, rhi, tlo, thi = box
        widths = np.concatenate([qhi - qlo, rhi - rlo, [thi - tlo]])
        for split in _split(box, int(np.argmax(widths))):
            value, child = bound(split)
            if value is not None:
                heapq.heappush(heap, (max(lowest, value), next(order), child))
    value = heap[0][0] if heap else float("inf")
    return HeaviestTermBoundResult(value, nodes, len(heap), target is not None and value >= target)


def generator_heaviest_term_bound(
    problem: _GeneratorProblem,
    *,
    target: float | None = None,
    max_nodes: int = 200_000,
    deadline: float | None = None,
) -> HeaviestTermBoundResult:
    """Certified lower bound on the generative complexity :math:`C_g` by branching on the heaviest *state*.

    A generator's states give an exact decomposition of the causal-state joint
    :math:`p(s^+, s^-) = \\sum_w \\pi_w Q_w R_w^\\top`, whose heaviest term is its heaviest state; that state also
    satisfies the generator equations. So :func:`heaviest_term_bound` applies with an extra cap on the heavy
    weight: the one-explicit-state generator-cap LP over the box of the state's beliefs
    (:func:`~sofic.generators.generator_caps.generator_weight_cap`). Where that LP excludes the
    :math:`\\op{G}{S^+ : S^-}`-optimal heavy term the bound exceeds :math:`\\op{G}{S^+ : S^-}`.

    The root's weight floor :math:`1/(2^m - 1)`, :math:`m = \\min(|S^+|, |S^-|)`, is the cardinality bound on
    optimal decompositions, and a generator may have more states. It stays valid: a generator whose
    heaviest state weighs less has :math:`\\H{\\pi} \\ge \\log_2(2^m - 1) \\ge \\log_2 m \\ge \\min(C_\\mu^+,
    C_\\mu^-) \\ge C_g`, so it cannot lower :math:`C_g`.
    """
    from sofic.generators.generator_caps import _HeaviestStatesRelaxation

    relaxation = _HeaviestStatesRelaxation(problem, 1)
    P = problem.lifted.plus.T @ (problem.p_pair[:, None] * problem.lifted.minus)

    def weight_cap(box: Box) -> float | None:
        left = None if deadline is None else deadline - time.monotonic()
        return relaxation.bound((box[:4],), left)

    return heaviest_term_bound(P, target=target, max_nodes=max_nodes, deadline=deadline, weight_cap=weight_cap)


class _LiftedTermTest:
    """Necessary conditions for a decomposition term to be a state of a *lifted* generator (see below)."""

    def __init__(self, lifted: Any) -> None:
        self.pa = lifted.plus.argmax(axis=1)
        self.pb = lifted.minus.argmax(axis=1)
        self.p = lifted.p
        self.edges = lifted.edges
        self.na, self.nb = lifted.plus.shape[1], lifted.minus.shape[1]

    def _forward(self, q: np.ndarray, r: np.ndarray) -> np.ndarray:
        u = q[self.pa] * r[self.pb] / self.p
        inner = np.einsum("pxq,q->px", self.edges, u)
        out = np.zeros((self.na, inner.shape[1]))
        np.add.at(out, self.pa, (r[self.pb] / self.p)[:, None] * inner)
        return out

    def _reverse(self, q: np.ndarray, r: np.ndarray) -> np.ndarray:
        u = q[self.pa] * r[self.pb] / self.p
        inner = np.einsum("pxq,p->qx", self.edges, u)
        out = np.zeros((self.nb, inner.shape[1]))
        np.add.at(out, self.pb, (q[self.pa] / self.p)[:, None] * inner)
        return out

    def violation(self, q: np.ndarray, r: np.ndarray, *, support_tol: float = 1e-12) -> float:
        """Largest spread of ``L_B`` over ``supp q`` or of ``L_A`` over ``supp r`` (0 for a lifted state)."""
        worst = 0.0
        for values, support in ((self._forward(q, r), q > support_tol), (self._reverse(q, r), r > support_tol)):
            rows = values[support]
            if len(rows) > 1:
                worst = max(worst, float((rows.max(axis=0) - rows.min(axis=0)).max()))
        return worst

    def excludes(self, box: Box) -> bool:
        """Whether no term with beliefs in ``box`` passes: two surely supported states have disjoint intervals."""
        qlo, qhi, rlo, rhi = box[:4]
        for f, support in ((self._forward, qlo > 0.0), (self._reverse, rlo > 0.0)):
            lo, hi = f(qlo, rlo)[support], f(qhi, rhi)[support]
            if len(lo) > 1 and (lo.max(axis=0) > hi.min(axis=0) + 1e-12).any():
                return True
        return False


def lifted_heaviest_term_bound(
    problem: _GeneratorProblem,
    *,
    target: float | None = None,
    max_nodes: int = 200_000,
    deadline: float | None = None,
) -> HeaviestTermBoundResult:
    """Certified lower bound on the state entropy of every *lifted* generator, by branching on its heaviest state.

    A lifted generator draws :math:`W_0` and :math:`W_1` from one channel :math:`c(w \\mid s^+, s^-)`, independently
    given the bidirectional edge (see :mod:`sofic.generators.lifted_common_information`). Its states form an exact
    decomposition :math:`p = \\sum_w t_w q_w r_w^\\top`, so state :math:`e` has channel
    :math:`c_e(a, b) = t\\, q_a r_b / p(a, b)`. Coarse-graining :math:`W_1` (or :math:`W_0`) to the event that it
    equals :math:`e` cannot raise the lifting conditions' conditional mutual informations, so with :math:`E` the
    joint edge law and :math:`u(a, b) = q_a r_b / p(a, b)`, for every symbol :math:`x`

    .. math::

       L_B(a, x) = \\sum_b \\frac{r_b}{p(a, b)} \\sum_{a', b'} E[(a, b), x, (a', b')]\\, u(a', b')
       \\quad\\text{is constant over } a \\in \\operatorname{supp} q,

       L_A(b', x) = \\sum_{a'} \\frac{q_{a'}}{p(a', b')} \\sum_{a, b} E[(a, b), x, (a', b')]\\, u(a, b)
       \\quad\\text{is constant over } b' \\in \\operatorname{supp} r

    (the factor :math:`t` cancels). A general generator need not satisfy them: it may couple consecutive states.
    The coefficients are nonnegative, so a box's lower and upper corners bound :math:`L_A, L_B`; a box is pruned
    when two surely supported states get disjoint intervals, inside :func:`heaviest_term_bound`. The root's weight
    floor stays valid because the forward ε-machine is a lifted generator of entropy
    :math:`C_\\mu^+ \\le \\log_2 |S^+|`, so a lifted generator whose heaviest state weighs less than
    :math:`1/(2^m - 1)` cannot be minimal.
    """
    test = _LiftedTermTest(problem.lifted)
    P = problem.lifted.plus.T @ (problem.p_pair[:, None] * problem.lifted.minus)
    return heaviest_term_bound(
        P,
        target=target,
        max_nodes=max_nodes,
        deadline=deadline,
        weight_cap=lambda box: None if test.excludes(box) else box[5],
    )


__all__ = [
    "HeaviestTermBoundResult",
    "capped_coverage_entropy",
    "generator_heaviest_term_bound",
    "heaviest_term_bound",
    "lifted_heaviest_term_bound",
]
