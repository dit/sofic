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
from dataclasses import dataclass

import numpy as np

from sofic.generators.coverage_bound import mu1_vertex, mu_k_spectral_cap
from sofic.generators.generator_caps import _tighten

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
) -> HeaviestTermBoundResult:
    """Certified lower bound on :math:`\\op{G}{X : Y}` for the joint ``P``; see the module docstring.

    Branches best first on the widest coordinate of the heaviest term's box and stops once every open box
    is at least ``target``, after ``max_nodes`` boxes, or at ``deadline`` (a :func:`time.monotonic` value).
    The caps are :func:`~sofic.generators.coverage_bound.mu1_vertex` and the spectral caps, as in the
    certified coverage bound. The work grows quickly with ``|X| + |Y|``; it is meant for small joints.
    """
    P = np.asarray(P, dtype=float)
    P = P / P.sum()
    na, nb = P.shape
    mu1 = mu1_vertex(P)
    t_hi = min(1.0, mu1.value if mu1 is not None else 1.0)
    t_lo = 1.0 / (2 ** min(na, nb) - 1)
    root: Box = (np.zeros(na), np.ones(na), np.zeros(nb), np.ones(nb), t_lo, t_hi)
    first = _box_bound(P, root)
    heap: list[tuple[float, int, Box]] = [] if first is None else [(first, 0, root)]
    order = itertools.count(1)
    nodes = 0
    goal = np.inf if target is None else target
    while heap and nodes < max_nodes and (deadline is None or time.monotonic() < deadline):
        bound, _, box = heap[0]
        if bound >= goal:
            break
        heapq.heappop(heap)
        nodes += 1
        qlo, qhi, rlo, rhi, tlo, thi = box
        widths = np.concatenate([qhi - qlo, rhi - rlo, [thi - tlo]])
        for child in _split(box, int(np.argmax(widths))):
            value = _box_bound(P, child)
            if value is not None:
                heapq.heappush(heap, (max(bound, value), next(order), child))
    value = heap[0][0] if heap else float("inf")
    return HeaviestTermBoundResult(value, nodes, len(heap), target is not None and value >= target)


__all__ = ["HeaviestTermBoundResult", "capped_coverage_entropy", "heaviest_term_bound"]
