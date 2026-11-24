r"""Exact minimal generators for cryptic order at most 1, and lookahead generators above it.

Let :math:`P` be the joint distribution of the forward and reverse causal states :math:`(S^+, S^-)` of a process.
Every generator state separates them, so :math:`C_g \ge \op{G}{S^+ : S^-}` :cite:`LohrAy2009`. The persistent
crypticity :math:`p.\chi^+ = \H{S^+_1 \mid S^-_0, X_0}` equals :math:`\H{S^+_1 \mid X_{0:}}`, because the reverse
causal state shields the future from the past, and that is :math:`\chi^+ - \H{S^+_0 \mid X_0, S^+_1}`, Mahoney,
Ellison and Crutchfield's Theorem 1 at :math:`k = 1` :cite:`Mahoney2009`. It therefore vanishes exactly when the
cryptic order is at most 1. Forward and reverse cryptic orders are equal (when :math:`\chi^+ = 0` the past depends
on the future only through :math:`S^+`, so :math:`S^-` is a relabeling of :math:`S^+`; for orders of at least 1,
apply the vanishing of the two persistent crypticities together to the block processes), the cryptic-order
counterpart of the time symmetry of the Markov order :cite:`Ellison2011`. So cryptic order at most 1 is exactly
the class where both persistent crypticities vanish. There every separating variable of :math:`P` is
a generator state, :math:`C_g = \op{G}{S^+ : S^-}`, and :func:`theorem_generator` writes the generator down
without an LP.

A decomposition :math:`P = \sum_w A_w R_w^\top` (row masses :math:`A_w = \pi_w Q_w`, column factors :math:`R_w`)
can be brought down to at most

.. math::

   K(P) = \left\lfloor \frac{|\mathrm{supp}\, P| + \min(|S^+|, |S^-|)}{2} \right\rfloor

terms without raising its entropy (:func:`reduce_decomposition`, on the side with fewer rows or columns): merge two
terms with proportional row masses or equal column factors, or, when the column factors of the terms covering a
row are linearly dependent, move mass along the dependency until one of those cells empties, at the cheaper end
(the entropy is concave along the move). At the end every row's terms have independent column factors and each row
has at most one single-row term, which caps the count at :math:`K(P)` (:func:`support_bound_states`). Kumar, Li
and El Gamal's argument gives :math:`|X||Y|` :cite:`Kumar2014`. So :math:`K(P)` states suffice for a minimal
generator when the cryptic order is at most 1, and :func:`separating_decomposition` computes
:math:`\op{G}{S^+ : S^-}` per connected component of the support: exactly when the component has no
:math:`2 \times 2` rectangle (:func:`~sofic.generators.generator_bounds.one_wide_exact_common_information`),
otherwise by SCIP at :math:`K` terms (:func:`~sofic.generators.global_bound.separating_global_bound`, default
settings and the presolve/OBBT-off second opinion), polished into an exact decomposition.

Above cryptic order 1, :func:`lookahead_generator` builds the least-entropy generator whose state determines its
next :math:`L - 1` symbols. Such a generator is a generator of the sliding-block process
:math:`Y_t = X_{t:t+L}`, whose persistent crypticity is :math:`\H{S^+_L \mid X_{0:}}`; that vanishes exactly when
:math:`L` reaches the cryptic order, which Mahoney et al. define as the least such :math:`L` :cite:`Mahoney2011`,
and then the construction above applies to :math:`Y_L`, giving

.. math::

   C_g \le \H{X_{0:L-1}} + \sum_u \Pr(u)\, \op{G}{S^+_{L-1} : S^-_0 \mid X_{0:L-1} = u}.

The support bound, the reduction, :func:`theorem_generator`'s kernel and the lookahead construction are not from
the literature, and none of :cite:`Mahoney2009,Mahoney2011,Crutchfield2009,Ellison2011` states the equality of
the cryptic orders.
"""

from __future__ import annotations

import itertools
import math
import time
from dataclasses import dataclass, field
from typing import Any

import networkx as nx
import numpy as np

from sofic.generators.bidirectional_epsilon_machine import BidirectionalEpsilonMachine
from sofic.generators.coupled_generators import _decompose, _GeneratorProblem

__all__ = [
    "LookaheadGenerator",
    "SeparatingDecomposition",
    "lookahead_generator",
    "reduce_decomposition",
    "separating_decomposition",
    "support_bound_states",
    "theorem_generator",
]

_ZERO = 1e-10
_EXACT = 1e-12


def _entropy(p: Any) -> float:
    p = np.asarray(p, dtype=float).ravel()
    p = p[p > 1e-300]
    return float(-(p * np.log2(p)).sum())


def support_bound_states(P: np.ndarray, *, tol: float = 0.0) -> int:
    r""":math:`K(P) = \lfloor (|\mathrm{supp}\, P| + \min(\text{rows}, \text{columns})) / 2 \rfloor`, counting cells
    above ``tol`` and rows and columns of positive mass.

    Some least-entropy separating variable of ``P`` has at most this many values (see the module docstring), so
    when the cryptic order is at most 1 some minimal generator has at most :math:`K(P)` states.
    """
    S = np.asarray(P, dtype=float) > tol
    return int((S.sum() + min(int(S.any(axis=1).sum()), int(S.any(axis=0).sum()))) // 2)


def _nullvec(cols: list[np.ndarray], tol: float) -> np.ndarray | None:
    M = np.array(cols, dtype=float).T
    _, s, vh = np.linalg.svd(M)
    rank = int(np.sum(s > tol * max(1.0, s[0] if len(s) else 0.0)))
    return None if rank == M.shape[1] else vh[-1]


def _merge(A: list[np.ndarray], R: list[np.ndarray], tol: float) -> bool:
    for u in range(len(A)):
        pu = A[u].sum()
        for v in range(u + 1, len(A)):
            pv = A[v].sum()
            if np.abs(A[u] / pu - A[v] / pv).max() <= tol:
                R[u] = (pu * R[u] + pv * R[v]) / (pu + pv)
            elif np.abs(R[u] - R[v]).max() > tol:
                continue
            A[u] = A[u] + A[v]
            del A[v], R[v]
            return True
    return False


def _null_move(A: list[np.ndarray], R: list[np.ndarray], tol: float) -> bool:
    for x in range(len(A[0])):
        T = [w for w in range(len(A)) if A[w][x] > tol]
        if len(T) < 2:
            continue
        c = _nullvec([R[w] for w in T], tol)
        if c is None:
            continue
        c = np.where(np.abs(c) > 1e-14, c, 0.0)
        if not ((c > 0).any() and (c < 0).any()):
            continue
        hi = min((A[w][x] / -ci, i) for i, (w, ci) in enumerate(zip(T, c, strict=True)) if ci < 0)
        lo = max((-A[w][x] / ci, i) for i, (w, ci) in enumerate(zip(T, c, strict=True)) if ci > 0)
        pi = np.array([a.sum() for a in A])
        ends = []
        for t, i in (lo, hi):
            q = pi.copy()
            q[T] += t * c
            ends.append((_entropy(q), t, i))
        _, t, binding = min(ends)
        for w, ci in zip(T, c, strict=True):
            A[w][x] = max(A[w][x] + t * ci, 0.0)
        A[T[binding]][x] = 0.0
        return True
    return False


def reduce_decomposition(
    P: np.ndarray, A: Any, R: Any, *, tol: float = 1e-9
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    """Merges and null moves on a decomposition ``P = sum_w outer(A[w], R[w])`` until neither applies.

    ``A[w]`` are row masses (weight times row belief) and ``R[w]`` column factors (distributions). The moves run on
    the side with fewer rows or columns of positive mass, never raise the entropy of the weights, keep the
    decomposition exact up to rounding, and end with at most :func:`support_bound_states` terms. Returns
    ``(A, R, moves)`` with ``moves`` counting each kind; ``tol`` decides equal factors, positive cells and
    numerical rank.
    """
    P = np.asarray(P, dtype=float)
    S = P > 0
    flip = int(S.any(axis=0).sum()) < int(S.any(axis=1).sum())
    terms = [(np.asarray(a, float), np.asarray(r, float)) for a, r in zip(A, R, strict=True)]
    if flip:
        terms = [(a.sum() * r, a / a.sum()) for a, r in terms if a.sum() > tol]
    A_, R_ = [a for a, _ in terms], [r for _, r in terms]
    moves = {"merge": 0, "null": 0}
    while True:
        keep = [w for w in range(len(A_)) if A_[w].sum() > tol]
        A_, R_ = [A_[w] for w in keep], [R_[w] for w in keep]
        if _merge(A_, R_, tol):
            moves["merge"] += 1
        elif _null_move(A_, R_, tol):
            moves["null"] += 1
        else:
            break
    if flip:
        A_, R_ = [a.sum() * r for a, r in zip(A_, R_, strict=True)], [a / a.sum() for a in A_]
    return np.array(A_), np.array(R_), moves


@dataclass(frozen=True)
class SeparatingDecomposition:
    """A least-entropy separating variable of a joint, from :func:`separating_decomposition`.

    ``pi``, ``Q`` (row beliefs) and ``R`` (column beliefs) give ``P = sum_w pi_w outer(Q[w], R[w])`` to
    ``residual``. ``value`` is its entropy, an upper bound on :math:`\\op{G}{X : Y}`; ``lower`` is a lower bound
    (exact on rectangle-free components, SCIP's dual bound at :math:`K` terms otherwise, numerical), ``exact``
    says every component was solved exactly, and ``components`` records each component's method and bracket.
    """

    pi: np.ndarray
    Q: np.ndarray
    R: np.ndarray
    value: float
    lower: float
    exact: bool
    residual: float
    components: list[dict[str, Any]] = field(default_factory=list)


def _components(P: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    g = nx.Graph()
    g.add_edges_from((("r", int(a)), ("c", int(b))) for a, b in np.argwhere(P > _ZERO))
    out = []
    for nodes in nx.connected_components(g):
        rows = np.array(sorted(i for t, i in nodes if t == "r"))
        cols = np.array(sorted(i for t, i in nodes if t == "c"))
        out.append((rows, cols))
    return sorted(out, key=lambda rc: (int(rc[0][0]), int(rc[1][0])))


def _residual(P: np.ndarray, A: np.ndarray, R: np.ndarray) -> float:
    return float(np.abs(np.einsum("wa,wb->ab", A, R) - P).max())


def _project_exact(P: np.ndarray, A: np.ndarray, R: np.ndarray, threshold: float):
    """Least-squares projection onto exact decompositions with the zero pattern of ``(A, R)`` (entries at most
    ``threshold`` fixed at 0)."""
    from scipy.optimize import least_squares

    live = (A.sum(axis=1) > 0) & (R.sum(axis=1) > 0)
    A, R = A[live], R[live]
    pi = A.sum(axis=1)
    q0, r0 = A / pi[:, None], R / R.sum(axis=1, keepdims=True)
    k = len(pi)
    mQ, mR = q0 > threshold, r0 > threshold
    nq = int(mQ.sum())

    def unpack(x: np.ndarray):
        q, r = np.zeros(q0.shape), np.zeros(r0.shape)
        q[mQ], r[mR] = x[k : k + nq], x[k + nq :]
        return x[:k], q, r

    def fun(x: np.ndarray) -> np.ndarray:
        lam, q, r = unpack(x)
        return np.concatenate([(np.einsum("w,wa,wb->ab", lam, q, r) - P).ravel(), q.sum(1) - 1, r.sum(1) - 1])

    sol = least_squares(
        fun,
        np.concatenate([pi, q0[mQ], r0[mR]]),
        bounds=(0.0, 1.0),
        xtol=1e-15,
        ftol=1e-15,
        gtol=1e-15,
        max_nfev=5000,
    )
    lam, q, r = unpack(sol.x)
    keep = lam > 1e-14
    A2 = lam[keep, None] * q[keep]
    return A2, r[keep], _residual(P, A2, r[keep])


def _descend(P: np.ndarray, A: np.ndarray, R: np.ndarray):
    pi = A.sum(axis=1)
    lam, Qp, Rp, res = _decompose(P, pi, A / pi[:, None], R)
    if res <= 1e-10 and _entropy(lam) < _entropy(pi) - 1e-12:
        A2, R2, res2 = _project_exact(P, lam[:, None] * np.clip(Qp, 0, None), np.clip(Rp, 0, None), 1e-13)
        if res2 <= _EXACT and _entropy(A2.sum(axis=1)) < _entropy(pi):
            return A2, R2, res2
    return A, R, _residual(P, A, R)


def _polish(
    P: np.ndarray,
    candidates: list[tuple[np.ndarray, np.ndarray]],
    k_max: int,
    starts: int,
    rng: np.random.Generator,
    deadline: float | None = None,
):
    """The least-entropy exact decomposition from approximate ``candidates`` (reduced at their tolerance, projected,
    descended) and multistart descents with 2 terms up to the most terms of a candidate (``k_max`` without
    candidates); ``None`` when none is exact. Multistarts stop at ``deadline`` once some decomposition is exact."""
    exact = []
    for A, R in candidates:
        A0, R0, _ = reduce_decomposition(P, A, R, tol=1e-6)
        for Ai, Ri in ((A0, R0), (A, R)):
            for threshold in (1e-7, 1e-5, 1e-3):
                A1, R1, res1 = _project_exact(P, np.asarray(Ai, float), np.asarray(Ri, float), threshold)
                if res1 <= _EXACT:
                    exact.append(_descend(P, A1, R1))
                    break
    na, nb = P.shape
    if candidates:
        k_max = max(len(A) for A, _ in candidates)
    for k in range(2, max(2, k_max) + 1):
        for _ in range(starts):
            if exact and deadline is not None and time.monotonic() > deadline:
                break
            lam, Qd, Rd, res = _decompose(
                P, rng.dirichlet(np.ones(k)), rng.dirichlet(np.ones(na), size=k), rng.dirichlet(np.ones(nb), size=k)
            )
            if res <= 1e-10:
                A1, R1, res1 = _project_exact(P, lam[:, None] * np.clip(Qd, 0, None), np.clip(Rd, 0, None), 1e-13)
                if res1 <= _EXACT:
                    exact.append((A1, R1, res1))
    return min(exact, key=lambda c: _entropy(c[0].sum(axis=1))) if exact else None


def _scip_component(P: np.ndarray, k: int, time_limit: float) -> tuple[list, float, dict[str, Any]]:
    from sofic.generators.global_bound import separating_global_bound

    runs = {
        "default": separating_global_bound(P, k, time_limit=time_limit),
        "second": separating_global_bound(P, k, time_limit=time_limit, presolve=False, obbt=False),
    }
    if runs["default"].status == "infeasible" and runs["second"].status != "infeasible":
        runs["default"] = runs["second"]
    candidates = []
    for res in runs.values():
        if res.factors is not None:
            p, Q, R = res.factors
            keep = p > 1e-9
            R = np.where(R[keep] > 1e-9, R[keep], 0.0)
            candidates.append(
                (p[keep, None] * np.where(Q[keep] > 1e-9, Q[keep], 0.0), R / R.sum(axis=1, keepdims=True))
            )
    lowers = [res.lower for res in runs.values() if res.status != "infeasible"]
    info = {name: {"status": res.status, "primal": res.primal, "lower": res.lower} for name, res in runs.items()}
    return candidates, min(lowers, default=0.0), info


def separating_decomposition(
    P: np.ndarray,
    *,
    time_limit: float = 60.0,
    starts: int = 6,
    rng: np.random.Generator | None = None,
) -> SeparatingDecomposition:
    r"""A least-entropy separating variable of the joint ``P``, solved per connected component of its support.

    A component in one row or column is a single term. A component without :math:`2 \times 2` rectangles is solved
    exactly (:func:`~sofic.generators.generator_bounds.one_wide_exact_common_information`). Otherwise SCIP
    minimizes the entropy over decompositions with :math:`K` terms (:func:`support_bound_states` of the component,
    which covers an optimum), with     default settings and with presolve and OBBT off
    (:func:`~sofic.generators.global_bound.separating_global_bound`, ``time_limit`` seconds each). SCIP's
    solutions are feasible only to its tolerance, so each is reduced, projected onto exact decompositions and
    descended, alongside ``starts`` multistart descents per term count (for up to ``time_limit`` seconds more);
    the best exact one is kept and the smaller
    of the two dual bounds is the component's lower bound (an "infeasible" counts as no bound, since a
    decomposition with :math:`K` terms always exists). Without PySCIPOpt the multistart descents alone give
    the upper bound and the lower bound is 0. Components combine as
    :math:`\H{C} + \sum_c \Pr(c)\, G_c` for the component label :math:`C`, a common function of both sides.
    """
    from sofic.generators.generator_bounds import one_wide_exact_common_information

    rng = rng if rng is not None else np.random.default_rng(0)
    P = np.asarray(P, dtype=float)
    P = P / P.sum()
    A_all, R_all, parts, masses = [], [], [], []
    lower, exact = 0.0, True
    for rows, cols in _components(P):
        sub = P[np.ix_(rows, cols)]
        mass = float(sub.sum())
        sub = sub / mass
        part: dict[str, Any] = {"rows": rows.tolist(), "cols": cols.tolist(), "K": support_bound_states(sub, tol=_ZERO)}
        if min(sub.shape) == 1:
            A, R = sub.sum(axis=1)[None, :], sub.sum(axis=0)[None, :]
            part.update(method="trivial", lower=0.0)
        else:
            try:
                ow = one_wide_exact_common_information(sub, atol=1e-13)
            except ValueError:
                ow = None
            if ow is not None:
                A, R = _blocks_to_terms(sub, ow.blocks)
                part.update(method="one-wide", lower=ow.value)
            else:
                exact = False
                try:
                    candidates, c_lower, info = _scip_component(sub, part["K"], time_limit)
                    part.update(method="scip", scip=info)
                except ImportError:
                    candidates, c_lower = [], 0.0
                    part.update(method="multistart")
                best = _polish(sub, candidates, part["K"], starts, rng, time.monotonic() + time_limit)
                if best is None:
                    raise RuntimeError(f"no exact decomposition found for a {sub.shape} component")
                A, R, _ = best
                part["lower"] = min(c_lower, _entropy(A.sum(axis=1)))
        part["value"] = _entropy(A.sum(axis=1))
        lower += mass * part["lower"]
        masses.append(mass)
        for a, r in zip(A, R, strict=True):
            a_full, r_full = np.zeros(P.shape[0]), np.zeros(P.shape[1])
            a_full[rows], r_full[cols] = mass * np.asarray(a), np.asarray(r)
            A_all.append(a_full)
            R_all.append(r_full)
        parts.append(part)
    A_full, R_full = np.array(A_all), np.array(R_all)
    pi = A_full.sum(axis=1)
    value = _entropy(pi)
    return SeparatingDecomposition(
        pi=pi,
        Q=A_full / pi[:, None],
        R=R_full,
        value=value,
        lower=min(value, lower + _entropy(masses)),
        exact=exact,
        residual=_residual(P, A_full, R_full),
        components=parts,
    )


def _blocks_to_terms(P: np.ndarray, blocks: Any) -> tuple[np.ndarray, np.ndarray]:
    A, R = [], []
    for block in blocks:
        mass = sum(P[c] for c in block)
        a, r = np.zeros(P.shape[0]), np.zeros(P.shape[1])
        rows = {c[0] for c in block}
        if len(rows) == 1:
            (row,) = rows
            a[row] = mass
            for _, b in block:
                r[b] = P[row, b] / mass
        else:
            (col,) = {c[1] for c in block}
            for x, _ in block:
                a[x] = P[x, col]
            r[col] = 1.0
        A.append(a)
        R.append(r)
    return np.array(A), np.array(R)


def _as_problem(bidir: BidirectionalEpsilonMachine | _GeneratorProblem) -> _GeneratorProblem:
    return bidir if isinstance(bidir, _GeneratorProblem) else _GeneratorProblem(bidir)


def theorem_generator(
    bidir: BidirectionalEpsilonMachine | _GeneratorProblem,
    pi: np.ndarray,
    Q: np.ndarray,
    R: np.ndarray,
    *,
    tol: float = 1e-9,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    r"""The generator of a separating variable ``(pi, Q, R)`` of the causal-state joint, when :math:`p.\chi^+ = 0`.

    With the channel :math:`c(w \mid a, b) = \pi_w Q_w(a) R_w(b) / P(a, b)` the state at time 1 is drawn from
    :math:`c(\cdot \mid S^+_1, S^-_1)`, and the kernel is

    .. math::

       K(x, v \mid w) = \sum_{b, b'} R_w(b) \Pr(X_0 = x, S^-_1 = b' \mid S^-_0 = b)\,
       c(v \mid \phi(b, x), b'),

    where :math:`\phi(b, x) = S^+_1` is a function of :math:`(S^-_0, X_0)` because :math:`p.\chi^+ = 0`. Returns
    ``(pi, Q, R, K)`` with ``K[w, x, v]``, symbols in the problem's order; it satisfies the coupled generator
    equations, and its word process is the source's. Raises ``ValueError`` when :math:`p.\chi^+ > ` ``tol``.
    """
    problem = _as_problem(bidir)
    lif = problem.lifted
    J = np.einsum("pb,pxq,qa->bxa", lif.minus, lif.edges, lif.plus)
    pchi = _entropy(J) - _entropy(J.sum(axis=2))
    if pchi > tol:
        raise ValueError(f"theorem_generator needs p.chi+ = 0 (cryptic order at most 1); p.chi+ = {pchi:.3g}")
    pi, Q, R = (np.asarray(t, dtype=float) for t in (pi, Q, R))
    P = np.einsum("p,pa,pb->ab", lif.p, lif.plus, lif.minus)
    pa, pb = lif.plus.argmax(axis=1), lif.minus.argmax(axis=1)
    c = (pi[None, :] * Q[:, pa].T * R[:, pb].T) / P[pa, pb][:, None]
    T = np.einsum("pb,pxq->bxq", lif.minus, lif.edges)
    K = np.einsum("wb,bxq,qv->wxv", R / problem.p_minus[None, :], T, c)
    return pi, Q, R, K


@dataclass(frozen=True)
class LookaheadGenerator:
    """The least-entropy generator whose state determines its next ``L - 1`` symbols (:func:`lookahead_generator`).

    ``model`` is a Mealy HMM for the source with ``states`` states, and ``kernel[w, x, v]`` its transition tensor
    over ``symbols``; ``value`` its state entropy, an upper bound on :math:`C_g`; ``lower`` bounds ``value``'s
    optimum over this class (exact when ``exact``); ``word_error`` the exact word-process deviation from the
    source.
    """

    L: int
    model: Any
    value: float
    lower: float
    exact: bool
    states: int
    word_error: float
    kernel: np.ndarray
    symbols: tuple


def _block_machine(machine: Any, L: int) -> Any:
    """The ε-machine of :math:`Y_t = X_{t:t+L}` (symbols are ``L``-tuples): states (last ``L - 1`` symbols,
    :math:`S^+`)."""
    from sofic.examples.epsilon_machines import from_symbol_matrices
    from sofic.generators.epsilon_machine import EpsilonMachine
    from sofic.generators.matrices import emission_tensors

    pi, T = emission_tensors(machine)
    pi = np.asarray(pi, dtype=float)
    syms = list(T)
    T = {x: np.asarray(m, dtype=float) for x, m in T.items()}
    states = []
    for u in itertools.product(syms, repeat=L - 1):
        v = pi.copy()
        for x in u:
            v = v @ T[x]
        states += [(u, a) for a in range(len(pi)) if v[a] > 1e-15]
    index = {s: i for i, s in enumerate(states)}
    mats: dict[tuple, np.ndarray] = {}
    for (u, a), i in index.items():
        for x in syms:
            for a2 in np.flatnonzero(T[x][a] > 0):
                y = (*u, x)
                j = index[(y[1:], int(a2))]
                mats.setdefault(y, np.zeros((len(states), len(states))))[i, j] += T[x][a, a2]
    labels = [f"{''.join(map(str, u))}|{a}" for u, a in states]
    m = from_symbol_matrices(labels, sorted(mats, key=repr), mats)
    return m if isinstance(m, EpsilonMachine) else EpsilonMachine.from_hmm(m)


def lookahead_generator(
    machine: Any,
    L: int | None = None,
    *,
    time_limit: float = 60.0,
    rng: np.random.Generator | None = None,
) -> LookaheadGenerator:
    r"""The least-entropy generator of ``machine``'s process whose state determines its next ``L - 1`` symbols.

    ``L`` defaults to the cryptic order (at least 1), the least ``L`` for which the sliding-block process
    :math:`Y_L` has :math:`p.\chi = 0`; ``ValueError`` when the cryptic order is infinite and no ``L`` is given, or
    when :math:`Y_L` still has :math:`p.\chi^+ > 0`. Builds :math:`Y_L`'s bidirectional machine, a least-entropy
    separating variable of its causal states (:func:`separating_decomposition`), reduces it, writes the generator
    of :func:`theorem_generator`, keeps the first symbol of each emission, and checks the word process against
    ``machine`` exactly. The value is :math:`\H{X_{0:L-1}} + \sum_u \Pr(u) \op{G}{S^+_{L-1} : S^-_0 \mid u}`, an
    upper bound on :math:`C_g`: on the :math:`(2, \infty)` process it is :math:`3/2 = C_g`, on the butterfly
    :math:`3.1219`, above :math:`C_g = C_\mu = \log_2 5`.
    """
    from sofic.generators.generative_search import resolve_bidirectional
    from sofic.generators.mealy import MealyHMM
    from sofic.generators.minimal_generative_model import _equivalence_error
    from sofic.graph import ATTR_EMISSION, ATTR_PROB

    if L is None:
        kappa = machine.cryptic_order()
        if kappa == math.inf:
            raise ValueError("the cryptic order is infinite; give L")
        L = max(int(kappa), 1)
    y_machine = _block_machine(machine, L)
    problem = _GeneratorProblem(resolve_bidirectional(y_machine)[0])
    P = np.einsum("p,pa,pb->ab", problem.lifted.p, problem.lifted.plus, problem.lifted.minus)
    dec = separating_decomposition(P, time_limit=time_limit, rng=rng)
    A, R, _ = reduce_decomposition(P, dec.pi[:, None] * dec.Q, dec.R)
    pi = A.sum(axis=1)
    pi, _, _, K = theorem_generator(problem, pi, A / pi[:, None], R)
    keep = np.flatnonzero(pi > 1e-14)
    firsts = sorted({y[0] for y in problem.symbols}, key=repr)
    KX = np.zeros((len(keep), len(firsts), len(keep)))
    for yi, y in enumerate(problem.symbols):
        KX[:, firsts.index(y[0]), :] += K[np.ix_(keep, [yi], keep)][:, 0, :]
    KX /= KX.sum(axis=(1, 2), keepdims=True)
    names = [f"G{i}" for i in range(len(keep))]
    weights = pi[keep] / pi[keep].sum()
    model = MealyHMM(
        initial_distribution=dict(zip(names, weights.tolist(), strict=True)), observation_alphabet=frozenset(firsts)
    )
    for s in names:
        model.graph.add_state(s)
    for w, xi, v in zip(*np.nonzero(KX > 1e-14), strict=True):
        model.graph.add_transition(names[w], names[v], **{ATTR_PROB: float(KX[w, xi, v]), ATTR_EMISSION: firsts[xi]})
    return LookaheadGenerator(
        L=L,
        model=model,
        value=_entropy(weights),
        lower=dec.lower,
        exact=dec.exact,
        states=len(keep),
        word_error=float(_equivalence_error(model, machine)),
        kernel=KX,
        symbols=tuple(firsts),
    )


def _block_persistent_crypticities(problem: _GeneratorProblem, L_max: int = 4, max_size: int = 200_000):
    r"""``[(L, p.chi+, p.chi-)]`` of the block processes :math:`Y_L` for ``L = 1 .. L_max``, from the tensor
    :math:`\Pr(S^-_0, X_{0:L}, S^+_L)`: :math:`p.\chi^+(Y_L) = \H{S^+_L \mid X_{0:}}` and
    :math:`p.\chi^-(Y_L) = \H{S^-_0 \mid X_{:L}}`, both zero exactly from the cryptic order on. Stops once both
    vanish or the tensor would exceed ``max_size`` entries."""
    lif = problem.lifted
    pp = lif.p
    T = lif.edges / np.where(pp > 0, pp, 1.0)[:, None, None]
    cur = np.einsum("pb,pxq->bxq", lif.minus, lif.edges)
    out = []
    for L in range(1, L_max + 1):
        if L > 1:
            if cur.size * T.shape[1] > max_size:
                break
            cur = np.einsum("bwq,qxr->bwxr", cur, T).reshape(cur.shape[0], -1, T.shape[-1])
        J = np.einsum("bwr,ra->bwa", cur, lif.plus)
        h = _entropy(J)
        plus, minus = max(0.0, h - _entropy(J.sum(axis=2))), max(0.0, h - _entropy(J.sum(axis=0)))
        out.append((L, plus, minus))
        if plus < 1e-10 and minus < 1e-10:
            break
    return out
