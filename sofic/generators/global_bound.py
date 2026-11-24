r"""Restricted global lower bound: the least state entropy of generators with at most ``k`` states.

Every generator with states :math:`w`, weights :math:`\pi_w`, causal-state beliefs :math:`Q_w` (over
:math:`S^+`) and :math:`R_w` (over :math:`S^-`) and kernel :math:`K(x, v \mid w)` satisfies the one-step coupled
generator equations of :mod:`~sofic.generators.coupled_generators`. In terms of the flows
:math:`F(w, x, v) = \pi_w K(x, v \mid w)` they read

.. math::

   \sum_{x, v} F(w, x, v) = \pi_w = \sum_{u, x} F(u, x, w), \qquad
   \sum_w \pi_w Q_w R_w^\top = P, \qquad
   \sum_{w, v} F(w, x, v)\, Q_w(a) R_v(b) = \Pr(S^+_0 = a, X_0 = x, S^-_1 = b),

together with forward stationarity of :math:`\pi_w Q_w` and reverse consistency of :math:`\pi_w R_w`. When the
problem carries the bidirectional edge chain, two-step path flows
:math:`V(w, x, v, y, u) = \pi_w K(x, v \mid w) K(y, u \mid v)` add :math:`V \pi_v = F(w, x, v) F(v, y, u)`, their
marginals, and the two-step edge law :math:`\sum V\, Q_w(a) R_u(b) = \Pr(S^+_0 = a, X_0 X_1 = x y, S^-_2 = b)`.
All of these are necessary for a generator, not sufficient. :func:`generator_global_bound` minimizes
:math:`\H{\pi}` over them with SCIP's spatial branch and bound :cite:`Achterberg2009,Vigerske2018` through
PySCIPOpt :cite:`Maher2016`. Its dual bound therefore bounds the state entropy of every generator with at most
``k`` states from below; it says nothing about larger generators, so it is a lower bound on the generative
complexity :math:`C_g` only once a bound on the number of states of a minimal generator is known. No general bound
is known, and :math:`\min(|S^+|, |S^-|) + 1` is not one: a Markov chain on six symbols with
:math:`|S^+| = |S^-| = 6` needs 8 states (see
:func:`~sofic.generators.generator_bounds.one_wide_exact_common_information`). When both persistent crypticities
vanish (cryptic order at most 1), every separating variable is a generator state, and some optimal one has at most
:math:`K(P) = \lfloor (|\mathrm{supp}\, P| + \min(|S^+|, |S^-|)) / 2 \rfloor` values
(:func:`~sofic.generators.separating_generators.support_bound_states`; Kumar, Li and El Gamal's argument
:cite:`Kumar2014`, restricted to the support, gives the support size), so ``k = K(P)`` suffices. The two-step flows,
the support bound and this restricted use are not from the literature.

The bound is numerical, not a certificate: the data are floats, the equations with data on the right-hand side
are relaxed to ``|lhs - rhs| <= data_tol`` (which only enlarges the feasible set), and SCIP works to its own
feasibility tolerances. PySCIPOpt is an optional dependency (``pip install sofic[global]``).

:func:`separating_global_bound` is the static counterpart: the least :math:`\H{W}` over separating variables of a
joint with at most ``k`` values, which at ``k`` equal to
:func:`~sofic.generators.separating_generators.support_bound_states` is :math:`\op{G}{X : Y}` itself.

SCIP settings matter on both models. Its presolve can declare a feasible model infeasible when every feasible
point ties the sorted weights (Kumar, Li and El Gamal's Example 2 :cite:`Kumar2014` at ``k = 2``, whose only
decomposition is :math:`W = X` at weights :math:`(1/2, 1/2)`), and its optimization-based bound tightening can cut
off the optimum (a 3 x 3 joint with 5 support cells, where ``k = 3`` returns "optimal" 1.447803 while a feasible
3-block partition has 1.343763; see ``tests/test_separating_generators.py``). ``presolve=False, obbt=False`` is the
second opinion for a decisive "infeasible" or "optimal".
"""

from __future__ import annotations

import math
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from sofic.generators.bidirectional_epsilon_machine import BidirectionalEpsilonMachine
from sofic.generators.coupled_generators import _GeneratorProblem

__all__ = ["GlobalBoundResult", "SeparatingBoundResult", "generator_global_bound", "separating_global_bound"]


@dataclass(frozen=True)
class GlobalBoundResult:
    """Outcome of :func:`generator_global_bound`.

    ``lower`` bounds the state entropy of every generator with at most ``k`` states (and only those): the SCIP
    dual bound minus ``slack``, or ``objective_limit - slack`` when SCIP proves that no point of the equations has
    entropy below the limit (``status == "infeasible"``); it is ``inf`` when the equations have no solution at all
    (no generator with at most ``k`` states). ``primal`` and ``seed`` (``(pi, Q, R, K)``, with
    ``K[w, x, v] = F(w, x, v) / pi_w``) describe SCIP's best solution, which solves the necessary equations only
    and must be verified (or polished with :func:`~sofic.generators.hmm_search.hmm_generator_search`, whose
    ``initial`` argument takes its symbol matrices) before it counts as a generator. ``two_step`` records whether
    the two-step flows were used.
    """

    k: int
    status: str
    lower: float
    dual_bound: float
    slack: float
    primal: float | None
    seed: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None
    nodes: int
    seconds: float
    two_step: bool


@dataclass(frozen=True)
class SeparatingBoundResult:
    """Outcome of :func:`separating_global_bound`.

    ``lower`` bounds :math:`\\H{W}` over separating variables with at most ``k`` values: the SCIP dual bound minus
    ``slack``, ``objective_limit - slack`` when SCIP proves nothing lies below the limit, and ``inf`` when no
    decomposition with ``k`` terms exists. ``factors`` is SCIP's best solution ``(pi, Q, R)``, feasible only to
    SCIP's tolerances; :func:`~sofic.generators.separating_generators.separating_decomposition` polishes it into
    an exact decomposition.
    """

    k: int
    status: str
    lower: float
    dual_bound: float
    slack: float
    primal: float | None
    factors: tuple[np.ndarray, np.ndarray, np.ndarray] | None
    nodes: int
    seconds: float


def _pyscipopt() -> Any:
    try:
        import pyscipopt
    except ImportError as exc:
        raise ImportError(
            "generator_global_bound needs PySCIPOpt; install it with `pip install sofic[global]`"
        ) from exc
    return pyscipopt


def _configure(m: Any, scip: Any, presolve: bool, obbt: bool, params: Mapping[str, Any] | None) -> None:
    if not presolve:
        m.setPresolve(scip.SCIP_PARAMSETTING.OFF)
    if not obbt:
        m.setParam("propagating/obbt/freq", -1)
    for key, value in (params or {}).items():
        m.setParam(key, value)


def _bounds(m: Any, status: str, objective_limit: float | None, slack: float) -> tuple[float, float]:
    dual = float(m.getDualbound())
    dual = math.inf if dual >= m.infinity() else -math.inf if dual <= -m.infinity() else dual
    if status == "infeasible":
        lower = math.inf if objective_limit is None else float(objective_limit) - slack
    else:
        lower = (dual if objective_limit is None else min(dual, float(objective_limit))) - slack
    return float(lower), dual


def separating_global_bound(
    P: np.ndarray,
    k: int,
    *,
    objective_limit: float | None = None,
    time_limit: float | None = None,
    pi_min: float = 1e-7,
    presolve: bool = True,
    obbt: bool = True,
    params: Mapping[str, Any] | None = None,
) -> SeparatingBoundResult:
    r"""Least :math:`\H{W}` over separating variables of the joint ``P`` with at most ``k`` values, by SCIP.

    A separating variable is a decomposition :math:`P = \sum_w \pi_w Q_w R_w^\top` (rows :math:`X`, columns
    :math:`Y`); its entropy is minimized with SCIP's spatial branch and bound :cite:`Achterberg2009,Vigerske2018`,
    the weights sorted and each entropy term evaluated at :math:`\max(\pi_w, \text{pi\_min})` (``slack`` corrects
    for that), as in :func:`generator_global_bound`. At ``k`` equal to
    :func:`~sofic.generators.separating_generators.support_bound_states` some optimal separating variable is
    covered, so ``lower`` bounds the exact common information :math:`\op{G}{X : Y}` itself (Kumar, Li and El
    Gamal's cardinality bound :cite:`Kumar2014` is :math:`|X||Y|`). ``presolve``, ``obbt`` and ``params`` set SCIP
    as described in the module docstring. Numerical, not a certificate. Requires PySCIPOpt.
    """
    scip = _pyscipopt()
    start = time.monotonic()
    P = np.asarray(P, dtype=float)
    P = P / P.sum()
    na, nb = P.shape
    pr, pc = P.sum(axis=1), P.sum(axis=0)
    W, A, B = range(k), range(na), range(nb)
    quicksum = scip.quicksum
    m = scip.Model()
    m.hideOutput()
    _configure(m, scip, presolve, obbt, params)
    p = {w: m.addVar(lb=0, ub=1) for w in W}
    s = {w: m.addVar(lb=pi_min, ub=1) for w in W}
    e = {w: m.addVar(lb=0, ub=1 / (math.e * math.log(2)) + 1e-9) for w in W}
    Q = {(w, a): m.addVar(lb=0, ub=1) for w in W for a in A}
    R = {(w, b): m.addVar(lb=0, ub=1) for w in W for b in B}
    Y = {(w, a): m.addVar(lb=0, ub=float(pr[a])) for w in W for a in A}
    Z = {(w, b): m.addVar(lb=0, ub=float(pc[b])) for w in W for b in B}
    m.addCons(quicksum(p.values()) == 1)
    for w in W:
        m.addCons(s[w] >= p[w])
        m.addCons(s[w] <= p[w] + pi_min)
        m.addCons(e[w] >= -s[w] * scip.log(s[w]) / math.log(2))
        m.addCons(quicksum(Q[w, a] for a in A) == 1)
        m.addCons(quicksum(R[w, b] for b in B) == 1)
        m.addCons(quicksum(Y[w, a] for a in A) == p[w])
        m.addCons(quicksum(Z[w, b] for b in B) == p[w])
        for a in A:
            m.addCons(Y[w, a] == p[w] * Q[w, a])
        for b in B:
            m.addCons(Z[w, b] == p[w] * R[w, b])
        if w + 1 < k:
            m.addCons(p[w] >= p[w + 1])
    for a in A:
        m.addCons(quicksum(Y[w, a] for w in W) == float(pr[a]))
    for b in B:
        m.addCons(quicksum(Z[w, b] for w in W) == float(pc[b]))
    for a in A:
        for b in B:
            m.addCons(quicksum(Y[w, a] * R[w, b] for w in W) == float(P[a, b]))
            m.addCons(quicksum(Q[w, a] * Z[w, b] for w in W) == float(P[a, b]))
    m.setObjective(quicksum(e.values()), "minimize")
    if objective_limit is not None:
        m.setObjlimit(float(objective_limit))
    if time_limit is not None:
        m.setParam("limits/time", max(1.0, float(time_limit)))
    m.optimize()
    status = str(m.getStatus())
    slack = k * pi_min * math.log2(1 / pi_min)
    lower, dual = _bounds(m, status, objective_limit, slack)
    primal, factors = None, None
    if m.getNSols() > 0:
        primal = float(m.getPrimalbound())
        sol = m.getBestSol()
        factors = (
            np.clip([m.getSolVal(sol, p[w]) for w in W], 0, None),
            np.clip([[m.getSolVal(sol, Q[w, a]) for a in A] for w in W], 0, None),
            np.clip([[m.getSolVal(sol, R[w, b]) for b in B] for w in W], 0, None),
        )
    return SeparatingBoundResult(
        k=k,
        status=status,
        lower=lower,
        dual_bound=dual,
        slack=slack,
        primal=primal,
        factors=factors,
        nodes=int(m.getNNodes()),
        seconds=time.monotonic() - start,
    )


def _two_step_law(problem: _GeneratorProblem) -> np.ndarray | None:
    """``Pe2[x, y, a, b] = Pr(S+_0 = a, X_0 X_1 = x y, S-_2 = b)`` from the bidirectional edge chain, if known."""
    lifted = problem.lifted
    edges = getattr(lifted, "edges", None)
    if edges is None:
        return None
    step = edges / np.maximum(lifted.p, 1e-300)[:, None, None]
    ns = edges.shape[1]
    out = np.empty((ns, ns, lifted.plus.shape[1], lifted.minus.shape[1]))
    for x in range(ns):
        for y in range(ns):
            out[x, y] = lifted.plus.T @ edges[:, x, :] @ step[:, y, :] @ lifted.minus
    return out


def _model_size(problem: _GeneratorProblem, k: int, *, two_step: bool = True) -> int:
    """Number of SCIP variables :func:`generator_global_bound` creates for ``problem`` and ``k`` states."""
    na, nb, ns = problem.n_plus, problem.n_minus, problem.n_sym
    n = k * (3 + 2 * (na + nb)) + k * ns * k * (1 + na)
    if two_step and getattr(problem.lifted, "edges", None) is not None:
        n += k * ns * k * ns * k * (1 + na)
    return n


def generator_global_bound(
    bidir: BidirectionalEpsilonMachine | _GeneratorProblem,
    k: int,
    *,
    objective_limit: float | None = None,
    two_step: bool = True,
    caps: Sequence[float] = (),
    time_limit: float | None = None,
    data_tol: float = 1e-7,
    pi_min: float = 1e-7,
    presolve: bool = True,
    obbt: bool = True,
    params: Mapping[str, Any] | None = None,
) -> GlobalBoundResult:
    r"""Lower bound on the state entropy of generators with at most ``k`` states, by SCIP global optimization.

    Builds the coupled generator equations (see the module docstring) for ``k`` states, with the two-step flows
    when ``two_step`` is set and ``bidir`` carries the bidirectional edge chain, and minimizes the state entropy
    with SCIP. Generators with fewer than ``k`` states are covered (their missing states get weight 0). The
    weights are sorted, so the optional ``caps`` (valid caps on the total weight of the ``j`` heaviest states of
    any generator, e.g. those of :func:`~sofic.generators.generator_caps.generator_coverage_bound`) become
    linear cuts. Each entropy term is evaluated at :math:`\max(\pi_w, \text{pi\_min})`, which overestimates it by
    at most :math:`\text{pi\_min} \log_2(1/\text{pi\_min})`; ``slack`` (``k`` times that) is subtracted.

    ``objective_limit`` (e.g. the entropy of a verified generator) lets SCIP prune everything above it; when it
    then reports ``"infeasible"``, no generator with at most ``k`` states has entropy below
    ``objective_limit - slack``. ``time_limit`` (seconds) stops SCIP early; the dual bound reached is still valid.
    The result is numerical, not a certificate, and says nothing about generators with more than ``k`` states;
    see :class:`GlobalBoundResult`. ``presolve=False, obbt=False`` is the second opinion for a decisive
    "infeasible" or "optimal" (see the module docstring); ``params`` sets further SCIP parameters, e.g.
    ``{"numerics/feastol": 1e-9}``.

    Requires PySCIPOpt (``pip install sofic[global]``).
    """
    scip = _pyscipopt()
    start = time.monotonic()
    problem = bidir if isinstance(bidir, _GeneratorProblem) else _GeneratorProblem(bidir)
    lifted = problem.lifted
    P = lifted.plus.T @ (problem.p_pair[:, None] * lifted.minus)
    P = P / P.sum()
    Pe, fwd, rev = np.asarray(problem.Pe, float), np.asarray(problem.F) > 0, np.asarray(problem.G) > 0
    pp, pm = np.asarray(problem.p_plus, float), np.asarray(problem.p_minus, float)
    na, nb, ns = P.shape[0], P.shape[1], Pe.shape[1]
    Pe2 = _two_step_law(problem) if two_step else None
    W, X, A, B = range(k), range(ns), range(na), range(nb)
    quicksum = scip.quicksum

    m = scip.Model()
    m.hideOutput()
    _configure(m, scip, presolve, obbt, params)

    def data_eq(lhs: Any, rhs: float) -> None:
        m.addCons(lhs <= rhs + data_tol)
        m.addCons(lhs >= rhs - data_tol)

    pi = {w: m.addVar(lb=0, ub=1) for w in W}
    s = {w: m.addVar(lb=pi_min, ub=1) for w in W}  # max(pi_w, pi_min), the argument of the entropy term
    e = {w: m.addVar(lb=0, ub=1 / (math.e * math.log(2)) + 1e-9) for w in W}
    Q = {(w, a): m.addVar(lb=0, ub=1) for w in W for a in A}
    R = {(w, b): m.addVar(lb=0, ub=1) for w in W for b in B}
    Y = {(w, a): m.addVar(lb=0, ub=float(pp[a])) for w in W for a in A}
    Z = {(w, b): m.addVar(lb=0, ub=float(pm[b])) for w in W for b in B}
    F = {(w, x, v): m.addVar(lb=0, ub=1) for w in W for x in X for v in W}
    FQ = {(w, x, v, a): m.addVar(lb=0, ub=1) for w in W for x in X for v in W for a in A}

    m.addCons(quicksum(pi.values()) == 1)
    for w in W:
        m.addCons(s[w] >= pi[w])
        m.addCons(s[w] <= pi[w] + pi_min)
        m.addCons(e[w] >= -s[w] * scip.log(s[w]) / math.log(2))
        m.addCons(quicksum(Q[w, a] for a in A) == 1)
        m.addCons(quicksum(R[w, b] for b in B) == 1)
        m.addCons(quicksum(Y[w, a] for a in A) == pi[w])
        m.addCons(quicksum(Z[w, b] for b in B) == pi[w])
        for a in A:
            m.addCons(Y[w, a] == pi[w] * Q[w, a])
        for b in B:
            m.addCons(Z[w, b] == pi[w] * R[w, b])
        m.addCons(quicksum(F[w, x, v] for x in X for v in W) == pi[w])
        m.addCons(quicksum(F[u, x, w] for u in W for x in X) == pi[w])
        if w + 1 < k:
            m.addCons(pi[w] >= pi[w + 1])
    for w in W:
        for x in X:
            for v in W:
                m.addCons(quicksum(FQ[w, x, v, a] for a in A) == F[w, x, v])
                for a in A:
                    m.addCons(FQ[w, x, v, a] == F[w, x, v] * Q[w, a])
    for a in A:
        data_eq(quicksum(Y[w, a] for w in W), float(pp[a]))
    for b in B:
        data_eq(quicksum(Z[w, b] for w in W), float(pm[b]))
    for a in A:
        for b in B:
            data_eq(quicksum(Y[w, a] * R[w, b] for w in W), float(P[a, b]))
            data_eq(quicksum(Q[w, a] * Z[w, b] for w in W), float(P[a, b]))
    for a in A:
        for x in X:
            for b in B:
                data_eq(quicksum(FQ[w, x, v, a] * R[v, b] for w in W for v in W), float(Pe[a, x, b]))
    for v in W:  # forward stationarity: S+_1 is a function of (S+_0, X_0)
        for a2 in A:
            src = [(a, x) for a in A for x in X if fwd[a, x, a2]]
            m.addCons(Y[v, a2] == quicksum(FQ[w, x, v, a] for a, x in src for w in W))
    for w in W:  # reverse consistency: S-_0 is a function of (X_0, S-_1)
        for b in B:
            tgt = [(x, c) for x in X for c in B if rev[x, c, b]]
            m.addCons(Z[w, b] == quicksum(F[w, x, v] * R[v, c] for x, c in tgt for v in W))
    for j, cap in enumerate(list(caps)[:k]):
        m.addCons(quicksum(pi[w] for w in range(j + 1)) <= float(cap))
    if Pe2 is not None:
        V = {(w, x, v, y, u): m.addVar(lb=0, ub=1) for w in W for x in X for v in W for y in X for u in W}
        VQ = {(*key, a): m.addVar(lb=0, ub=1) for key in V for a in A}
        for w in W:
            for x in X:
                for v in W:
                    m.addCons(quicksum(V[w, x, v, y, u] for y in X for u in W) == F[w, x, v])
                    for y in X:
                        for u in W:
                            # the hidden chain is Markov: Pr(w, x, v, y, u) pi_v = Pr(w, x, v) Pr(v, y, u)
                            m.addCons(V[w, x, v, y, u] * pi[v] == F[w, x, v] * F[v, y, u])
                            m.addCons(quicksum(VQ[w, x, v, y, u, a] for a in A) == V[w, x, v, y, u])
                            for a in A:
                                m.addCons(VQ[w, x, v, y, u, a] == V[w, x, v, y, u] * Q[w, a])
        for v in W:
            for y in X:
                for u in W:
                    m.addCons(quicksum(V[w, x, v, y, u] for w in W for x in X) == F[v, y, u])
        for x in X:
            for y in X:
                for a in A:
                    for b in B:
                        data_eq(
                            quicksum(VQ[w, x, v, y, u, a] * R[u, b] for w in W for v in W for u in W),
                            float(Pe2[x, y, a, b]),
                        )
    m.setObjective(quicksum(e.values()), "minimize")
    if objective_limit is not None:
        m.setObjlimit(float(objective_limit))
    if time_limit is not None:
        m.setParam("limits/time", max(1.0, float(time_limit)))
    m.optimize()

    status = str(m.getStatus())
    slack = k * pi_min * math.log2(1 / pi_min)
    lower, dual = _bounds(m, status, objective_limit, slack)
    primal, seed = None, None
    if m.getNSols() > 0:
        primal = float(m.getPrimalbound())
        sol = m.getBestSol()
        pi_v = np.clip([m.getSolVal(sol, pi[w]) for w in W], 0, None)
        Q_v = np.clip([[m.getSolVal(sol, Q[w, a]) for a in A] for w in W], 0, None)
        R_v = np.clip([[m.getSolVal(sol, R[w, b]) for b in B] for w in W], 0, None)
        F_v = np.clip([[[m.getSolVal(sol, F[w, x, v]) for v in W] for x in X] for w in W], 0, None)
        K_v = F_v / np.maximum(F_v.sum(axis=(1, 2), keepdims=True), 1e-300)
        seed = (pi_v / pi_v.sum(), Q_v, R_v, K_v)
    return GlobalBoundResult(
        k=k,
        status=status,
        lower=lower,
        dual_bound=dual,
        slack=slack,
        primal=primal,
        seed=seed,
        nodes=int(m.getNNodes()),
        seconds=time.monotonic() - start,
        two_step=Pe2 is not None,
    )
