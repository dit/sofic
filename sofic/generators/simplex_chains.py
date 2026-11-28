r"""Exact one-sided generators built as simplex chains in belief coordinates.

A generator state :math:`w` is *reverse-sufficient* when its future is the :math:`R_w`-mixture of the reverse causal
states' futures, :math:`R_w = \Pr(S^-_0 \mid W_0 = w)`, and a generator is reverse-sufficient when all its states
are. Such a generator is a finitely supported law :math:`\mu` on the simplex :math:`\Delta(S^-)` (its weights on the
atoms :math:`R_w`). After a symbol :math:`x`, emitted with probability :math:`m_x(s) = (s P_x) \mathbf{1}`, a belief
:math:`s` moves to the *target* :math:`f_x(s) = s P_x / m_x(s)`, and the generator reproduces the process exactly when
each target :math:`f_x(R_w)` is the barycenter of the successor atoms under the generator's transition law. So the
image of :math:`\mu` under the belief maps is dominated by :math:`\mu` in convex order, and a law dominates another
in convex order exactly when a mean-preserving kernel carries one to the other :cite:`Strassen1965`.

This module turns that condition into a construction. A finite set :math:`S \subset \Delta(S^-)` is *invariant* when
every target of every point of :math:`S` lies in :math:`\operatorname{conv} S`. Writing each target as a convex
combination :math:`f_x(s) = \sum_{s'} \lambda^{s, x}_{s'} s'` gives a chain :math:`T^x[s, s'] = m_x(s)
\lambda^{s, x}_{s'}` on :math:`S`, and the stationary law of every closed class of :math:`T = \sum_x T^x` is an exact
reverse-sufficient generator whose beliefs are the atoms. :math:`H[\pi]` is concave on the polytope of these
representations, so its minimum over a pool is at a vertex: a *simplex chain*, in which each block's successors are
affinely independent. Forward-sufficient generators are the mirror image on :math:`\Delta(S^+)`, with the chain run
in reversed time.

:func:`simplex_chain_search` alternates two steps on each side:

* the *assignment step*, successive linear programming for :math:`\min H[\pi]` over the representation polytope of a
  pool of candidate atoms (the simplex vertices, the companion beliefs :math:`\rho_a`, images under the belief maps,
  fixed points of short words, random points and ray exits), each vertex projected onto its wall equations and
  rebuilt with exact barycentric flows;
* the *atom step*, which holds the assignment and moves the atoms along the gradient of :math:`H[\pi]` (through the
  chain's fundamental matrix) projected onto the walls it touches. A step that crosses a wall adds the moved atoms
  to the pool, so the next assignment step flips the assignment there.

Each side starts from its own ε-machine (the atoms at the vertices), so its result is never worse than that
machine's entropy, and every accepted iterate passes a word check over a Tzeng basis :cite:`Tzeng1992`. With cryptic
orders at most :math:`\kappa`, every generator synchronizes to both causal states within :math:`\kappa` symbols, so
the support of each atom is a face of the complex of states whose sets of length-:math:`\kappa` words are pairwise
disjoint, and pool points off that complex are dropped. The chain construction, the search and the synchronization
constraint are not from the literature.
"""

from __future__ import annotations

import itertools
import math
import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from functools import reduce
from typing import Any

import networkx as nx
import numpy as np
from scipy.optimize import linprog, nnls
from scipy.sparse import coo_matrix

from sofic.generators.bidirectional_epsilon_machine import BidirectionalEpsilonMachine
from sofic.generators.coupled_generators import _GeneratorProblem

__all__ = ["SimplexChainResult", "simplex_chain_search"]

_DEPTH, _POOL_CAP, _N_RANDOM, _N_RAYS, _STARTS = 3, 60, 12, 24, 6
_EXACT = 1e-10
# Faces of the synchronizing complex are enumerated only for small cryptic orders and side machines.
_MAX_FACE_ORDER, _MAX_FACE_STATES = 4, 30

Assignment = dict[tuple[int, int], tuple[int, ...]]


@dataclass(frozen=True)
class SimplexChainResult:
    """An exact one-sided generator from :func:`simplex_chain_search`.

    ``side`` is ``"reverse"`` (the atoms are beliefs on the reverse causal states) or ``"forward"``.
    ``K[w, x, v] = Pr(X_0 = x, W_1 = v | W_0 = w)`` in the process's time and symbol order (``problem.symbols``),
    ``atoms[w]`` is state ``w``'s belief on the side's causal states, and ``word_error`` is the largest error in a
    word probability over a Tzeng basis of the joint (generator, ε-machine) space. ``log`` lists the improvements as
    ``(event, entropy, states, seconds)``.
    """

    side: str
    entropy: float
    pi: np.ndarray
    K: np.ndarray
    atoms: np.ndarray
    word_error: float
    log: tuple[tuple[str, float, int, float], ...]

    @property
    def matrices(self) -> tuple[np.ndarray, ...]:
        """Symbol matrices ``T^(x)[w, v]`` of the generator."""
        return tuple(self.K[:, x, :] for x in range(self.K.shape[1]))


@dataclass
class _Side:
    """One side in its own time: ``P[x, i, j] = Pr(x, own state j next | own state i)``, and ``rho[a]`` is companion
    state ``a``'s belief on the own states."""

    name: str
    P: np.ndarray
    rho: np.ndarray
    kappa: float = math.inf

    @property
    def n(self) -> int:
        return self.P.shape[1]

    @property
    def n_sym(self) -> int:
        return self.P.shape[0]


def _entropy(p: Any) -> float:
    p = np.asarray(p, float).ravel()
    p = p[p > 1e-300]
    return float(-(p * np.log2(p)).sum())


def _sides(problem: _GeneratorProblem, kappa: float) -> dict[str, _Side]:
    lif = problem.lifted
    E = np.asarray(lif.edges, float)  # [p, x, q] = Pr(B_0 = p, X_0 = x, B_1 = q)
    p = np.asarray(lif.p, float)
    plus, minus = np.argmax(lif.plus, axis=1), np.argmax(lif.minus, axis=1)
    out = {}
    for name, edges, own, comp, n_own, n_comp in (
        ("reverse", E, minus, plus, problem.n_minus, problem.n_plus),
        ("forward", E.transpose(2, 1, 0), plus, minus, problem.n_plus, problem.n_minus),
    ):
        Mown, Mcomp = np.eye(n_own)[own], np.eye(n_comp)[comp]
        p_own = Mown.T @ p
        P = np.einsum("pi,pxq,qj->xij", Mown, edges, Mown) / p_own[None, :, None]
        J = Mcomp.T @ (p[:, None] * Mown)
        out[name] = _Side(name, P, J / J.sum(axis=1, keepdims=True), kappa)
    return out


# --- the synchronizing complex -----------------------------------------------------------------------------------


def _emit_words(side: _Side, L: int, tol: float = 1e-12) -> np.ndarray:
    """``W[i, j]``: own state ``i`` emits the ``j``-th word of length ``L`` (side time)."""
    vec = {(): np.ones(side.n)}
    for _ in range(L):
        vec = {(x, *u): side.P[x] @ v for u, v in vec.items() for x in range(side.n_sym)}
    return np.array([vec[u] > tol for u in sorted(vec)]).T


def _sync_complex(side: _Side, L: int) -> list[tuple[int, ...]]:
    """Maximal sets of own states with pairwise disjoint sets of length-``L`` words."""
    W = _emit_words(side, L).astype(int)
    share = (W @ W.T) > 0
    G = nx.Graph()
    G.add_nodes_from(range(side.n))
    G.add_edges_from((i, j) for i in range(side.n) for j in range(i + 1, side.n) if not share[i, j])
    return sorted((tuple(sorted(c)) for c in nx.find_cliques(G)), key=lambda c: (-len(c), c))


def _in_complex(support: Iterable[int], faces: list[tuple[int, ...]]) -> bool:
    s = {int(i) for i in support}
    return any(s <= set(f) for f in faces)


def _faces(side: _Side) -> list[tuple[int, ...]]:
    if math.isfinite(side.kappa) and side.kappa <= _MAX_FACE_ORDER and side.n <= _MAX_FACE_STATES:
        return _sync_complex(side, int(side.kappa))
    return [tuple(range(side.n))]


def _allowed(point: np.ndarray, faces: list[tuple[int, ...]], tol: float = 1e-12) -> bool:
    return _in_complex(np.flatnonzero(point > tol), faces)


# --- pools -----------------------------------------------------------------------------------------------------


def _fixed_points(P: np.ndarray, max_len: int = 2) -> list[np.ndarray]:
    """Beliefs fixed by the maps of the words of length ``1 .. max_len`` (nonnegative left eigenvectors)."""
    out = []
    for length in range(1, max_len + 1):
        for u in itertools.product(range(P.shape[0]), repeat=length):
            M = reduce(np.matmul, (P[x] for x in u))
            vals, vecs = np.linalg.eig(M.T)
            for lam, v in zip(vals, vecs.T, strict=True):
                if abs(lam.imag) > 1e-10 or lam.real <= 1e-10:
                    continue
                v = v.real * np.sign(v.real.sum() or 1.0)
                if v.min() < -1e-9 * np.abs(v).max() or v.sum() <= 0:
                    continue
                v = np.clip(v, 0, None)
                out.append(v / v.sum())
    return out


def _close(side: _Side, seeds: list[np.ndarray], depth: int, cap: int) -> list[np.ndarray]:
    """The seeds and their images under the belief maps, up to ``depth`` steps or ``cap`` points."""
    pts: list[np.ndarray] = []
    keys: set[tuple[float, ...]] = set()

    def add(v: np.ndarray) -> bool:
        key = tuple(np.round(v, 10))
        if key in keys:
            return False
        keys.add(key)
        pts.append(v)
        return True

    frontier = [np.asarray(v, float) for v in seeds if add(np.asarray(v, float))]
    for _ in range(depth):
        new = []
        for v in frontier:
            for x in range(side.n_sym):
                y = v @ side.P[x]
                if y.sum() > 1e-12 and len(pts) < cap and add(y / y.sum()):
                    new.append(pts[-1])
        frontier = new
    return pts


def _random_points(faces: list[tuple[int, ...]], n: int, rng: np.random.Generator, dim: int) -> list[np.ndarray]:
    out = []
    big = [f for f in faces if len(f) > 1]
    for _ in range(n if big else 0):
        f = big[rng.integers(len(big))]
        v = np.zeros(dim)
        v[list(f)] = rng.dirichlet(np.ones(len(f)))
        out.append(v)
    return out


def _dedupe(points: Sequence[np.ndarray], tol: float = 1e-9) -> np.ndarray:
    out: list[np.ndarray] = []
    for p in points:
        if all(np.abs(p - q).max() > tol for q in out):
            out.append(np.asarray(p, float))
    return np.array(out)


def _ray_exits(
    points: Sequence[np.ndarray],
    targets: Sequence[np.ndarray],
    faces: list[tuple[int, ...]],
    cap: int,
    rng: np.random.Generator,
) -> list[np.ndarray]:
    """Where the ray from a point through a target leaves the simplex, when the target lies strictly inside that
    segment: the farthest atom that can carry the target on a two-atom block with the point."""
    out = []
    for p in points:
        for q in targets:
            d = np.asarray(q) - np.asarray(p)
            neg = d < -1e-12
            if not neg.any():
                continue
            t = float(np.min(p[neg] / -d[neg]))
            if t <= 1 + 1e-9:
                continue
            e = np.clip(p + t * d, 0.0, None)
            e /= e.sum()
            if _allowed(e, faces):
                out.append(e)
    exits = _dedupe(out) if out else np.zeros((0, len(points[0]) if len(points) else 0))
    if len(exits) > cap:
        exits = exits[rng.choice(len(exits), cap, replace=False)]
    return list(exits)


def _base_pool(side: _Side, rng: np.random.Generator) -> tuple[np.ndarray, list[tuple[int, ...]]]:
    faces = _faces(side)
    seeds = [*np.eye(side.n), *side.rho, *_fixed_points(side.P)]
    pts = [p for p in _close(side, seeds, _DEPTH, _POOL_CAP) if _allowed(p, faces)]
    core = [p for p in _close(side, seeds, 1, _POOL_CAP) if _allowed(p, faces)]
    pts += _ray_exits(core, core, faces, _N_RAYS, rng)
    pts += _random_points(faces, _N_RANDOM, rng, side.n)
    return _dedupe(pts), faces


def _images(side: _Side, atoms: np.ndarray) -> list[np.ndarray]:
    out = []
    for a in atoms:
        for x in range(side.n_sym):
            y = a @ side.P[x]
            if y.sum() > 1e-12:
                out.append(y / y.sum())
    return out


# --- chains: flows, closed classes, generators -------------------------------------------------------------------


def _blocks(T: np.ndarray, rel: float = 1e-6, floor: float = 1e-14) -> Assignment:
    """``{(s, x): successors}`` of a chain, dropping blocks of mass at most ``floor`` and flows below ``rel`` times
    the block's mass."""
    out = {}
    for x, s in itertools.product(range(T.shape[0]), range(T.shape[1])):
        m = T[x, s].sum()
        if m > floor:
            out[(s, x)] = tuple(int(v) for v in np.flatnonzero(T[x, s] > rel * m))
    return out


def _complete_assign(side: _Side, atoms: np.ndarray, assign: Assignment, tol: float = 1e-12) -> tuple[Assignment, int]:
    """Give every unassigned block with ``m_x(s) > tol`` the successors of a basic solution of its representation LP.
    Returns the assignment and the number of targets outside ``conv(atoms)``."""
    out, outside = dict(assign), 0
    A = np.vstack([atoms.T, np.ones(len(atoms))])
    for s in range(len(atoms)):
        for x in range(side.n_sym):
            y = atoms[s] @ side.P[x]
            if (s, x) in out or y.sum() <= tol:
                continue
            r = linprog(
                np.zeros(len(atoms)), A_eq=A, b_eq=np.append(y / y.sum(), 1.0), bounds=(0, None), method="highs-ds"
            )
            if r.status != 0:
                outside += 1
                continue
            out[(s, x)] = tuple(int(v) for v in np.flatnonzero(r.x > 1e-12))
    return out, outside


def _transport(side: _Side, atoms: np.ndarray, assign: Assignment) -> tuple[np.ndarray, float, float]:
    """``T[x]`` with least-squares barycentric flows, the largest representation residual and the most negative
    flow."""
    N = len(atoms)
    T = np.zeros((side.n_sym, N, N))
    worst, neg = 0.0, 0.0
    for (s, x), succ in assign.items():
        idx = list(succ)
        y = atoms[s] @ side.P[x]
        t = np.linalg.lstsq(atoms[idx].T, y, rcond=None)[0]
        if t.min() < -1e-12:  # dependent successors: any nonnegative representation will do
            t = nnls(atoms[idx].T, y)[0]
        T[x, s, idx] = t
        worst = max(worst, float(np.abs(t @ atoms[idx] - y).max()))
        neg = min(neg, float(t.min()))
    return T, worst, neg


def _closed_classes(M: np.ndarray) -> list[list[int]]:
    """Closed classes of a substochastic matrix, ignoring states without outflow (atoms that carry no block)."""
    G = nx.DiGraph()
    G.add_nodes_from(range(len(M)))
    G.add_edges_from(zip(*np.nonzero(M > 0), strict=True))
    C = nx.condensation(G)
    rows = M.sum(axis=1)
    return [
        sorted(C.nodes[c]["members"])
        for c in C
        if C.out_degree(c) == 0 and min(rows[list(C.nodes[c]["members"])]) > 0.5
    ]


def _stationary_on(M: np.ndarray, cls: list[int]) -> np.ndarray:
    sub = M[np.ix_(cls, cls)]
    k = len(cls)
    A = np.vstack([sub.T - np.eye(k), np.ones(k)])
    b = np.zeros(k + 1)
    b[-1] = 1.0
    pi = np.zeros(len(M))
    pi[cls] = np.linalg.lstsq(A, b, rcond=None)[0]
    return pi


def _stationaries(T: np.ndarray) -> list[np.ndarray]:
    M = T.sum(axis=0)
    return [_stationary_on(M, c) for c in _closed_classes(M)]


def _to_generator(side: _Side, T: np.ndarray, pi: np.ndarray, tol: float = 1e-13) -> tuple[np.ndarray, ...]:
    """``(pi, K[w, x, v], kept atoms)`` in the process's time from a side chain and one of its stationary laws."""
    keep = np.flatnonzero(pi > tol)
    T = T[:, keep][:, :, keep]
    pi = pi[keep] / pi[keep].sum()
    K = T.transpose(1, 0, 2) if side.name == "reverse" else np.einsum("v,xvw->wxv", pi, T) / pi[:, None, None]
    return pi, K, keep


def _process_matrices(problem: _GeneratorProblem) -> tuple[np.ndarray, np.ndarray]:
    lif = problem.lifted
    Mp = np.eye(problem.n_plus)[np.argmax(lif.plus, axis=1)]
    T = np.einsum("pa,pxq,qb->xab", Mp, lif.edges, Mp) / problem.p_plus[None, :, None]
    return np.asarray(problem.p_plus, float), T


def _word_error(problem: _GeneratorProblem, pi: np.ndarray, K: np.ndarray) -> float:
    """Largest error in a word probability over a Tzeng basis of the joint (generator, ε-machine) space."""
    p0, Tp = _process_matrices(problem)
    k = len(pi)
    mats = [
        np.block([[K[:, x, :], np.zeros((k, len(p0)))], [np.zeros((len(p0), k)), Tp[x]]]) for x in range(K.shape[1])
    ]
    basis: list[np.ndarray] = []
    queue, worst = [np.concatenate([pi, p0])], 0.0
    while queue:
        v = queue.pop(0)
        r = v.copy()
        for b in basis:
            r -= (r @ b) * b
        if np.linalg.norm(r) <= 1e-9 * max(1.0, np.linalg.norm(v)):
            continue
        basis.append(r / np.linalg.norm(r))
        worst = max(worst, abs(v[:k].sum() - v[k:].sum()))
        queue.extend(v @ M for M in mats)
    return float(worst)


def _exact_generator(
    side: _Side, problem: _GeneratorProblem, atoms: np.ndarray, assign: Assignment
) -> dict[str, Any] | None:
    """Rebuild a chain from atoms and an assignment, keep its lowest-entropy closed class, and word-check it."""
    T, res, neg = _transport(side, atoms, assign)
    best: dict[str, Any] | None = None
    for pi in _stationaries(T):
        g_pi, K, keep = _to_generator(side, T, pi)
        H = _entropy(g_pi)
        if best is None or best["entropy"] > H:
            best = {"entropy": H, "pi": g_pi, "K": K, "keep": keep, "atoms": atoms[keep], "residual": res}
            best["negative"] = neg
    if best is not None:
        best["word_error"] = _word_error(problem, best["pi"], best["K"])
    return best


# --- walls --------------------------------------------------------------------------------------------------------


def _wall_residuals(side: _Side, atoms: np.ndarray, assign: Assignment, faces: list[tuple[int, ...]]) -> np.ndarray:
    """Each block's target in the affine hull of its successors, and the zero coordinates ``faces[s]`` of atom
    ``s``, as stacked residuals."""
    out = []
    for (s, x), idx in sorted(assign.items()):
        y = atoms[s] @ side.P[x]
        f = y / max(y.sum(), 1e-300)
        base = atoms[idx[0]]
        D = (atoms[list(idx[1:])] - base).T if len(idx) > 1 else np.zeros((len(f), 0))
        r = f - base
        if D.shape[1]:
            r = r - D @ np.linalg.lstsq(D, r, rcond=None)[0]
        out.append(r)
    for s, zeros in enumerate(faces):
        out.append(atoms[s, list(zeros)])
    return np.concatenate(out) if out else np.zeros(0)


def _snap(
    side: _Side, atoms: np.ndarray, assign: Assignment, tol: float = 1e-7, rounds: int = 30
) -> tuple[np.ndarray, float]:
    """Gauss-Newton projection of approximate atoms onto their wall equations, keeping their zero coordinates.
    Returns the atoms and the final residual."""
    N, n = atoms.shape
    faces = [tuple(int(c) for c in np.flatnonzero(a <= tol)) for a in atoms]
    A0 = np.clip(atoms, 0.0, None)
    for s, z in enumerate(faces):
        A0[s, list(z)] = 0.0
    A0 /= A0.sum(axis=1, keepdims=True)
    free = [(s, c) for s in range(N) for c in range(n) if c not in faces[s]]
    # every free coordinate but the last of each atom is a variable; the last is 1 minus the others
    last = {s: max(c for t, c in free if t == s) for s in range(N)}
    var = [(s, c) for s, c in free if c != last[s]]
    none = [()] * N

    def lift(theta: np.ndarray) -> np.ndarray:
        A = np.zeros((N, n))
        for (s, c), v in zip(var, theta, strict=True):
            A[s, c] = v
        for s in range(N):
            A[s, last[s]] = 1.0 - A[s].sum()
        return A

    theta = np.array([A0[s, c] for s, c in var])
    r = _wall_residuals(side, lift(theta), assign, none)
    for _ in range(rounds):
        if not len(r) or np.abs(r).max() < 1e-15:
            break
        h = 1e-8
        J = np.empty((len(r), len(theta)))
        for j in range(len(theta)):
            d = np.zeros_like(theta)
            d[j] = h
            J[:, j] = (
                _wall_residuals(side, lift(theta + d), assign, none)
                - _wall_residuals(side, lift(theta - d), assign, none)
            ) / (2 * h)
        if not (np.isfinite(J).all() and np.isfinite(r).all()):  # LAPACK's SVD can hang on NaN input
            return lift(theta), math.inf
        try:
            theta = theta - np.linalg.lstsq(J, r, rcond=None)[0]
            r = _wall_residuals(side, lift(theta), assign, none)
        except np.linalg.LinAlgError:  # the SVD can fail to converge on finite input too
            return lift(theta), math.inf
    return lift(theta), float(np.abs(r).max()) if len(r) else 0.0


def _valid(ex: dict[str, Any] | None) -> bool:
    return ex is not None and ex["word_error"] <= _EXACT and ex["negative"] >= -1e-12 and ex["residual"] <= 1e-12


def _realize(side: _Side, problem: _GeneratorProblem, atoms: np.ndarray, assign: Assignment) -> dict[str, Any] | None:
    """Snap a chain onto its walls, complete its assignment and rebuild it exactly; ``None`` unless exact."""
    snapped, res = _snap(side, atoms, assign, tol=1e-12)
    if res > 1e-12 or snapped.min() < -1e-13:
        return None
    snapped = np.clip(snapped, 0.0, None)
    snapped /= snapped.sum(axis=1, keepdims=True)
    full, outside = _complete_assign(side, snapped, assign)
    if outside:
        return None
    ex = _exact_generator(side, problem, snapped, full)
    if ex is None or not _valid(ex):
        return None
    index = {int(s): i for i, s in enumerate(ex["keep"])}
    ex["assign"] = {
        (index[s], x): tuple(index[v] for v in idx)
        for (s, x), idx in full.items()
        if s in index and all(v in index for v in idx)
    }
    return ex


# --- the assignment step -------------------------------------------------------------------------------------------


def _lifted_lp(side: _Side, S: np.ndarray, c: np.ndarray, tol: float = 1e-12) -> tuple[Any, ...] | None:
    """Minimize ``c . pi`` over the representation polytope of the pool ``S``: flows ``g[s, x, s'] >= 0`` with
    ``sum_s' g[s, x, s'] S[s'] = pi_s S[s] P_x``, inflow ``pi`` and ``sum pi = 1``, offered only to atoms in the
    target's face. Returns ``(pi, T, assignment)`` of a basic optimum, or ``None`` if infeasible."""
    N, n = S.shape
    var: list[tuple[int, int, int]] = []
    offers = {}
    for s in range(N):
        for x in range(side.n_sym):
            y = S[s] @ side.P[x]
            if y.sum() <= tol:
                continue
            face = y > tol
            succ = [v for v in range(N) if not (S[v] > tol)[~face].any()]
            offers[(s, x)] = (len(var), succ, y)
            var.extend((s, x, v) for v in succ)
    nv, ng = N + len(var), len(var)
    rows, cols, vals, b = [], [], [], []
    r = 0
    for (s, _x), (start, succ, y) in offers.items():
        for coord in range(n):
            for j, v in enumerate(succ):
                if S[v, coord] != 0:
                    rows.append(r)
                    cols.append(N + start + j)
                    vals.append(S[v, coord])
            if y[coord] != 0:
                rows.append(r)
                cols.append(s)
                vals.append(-y[coord])
            b.append(0.0)
            r += 1
    for g, (_s, _x, v) in enumerate(var):
        rows.append(r + v)
        cols.append(N + g)
        vals.append(1.0)
    for v in range(N):
        rows.append(r + v)
        cols.append(v)
        vals.append(-1.0)
        b.append(0.0)
    r += N
    rows.extend([r] * N)
    cols.extend(range(N))
    vals.extend([1.0] * N)
    b.append(1.0)
    A = coo_matrix((vals, (rows, cols)), shape=(r + 1, nv)).tocsr()
    cost = np.concatenate([np.asarray(c, float), np.zeros(ng)])
    res = linprog(cost, A_eq=A, b_eq=np.array(b), bounds=(0, None), method="highs-ds")
    if res.status != 0:
        return None
    pi, g = res.x[:N], res.x[N:]
    T = np.zeros((side.n_sym, N, N))
    for gi, (s, x, v) in enumerate(var):
        if pi[s] > 1e-14:
            T[x, s, v] += g[gi] / pi[s]
    return pi, T, _blocks(T, rel=1e-9)


def _slp(
    side: _Side, S: np.ndarray, start: np.ndarray | None = None, rounds: int = 60, floor: float = 1e-12
) -> tuple[Any, ...] | None:
    """Successive linear programming for ``min H(pi)`` over the representation polytope of ``S``, from the
    linearization at ``start`` (default uniform)."""
    pi = np.full(len(S), 1.0 / len(S)) if start is None else np.asarray(start, float)
    best = None
    for _ in range(rounds):
        out = _lifted_lp(side, S, -np.log2(np.maximum(pi, floor)))
        if out is None:
            return best
        if best is not None and _entropy(out[0]) >= _entropy(best[0]) - 1e-13:
            break
        best, pi = out, out[0]
    return best


def _assignment_step(
    side: _Side,
    problem: _GeneratorProblem,
    pool: np.ndarray,
    rng: np.random.Generator,
    starts: int,
    extra_start: np.ndarray | None = None,
) -> dict[str, Any] | None:
    """The best exact chain over sequential-LP runs on the pool (random linearizations, plus ``extra_start``)."""
    best = None
    inits = [rng.dirichlet(0.3 * np.ones(len(pool))) for _ in range(starts)]
    if extra_start is not None:
        inits.insert(0, extra_start)
    for init in inits:
        out = _slp(side, pool, start=init)
        if out is None:
            continue
        pi, _T, assign = out
        supp = np.flatnonzero(pi > 1e-12)
        index = {int(s): i for i, s in enumerate(supp)}
        sub = {
            (index[s], x): tuple(index[v] for v in idx if v in index) for (s, x), idx in assign.items() if s in index
        }
        ex = _realize(side, problem, pool[supp], {b: idx for b, idx in sub.items() if idx})
        if ex is not None and (best is None or ex["entropy"] < best["entropy"] - 1e-13):
            best = ex
    return best


# --- the atom step --------------------------------------------------------------------------------------------------


def _chamber_gradient(side: _Side, atoms: np.ndarray, assign: Assignment, pi: np.ndarray | None = None) -> np.ndarray:
    """``dH/d atoms`` (bits) with the assignment held: the flows solve ``atoms[idx]^T t = atoms[s] P_x``, the
    stationary law moves by ``d pi = pi dT Z`` with ``Z = (I - T + 1 pi)^{-1}``, and
    ``dH = -sum (log2 pi + 1/ln 2) d pi``. Valid while every target stays in the affine hull of its successors."""
    T, _, _ = _transport(side, atoms, assign)
    M = T.sum(axis=0)
    N = len(atoms)
    if pi is None:
        pi = _stationaries(T)[0]
    Z = np.linalg.inv(np.eye(N) - M + np.outer(np.ones(N), pi))
    hvec = Z @ -(np.log2(np.maximum(pi, 1e-300)) + 1.0 / math.log(2.0))
    grad = np.zeros_like(atoms)
    for (s, x), succ in assign.items():
        idx = list(succ)
        t = T[x, s, idx]
        q = np.linalg.pinv(atoms[idx].T).T @ hvec[idx]
        grad[s] += pi[s] * (side.P[x] @ q)
        for j, v in enumerate(idx):
            grad[v] -= pi[s] * t[j] * q
    return grad


def _wall_tangent(
    side: _Side,
    atoms: np.ndarray,
    assign: Assignment,
    tol: float = 1e-12,
    h: float = 1e-7,
    deadline: float = math.inf,
) -> np.ndarray | None:
    """Orthonormal basis of the directions that keep every wall equation, zero coordinate and unit sum to first
    order, or ``None`` if ``deadline`` (a :func:`time.monotonic` value) passes first."""
    N, n = atoms.shape
    faces = [tuple(int(c) for c in np.flatnonzero(a <= tol)) for a in atoms]

    def eqs(flat: np.ndarray) -> np.ndarray:
        A = flat.reshape(N, n)
        return np.concatenate([_wall_residuals(side, A, assign, faces), A.sum(axis=1) - 1.0])

    flat = atoms.ravel()
    J = np.empty((len(eqs(flat)), flat.size))
    for j in range(flat.size):
        # ~10^4 columns of two wall solves each on the ~100-state reverse side of a truncated build
        if time.monotonic() > deadline:
            return None
        d = np.zeros_like(flat)
        d[j] = h
        J[:, j] = (eqs(flat + d) - eqs(flat - d)) / (2 * h)
    _, s, Vt = np.linalg.svd(J)
    rank = int((s > 1e-8 * max(s[0], 1e-300)).sum())
    return Vt[rank:].T


def _atom_step(
    side: _Side,
    problem: _GeneratorProblem,
    chain: dict[str, Any],
    max_iter: int = 15,
    deadline: float = math.inf,
) -> tuple[dict[str, Any], list[np.ndarray]]:
    """Projected gradient descent with the assignment held, stopping at ``deadline``. Returns the improved exact
    chain (or the input) and the moved atoms of steps that crossed a wall."""
    atoms, assign, H = chain["atoms"], chain["assign"], chain["entropy"]
    crossed = []
    for _ in range(max_iter):
        if time.monotonic() > deadline:
            break
        try:
            grad = _chamber_gradient(side, atoms, assign, chain["pi"])
        except np.linalg.LinAlgError:
            break
        B = _wall_tangent(side, atoms, assign, deadline=deadline)
        if B is None or not B.shape[1]:
            break
        g = B @ (B.T @ grad.ravel())
        norm = np.linalg.norm(g)
        if norm < 1e-10:
            break
        t, moved = 0.05 / norm, False
        for _ in range(12):
            cand = atoms - t * g.reshape(atoms.shape)
            if cand.min() < -1e-13:
                t /= 2
                continue
            ex = _realize(side, problem, cand, assign)
            if ex is None:
                clipped = np.clip(cand, 0.0, None)
                crossed.append(clipped / clipped.sum(axis=1, keepdims=True))
            elif ex["entropy"] < H - 1e-13:
                chain, atoms, assign, H, moved = ex, ex["atoms"], ex["assign"], ex["entropy"], True
                break
            t /= 2
        if not moved:
            break
    return chain, crossed


def _descend(
    side: _Side,
    problem: _GeneratorProblem,
    pool: np.ndarray,
    faces: list[tuple[int, ...]],
    chain: dict[str, Any],
    rng: np.random.Generator,
    log: list[tuple[str, float, int, float]],
    t0: float,
    deadline: float,
) -> dict[str, Any]:
    """Alternate atom steps and flips from ``chain`` until neither improves it."""
    for _ in range(12):
        if time.monotonic() > deadline:
            break
        improved = False
        moved, crossed = _atom_step(side, problem, chain, deadline=deadline)
        if moved["entropy"] < chain["entropy"] - 1e-13:
            chain, improved = moved, True
            log.append(("atoms", chain["entropy"], len(chain["atoms"]), time.monotonic() - t0))
        if time.monotonic() > deadline:
            break
        extra = [*chain["atoms"], *_images(side, chain["atoms"]), *[a for c in crossed[-3:] for a in c]]
        extra += _ray_exits(list(chain["atoms"]), _images(side, chain["atoms"]), faces, _N_RAYS // 2, rng)
        extra += _random_points(faces, 4, rng, side.n)
        pool = _dedupe([*pool[:_POOL_CAP], *[p for p in extra if _allowed(p, faces)]])
        start = np.array([0.5 if any(np.abs(p - a).max() < 1e-9 for a in chain["atoms"]) else 1e-3 for p in pool])
        cand = _assignment_step(side, problem, pool, rng, 2, extra_start=start / start.sum())
        if cand is not None and cand["entropy"] < chain["entropy"] - 1e-13:
            chain, improved = cand, True
            log.append(("flip", chain["entropy"], len(chain["atoms"]), time.monotonic() - t0))
        if not improved:
            break
    return chain


def _search_side(
    side: _Side,
    problem: _GeneratorProblem,
    seconds: float,
    rng: np.random.Generator,
    log: list[tuple[str, float, int, float]],
    restarts: int = 10**6,
    target: float = -math.inf,
) -> dict[str, Any] | None:
    """The side's ε-machine descended first, then restarts from fresh pools until the time or the restarts run out,
    or the entropy reaches ``target``."""
    t0 = time.monotonic()
    deadline = t0 + seconds
    best = None
    blocks = {(s, x): tuple(range(side.n)) for x in range(side.n_sym) for s in range(side.n) if side.P[x, s].sum() > 0}
    vertex = _realize(side, problem, np.eye(side.n), blocks)
    if vertex is not None:
        pool, faces = _base_pool(side, rng)
        best = _descend(side, problem, pool, faces, vertex, rng, log, t0, deadline)
        log.append(("restart vertices", best["entropy"], len(best["atoms"]), time.monotonic() - t0))
    for r in range(restarts):
        if time.monotonic() > deadline or (best is not None and best["entropy"] <= target):
            break
        pool, faces = _base_pool(side, rng)
        chain = _assignment_step(side, problem, pool, rng, _STARTS)
        if chain is None:
            continue
        chain = _descend(side, problem, pool, faces, chain, rng, log, t0, deadline)
        if best is None or chain["entropy"] < best["entropy"] - 1e-13:
            best = chain
            log.append((f"restart {r}", best["entropy"], len(best["atoms"]), time.monotonic() - t0))
    return best


def _max_cryptic_order(bidir: BidirectionalEpsilonMachine) -> float:
    try:
        return float(max(bidir.forward_machine.cryptic_order(), bidir.reverse_machine.cryptic_order()))
    except Exception:  # noqa: BLE001 - without the orders every face of the simplex is allowed
        return math.inf


def simplex_chain_search(
    bidir: BidirectionalEpsilonMachine | _GeneratorProblem,
    *,
    sides: Sequence[str] = ("reverse", "forward"),
    seconds: float = 10.0,
    deadline: float | None = None,
    kappa: float | None = None,
    restarts: int = 10**6,
    target: float = -math.inf,
    rng: np.random.Generator | None = None,
) -> list[SimplexChainResult]:
    """Exact one-sided generators of the process, one per side in ``sides``, lowest entropy first; see the module
    docstring.

    Each side gets ``seconds``, or an equal share of the time left before ``deadline`` (a :func:`time.monotonic`
    value) if that is less; ``restarts`` caps the restarts from fresh pools after the ε-machine's descent. The
    search stops once a generator's entropy is at most ``target`` (a lower bound on :math:`C_g` plus a tolerance).
    ``kappa`` is the larger cryptic order, computed from the ε-machines when ``None``; with ``kappa`` at most 4 the
    atoms are kept on the faces of the synchronizing complex.
    """
    problem = bidir if isinstance(bidir, _GeneratorProblem) else _GeneratorProblem(bidir)
    rng = rng or np.random.default_rng()
    if kappa is None:
        kappa = _max_cryptic_order(problem.bidir)
    by_name = _sides(problem, kappa)
    out = []
    for i, name in enumerate(sides):
        budget = seconds
        if deadline is not None:
            budget = min(budget, max(0.0, deadline - time.monotonic()) / (len(sides) - i))
        log: list[tuple[str, float, int, float]] = []
        best = _search_side(by_name[name], problem, budget, rng, log, restarts=restarts, target=target)
        if best is not None:
            out.append(
                SimplexChainResult(
                    side=name,
                    entropy=float(best["entropy"]),
                    pi=np.asarray(best["pi"], float),
                    K=np.asarray(best["K"], float),
                    atoms=np.asarray(best["atoms"], float),
                    word_error=float(best["word_error"]),
                    log=tuple(log),
                )
            )
            if best["entropy"] <= target:
                break
    return sorted(out, key=lambda r: r.entropy)
