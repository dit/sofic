r"""Partial-sum (coverage) lower bounds on exact common information from rank-1 decompositions.

For :math:`M = p(s^+, s^-)` and an exact decomposition into nonnegative rank-1 terms with weights
:math:`\lambda_w`, the :math:`k` largest weights sum to at most :math:`\mu_k(M)` — the maximum total
mass of :math:`k` rank-1 terms that fit entrywise under :math:`M`. Entropy is Schur-concave, so
:math:`\operatorname{G}(S^+ : S^-) \ge \operatorname{H}(\mu_1, \mu_2 - \mu_1, \ldots)` for any
certified **upper** bounds on the :math:`\mu_k`.
"""

from __future__ import annotations

import heapq
import math
from collections.abc import Iterable
from dataclasses import dataclass
from itertools import combinations, count, product

import numpy as np
from scipy.optimize import differential_evolution, linprog, minimize


@dataclass(frozen=True)
class Mu1Result:
    """Estimate of ``mu_1(M)`` with optional witness on the probability simplices."""

    value: float
    certified: bool
    u: np.ndarray | None = None
    v: np.ndarray | None = None


def mu1_rank1_mass(M: np.ndarray, u: np.ndarray, v: np.ndarray) -> float:
    """Rank-1 mass ``lambda`` with ``lambda u v^T <= M`` entrywise (equals ``min_{a,b} M/(u_a v_b)``)."""
    M = np.asarray(M, dtype=float)
    u = np.asarray(u, dtype=float)
    v = np.asarray(v, dtype=float)
    ratios = M / np.maximum(np.outer(u, v), 1e-300)
    return float(np.min(ratios))


def joint_matrix(joint: dict[tuple, float]) -> tuple[np.ndarray, list, list]:
    """Dense matrix and axis labels for a small joint dict."""
    splus = sorted({a for a, _ in joint}, key=repr)
    sminus = sorted({b for _, b in joint}, key=repr)
    M = np.zeros((len(splus), len(sminus)))
    for (a, b), p in joint.items():
        M[splus.index(a), sminus.index(b)] = p
    return M, splus, sminus


def coverage_entropy_lower(mu_caps: Iterable[float]) -> float:
    r"""Least entropy (bits) of a distribution whose ``j`` heaviest weights sum to at most ``mu_caps[j - 1]``.

    For nondecreasing caps :math:`c_1 \le \dots \le c_k` the increments :math:`(c_1, c_2 - c_1, \ldots, 1 - c_k)`
    majorize every distribution under the caps, so their entropy is a lower bound (entropy is Schur-concave
    :cite:`marshall2011inequalities`), and the least entropy when they do not increase. When they do, the least
    entropy is larger. Given the heaviest :math:`k` weights, with smallest :math:`w_k`, the rest of mass :math:`m`
    has least entropy as :math:`\lfloor m / w_k \rfloor` weights :math:`w_k` and a remainder (that rest majorizes
    every other), and for a fixed count of them the entropy is concave in the heaviest weights. Its minimum is then
    at a vertex of their polytope or at a point of an edge where the remainder vanishes. Counts above :math:`Q` put
    every weight from :math:`w_k` on at most :math:`r / (Q + 1)` with :math:`r = 1 - S_{k-1}`, so the entropy is at
    least :math:`\sum_{i<k} \phi(w_i) + \phi(r) + r \log_2 (Q + 1)` with :math:`\phi(w) = -w \log_2 w`, which is
    concave in the heaviest :math:`k - 1` weights; :math:`Q` doubles until the least value of that bound at the
    vertices of their polytope reaches the least candidate value, or until :math:`Q = 2^{20}`, where the result is
    the smaller of the two and still a lower bound. Only the first eight caps enter that search; later ones enter
    through the increment entropy alone. The construction is this package's; we know of no literature source for it.
    The result is never below the increment entropy.
    """
    caps = np.asarray(list(mu_caps), float)
    # S_j <= S_{j+1} <= c_{j+1}, and caps from the first one at 1 on constrain nothing.
    c = np.minimum.accumulate(np.clip(caps, 0.0, 1.0)[::-1])[::-1]
    c = c[: int(np.argmax(c >= 1.0)) if np.any(c >= 1.0) else len(c)]
    floor = _increment_entropy(c)
    c = c[:_HEAD_LIMIT]
    if len(c) == 0 or c[0] <= 0.0:
        return floor
    edges = _head_edges(c)
    if not edges:
        return floor
    best = float(_with_least_tail(np.array([p for edge in edges for p in edge])).min())
    lighter = _lighter_heads(c)
    done, q = 0, 64
    while True:
        for p0, p1 in edges:
            heads = _remainder_free_heads(p0, p1, done + 1, q + 1)
            if len(heads):
                best = min(best, float(_with_least_tail(heads).min()))
        rest = float(np.min(lighter[0] + lighter[1] * math.log2(q + 1)))
        if best <= floor or rest >= best or q >= _TAIL_COUNT_LIMIT:
            break
        done, q = q + 1, 2 * q
    return max(floor, min(best, rest))


# Counts of tail weights equal to the k-th heaviest at which coverage_entropy_lower stops doubling Q.
_TAIL_COUNT_LIMIT = 1 << 20
# The head polytope of k caps has C(2k, k - 1) candidate edges.
_HEAD_LIMIT = 8


def _increment_entropy(caps: np.ndarray) -> float:
    """Entropy of the cap increments, with the mass above the last cap as one more weight."""
    if len(caps) == 0:
        return 0.0
    inc = np.clip(np.diff(np.r_[0.0, caps]), 0.0, 1.0)
    inc = np.r_[inc, max(0.0, 1.0 - caps[-1])]
    inc = inc[inc > 0]
    return float(_phi(inc / inc.sum()).sum())


def _phi(w: np.ndarray) -> np.ndarray:
    w = np.clip(w, 0.0, None)
    return np.where(w > 0, -w * np.log2(np.where(w > 0, w, 1.0)), 0.0)


def _head_edges(c: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    """End points of the edges of the polytope of sorted ``h`` in ``R^k`` with ``h_k >= 0`` and ``S_j(h) <= c_j``.

    A degenerate edge is a vertex twice, and every vertex is the end of some edge.
    """
    k = len(c)
    eye = np.eye(k)
    rows = [eye[i + 1] - eye[i] for i in range(k - 1)] + [-eye[k - 1]]
    rows += [np.r_[np.ones(j), np.zeros(k - j)] for j in range(1, k + 1)]
    A, b = np.array(rows), np.r_[np.zeros(k), c]
    if k == 1:
        d, p = np.ones((1, 1)), np.zeros((1, 1))
    else:
        active = np.array(list(combinations(range(len(A)), k - 1)))
        _, s, vt = np.linalg.svd(A[active])
        full = np.sum(s > 1e-12, axis=1) == k - 1
        d = vt[full, -1, :]
        p = np.einsum("mij,mj->mi", np.linalg.pinv(A[active[full]]), b[active[full]])
    Ad, slack = d @ A.T, b - p @ A.T
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = slack / Ad
    lo = np.max(np.where(Ad < -1e-12, ratio, -np.inf), axis=1)
    hi = np.min(np.where(Ad > 1e-12, ratio, np.inf), axis=1)
    keep = ~np.any((np.abs(Ad) <= 1e-12) & (slack < -1e-12), axis=1)
    keep &= np.isfinite(lo) & np.isfinite(hi) & (lo <= hi + 1e-12)
    ends = p[keep] + lo[keep, None] * d[keep], p[keep] + np.maximum(lo, hi)[keep, None] * d[keep]
    return list(zip(*ends, strict=True))


def _with_least_tail(heads: np.ndarray) -> np.ndarray:
    """Entropy of each row of ``heads`` followed by its least-entropy tail; ``inf`` where no tail fits."""
    x = np.clip(heads[:, -1], 0.0, None)
    m = np.clip(1.0 - heads.sum(axis=1), 0.0, None)
    with np.errstate(divide="ignore", invalid="ignore"):
        q = np.where(x > 0, np.floor(m / np.where(x > 0, x, 1.0)), 0.0)
    value = _phi(heads).sum(axis=1) + q * _phi(x) + _phi(m - q * x)
    return np.where((x > 0) | (m <= 1e-12), value, np.inf)


def _remainder_free_heads(p0: np.ndarray, p1: np.ndarray, q_min: int, q_max: int) -> np.ndarray:
    """Points of the edge ``[p0, p1]`` where the mass ``1 - S_k`` is ``q`` copies of ``h_k``, ``q_min <= q <= q_max``."""
    x0, x1, m0, m1 = p0[-1], p1[-1], 1.0 - p0.sum(), 1.0 - p1.sum()
    g = [m / x if x > 1e-12 else np.inf for m, x in ((m0, x0), (m1, x1))]
    if min(g) > q_max:
        return np.empty((0, len(p0)))
    lo = max(q_min, math.ceil(min(g) - 1e-9))
    hi = q_max if max(g) > q_max else math.floor(max(g) + 1e-9)
    qs = np.arange(lo, hi + 1, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = (qs * x0 - m0) / ((m1 - m0) - qs * (x1 - x0))
    t = np.clip(t[np.isfinite(t) & (t >= -1e-12) & (t <= 1.0 + 1e-12)], 0.0, 1.0)
    return p0 + t[:, None] * (p1 - p0)


def _lighter_heads(c: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``(sum phi(h) + phi(r), r)`` with ``r = 1 - S_{k-1}(h)`` at the vertices ``h`` of the polytope of the ``k - 1``
    heaviest weights under ``c[:-1]``.

    When more than ``q`` tail weights equal the ``k``-th heaviest, every weight from the ``k``-th on is at most
    ``r / (q + 1)``, so the entropy is at least ``sum_{i<k} phi(h_i) + phi(r) + r log2(q + 1)``, which is concave in
    those weights and so at least its least value at these vertices.
    """
    if len(c) == 1:
        return np.zeros(1), np.ones(1)
    vertices = np.array([p for edge in _head_edges(c[:-1]) for p in edge])
    r = np.clip(1.0 - vertices.sum(axis=1), 0.0, 1.0)
    return _phi(vertices).sum(axis=1) + _phi(r), r


def _softmax(z: np.ndarray) -> np.ndarray:
    z = np.exp(z - z.max())
    return z / z.sum()


@dataclass(frozen=True)
class MuKResult:
    """Upper cap on ``mu_k(M)``; ``certified`` means ``value`` is a valid global upper bound."""

    value: float
    certified: bool
    k: int
    feasible: float | None = None


def lp_rank_k_weights(M: np.ndarray, U: np.ndarray, V: np.ndarray) -> float:
    r"""Maximum total weight of :math:`k` rank-1 terms with shapes ``U[i]``, ``V[i]`` on the simplices.

    Exact linear program for fixed factor directions (cpfci ``coverage_bound.lp_weights``).
    """
    M = np.asarray(M, dtype=float)
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    k = len(U)
    if k == 0:
        return 0.0
    A = np.column_stack([np.outer(U[i], V[i]).ravel() for i in range(k)])
    res = linprog(-np.ones(k), A_ub=A, b_ub=M.ravel(), bounds=[(0.0, None)] * k, method="highs")
    if not res.success:
        return 0.0
    return float(-res.fun)


def _softmax_rows(z: np.ndarray, k: int, na: int, nb: int) -> tuple[np.ndarray, np.ndarray]:
    z = np.asarray(z, float)
    U = np.vstack([_softmax(z[i * na : (i + 1) * na]) for i in range(k)])
    V = np.vstack([_softmax(z[k * na + i * nb : k * na + (i + 1) * nb]) for i in range(k)])
    return U, V


def mu_k_lp_simplex(
    M: np.ndarray,
    k: int,
    *,
    starts: int = 80,
    seed: int = 0,
    seeds: tuple[np.ndarray, ...] = (),
) -> MuKResult:
    """Heuristic ``mu_k`` via multistart simplex factors and inner :func:`lp_rank_k_weights`."""
    M = np.asarray(M, dtype=float)
    na, nb = M.shape
    n = k * (na + nb)
    rng = np.random.default_rng(seed)
    best = 0.0

    def neg(z: np.ndarray) -> float:
        U, V = _softmax_rows(z, k, na, nb)
        return -lp_rank_k_weights(M, U, V)

    inits = list(seeds) + [rng.normal(scale=2.0, size=n) for _ in range(starts)]
    for z0 in inits:
        r = minimize(neg, z0, method="Nelder-Mead", options={"maxiter": 2500, "xatol": 1e-9, "fatol": 1e-12})
        best = max(best, -neg(r.x))
    return MuKResult(min(1.0, best), False, k, feasible=best)


def mu_k_certified_trivial_cap(k: int, mu1: float) -> float:
    r"""Valid upper bound ``mu_k <= min(1, k * mu_1)`` (each term weight at most ``mu_1``)."""
    if k <= 0:
        return 0.0
    return min(1.0, k * float(mu1))


def mu_k_spectral_cap(M: np.ndarray, k: int) -> float:
    r"""Certified ``mu_k(M) <= mass(M) - sum_{i>k} sigma_i(M)``.

    If ``N`` is a sum of ``k`` rank-1 terms with ``0 <= N <= M``, the uncovered mass
    ``sum(M - N)`` is the entrywise :math:`\ell_1` norm of ``M - N``. That dominates the
    nuclear norm (every matrix of spectral norm at most one has entries at most one in absolute
    value), and the nuclear distance from ``M`` to rank-``k`` matrices is the singular-value tail
    (Eckart–Young–Mirsky, :cite:`mirsky1960symmetric`).
    """
    M = np.asarray(M, dtype=float)
    if k <= 0:
        return 0.0
    sv = np.linalg.svd(M, compute_uv=False)
    tail = float(sv[k:].sum())
    slack = 1e-12 * max(1.0, float(sv[0])) * len(sv)
    return float(M.sum()) - max(0.0, tail - slack)


def _cross(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.stack(
        [
            a[..., 1] * b[..., 2] - a[..., 2] * b[..., 1],
            a[..., 2] * b[..., 0] - a[..., 0] * b[..., 2],
            a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0],
        ],
        axis=-1,
    )


def _solve_square_systems(G: np.ndarray, h: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Solutions of ``G x = h`` over the leading axes, and which systems are regular (``|det G| > 1e-14``).

    The solutions of singular systems are arbitrary finite values. Systems of size three, the common case,
    are solved by Cramer's rule, which beats a LAPACK call per tiny matrix.
    """
    m = G.shape[-1]
    if m != 3:
        regular = np.abs(np.linalg.det(G)) > 1e-14
        return np.linalg.solve(np.where(regular[..., None, None], G, np.eye(m)), h[..., None])[..., 0], regular
    r0, r1, r2 = G[..., 0, :], G[..., 1, :], G[..., 2, :]
    c0, c1, c2 = _cross(r1, r2), _cross(r2, r0), _cross(r0, r1)
    det = (r0 * c0).sum(axis=-1)
    regular = np.abs(det) > 1e-14
    x = (h[..., 0, None] * c0 + h[..., 1, None] * c1 + h[..., 2, None] * c2) / np.where(regular, det, 1.0)[..., None]
    return x, regular


def _column_box_lp(U: np.ndarray, A: np.ndarray, C: np.ndarray) -> np.ndarray:
    """``max 1.x`` over ``0 <= x <= U[b]``, ``A[p].x <= 0``, ``C[p].x >= 0`` for every column ``b`` and pair ``p``.

    Vertex enumeration over the ``2m + 2`` constraints in ``R^m``; ``x = 0`` is always feasible.
    Returns an array of shape ``(len(U), len(A))``.
    """
    nb, m = U.shape
    npairs = A.shape[0]
    G = np.zeros((nb, npairs, 2 * m + 2, m))
    h = np.zeros((nb, npairs, 2 * m + 2))
    G[:, :, :m] = -np.eye(m)
    G[:, :, m : 2 * m] = np.eye(m)
    h[:, :, m : 2 * m] = U[:, None, :]
    G[:, :, 2 * m] = A[None]
    G[:, :, 2 * m + 1] = -C[None]
    combos = np.array(list(combinations(range(2 * m + 2), m)))
    x, regular = _solve_square_systems(G[:, :, combos], h[:, :, combos])
    slack = np.einsum("bpkm,bpcm->bpck", G, x) - h[:, :, None, :]
    feasible = regular & (slack.max(axis=-1) <= 1e-12 * (1.0 + float(U.max())))
    return np.where(feasible, x.sum(axis=-1), -np.inf).max(axis=-1)


def mu_hyperplane_cap(M: np.ndarray, *, max_nodes: int = 4000, tol: float = 1e-8) -> MuKResult:
    r"""Certified cap on ``mu_{m-1}``, ``m = min(M.shape)``, by branch and bound over hyperplanes.

    A sum ``N`` of ``m - 1`` rank-1 terms has rank at most ``m - 1``, so its columns lie in a
    hyperplane ``nu^T x = 0`` of ``R^m``, and for a fixed normal the columns are independent:
    ``mu_{m-1} <= max_nu sum_b max{1.x : 0 <= x <= M_b, nu.x = 0}``. For ``m = 3`` this is
    equality, since nonnegative matrices of rank at most two have nonnegative rank at most two
    :cite:`cohen1993nonnegative`. The normals are branched over spherical simplices (orthant cones,
    bisected along their longest edge); a column lies in the hyperplane of some normal in the cone
    of ``v_1, ..., v_m`` exactly when ``min_i v_i.x <= 0 <= max_j v_j.x``, which bounds each column
    by ``m^2`` small LPs. The value is the largest open bound, valid at any ``max_nodes``;
    ``feasible`` is the best normal found, exact for ``m = 3`` when the search converges.
    """
    M = np.asarray(M, dtype=float)
    U = M.T if M.shape[0] <= M.shape[1] else M
    m = U.shape[1]
    if m < 2:
        return MuKResult(float(M.sum()), True, max(m - 1, 0), feasible=float(M.sum()))

    def lower(nu: np.ndarray) -> float:
        nu = nu / np.linalg.norm(nu)
        return float(_column_box_lp(U, nu[None], nu[None]).sum())

    def upper(V: np.ndarray) -> float:
        pairs = list(product(range(m), repeat=2))
        A = np.array([V[i] for i, _ in pairs])
        C = np.array([V[j] for _, j in pairs])
        return float(_column_box_lp(U, A, C).max(axis=1).sum())

    best = 0.0
    heap: list[tuple[float, int, np.ndarray]] = []
    order = count()
    for signs in product((1.0, -1.0), repeat=m - 1):
        V = np.eye(m)
        V[1:] *= np.array(signs)[:, None]
        best = max(best, lower(V.mean(axis=0)))
        heapq.heappush(heap, (-upper(V), next(order), V))
    nodes = 0
    while heap and nodes < max_nodes and -heap[0][0] > best + tol:
        bound, _, V = heapq.heappop(heap)
        nodes += 1
        i, j = max(combinations(range(m), 2), key=lambda e: float(np.linalg.norm(V[e[0]] - V[e[1]])))
        mid = (V[i] + V[j]) / np.linalg.norm(V[i] + V[j])
        for side in (i, j):
            W = V.copy()
            W[side] = mid
            best = max(best, lower(W.mean(axis=0)))
            ub = min(upper(W), -bound)
            if ub > best + tol:
                heapq.heappush(heap, (-ub, next(order), W))
    value = max(best, -heap[0][0]) if heap else best
    return MuKResult(min(float(M.sum()), value + 1e-12), True, m - 1, feasible=best)


def mu_k_optimize(
    M: np.ndarray,
    k: int,
    *,
    seed: int = 0,
    mu1: Mu1Result | None = None,
) -> MuKResult:
    """Certified cap on ``mu_k``; the last entry of :func:`mu_caps`."""
    if k <= 0:
        return MuKResult(0.0, True, k)
    caps, certified = mu_caps(M, k=k, seed=seed, mu1=mu1)
    return MuKResult(caps[-1], certified, k)


def mu1_search(M: np.ndarray, *, seed: int = 0, maxiter: int = 300) -> float:
    """Heuristic upper estimate of ``mu_1`` via differential evolution on simplex parameters."""
    return mu1_softmax(M, seed=seed, maxiter=maxiter).value


def mu1_softmax(M: np.ndarray, *, seed: int = 0, maxiter: int = 300) -> Mu1Result:
    """Heuristic ``mu_1`` via differential evolution on softmax simplex parameters."""
    M = np.asarray(M, dtype=float)
    na, nb = M.shape

    def neg(z: np.ndarray) -> float:
        u = _softmax(z[:na])
        v = _softmax(z[na:])
        return -mu1_rank1_mass(M, u, v)

    bounds = [(-8.0, 8.0)] * (na + nb)
    r = differential_evolution(
        neg,
        bounds,
        seed=seed,
        polish=True,
        maxiter=maxiter,
        tol=1e-11,
        workers=1,
    )
    z = r.x
    u = _softmax(z[:na])
    v = _softmax(z[na:])
    return Mu1Result(-float(r.fun), False, u, v)


def _logsumexp(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    m = float(x.max())
    return m + float(np.log(np.exp(x - m).sum()))


_LOG_FLOOR = float(np.log(1e-300))
_VERTEX_PERTURBATION = 1e-7


def _tree_walk(tree: frozenset, L: np.ndarray) -> tuple[np.ndarray, np.ndarray, list[tuple[int, int]], list[int]]:
    """Solve ``p_a + q_b = L_ab`` on the edges of a spanning tree of ``K_{na,nb}`` with ``p_0 = 0``.

    Also returns the tree edges (in the tree's iteration order) and, for each edge ``(a, b)``, the bitmask
    of the nodes (rows ``0..na-1``, then columns) left on row ``a``'s side when the edge is removed.
    """
    na, nb = L.shape
    adj: list[list[int]] = [[] for _ in range(na + nb)]
    for a, b in tree:
        adj[a].append(na + b)
        adj[na + b].append(a)
    pot = [0.0] * (na + nb)
    parent = [-1] * (na + nb)
    parent[0] = 0
    order = []
    stack = [0]
    while stack:
        x = stack.pop()
        order.append(x)
        for y in adj[x]:
            if parent[y] < 0:
                parent[y] = x
                a, b = (x, y - na) if x < na else (y, x - na)
                pot[y] = L[a, b] - pot[x]
                stack.append(y)
    subtree = [1 << x for x in range(na + nb)]
    for x in reversed(order[1:]):
        subtree[parent[x]] |= subtree[x]
    full = (1 << (na + nb)) - 1
    edges, sides = [], []
    for a, b in tree:
        below = subtree[na + b] if parent[na + b] == a else subtree[a]
        edges.append((a, b))
        sides.append(below if below >> a & 1 else full ^ below)
    return np.array(pot[:na]), np.array(pot[na:]), edges, sides


def _initial_tree(L: np.ndarray) -> frozenset:
    """A spanning tree of tight cells at some vertex of ``{p_a + q_b <= L_ab}``."""
    na, nb = L.shape
    p = np.zeros(na)
    q = np.min(L, axis=0)
    p = np.min(L - q[None, :], axis=1)
    while True:
        slack = L - p[:, None] - q[None, :]
        tight = slack <= 1e-12 * (1.0 + np.abs(L))
        comp = _components(na, nb, zip(*np.nonzero(tight), strict=True))
        if len(set(comp)) == 1:
            break
        rows = np.array([comp[a] != comp[0] for a in range(na)])
        cols = np.array([comp[na + b] != comp[0] for b in range(nb)])
        p[rows] += slack[np.ix_(rows, ~cols)].min()
        q[cols] -= slack[np.ix_(rows, ~cols)].min()
    tree: set = set()
    parent = list(range(na + nb))
    for a, b in sorted(zip(*np.nonzero(tight), strict=True), key=lambda e: slack[e]):
        ra, rb = _find(parent, int(a)), _find(parent, na + int(b))
        if ra != rb:
            parent[ra] = rb
            tree.add((int(a), int(b)))
    return frozenset(tree)


def _find(parent: list[int], x: int) -> int:
    while parent[x] != x:
        parent[x] = parent[parent[x]]
        x = parent[x]
    return x


def _components(na: int, nb: int, edges: Iterable[tuple[int, int]]) -> list[int]:
    parent = list(range(na + nb))
    for a, b in edges:
        parent[_find(parent, int(a))] = _find(parent, na + int(b))
    return [_find(parent, x) for x in range(na + nb)]


def mu1_vertex(M: np.ndarray, *, max_vertices: int = 200_000) -> Mu1Result | None:
    r"""Certified ``mu_1`` by enumerating vertices of ``{(p, q) : p_a + q_b <= log M_ab}``.

    A rank-1 term ``x y^T <= M`` is ``x = exp(p)``, ``y = exp(q)`` with ``(p, q)`` in that polyhedron,
    and its mass ``sum_ab exp(p_a + q_b)`` is convex in ``(p, q)``, so the maximum sits at a vertex.
    Vertices are spanning trees of tight cells of ``K_{na,nb}``; they are visited by simplex pivots.
    Zero cells are floored at ``1e-300`` and ``log M`` is raised by at most ``1e-7`` to break ties;
    both only enlarge the polyhedron, so the value stays a valid upper bound.
    Returns ``None`` when more than ``max_vertices`` vertices would be visited.
    """
    M = np.asarray(M, dtype=float)
    na, nb = M.shape
    L = np.maximum(np.log(np.maximum(M, 1e-300)), _LOG_FLOOR)
    L = L + _VERTEX_PERTURBATION * np.random.default_rng(0).random((na, nb))
    start = _initial_tree(L)
    seen = {start}
    queue = [start]
    best, best_pq = -np.inf, None
    bits = 1 << np.arange(na + nb)
    while queue:
        tree = queue.pop()
        p, q, edges, sides = _tree_walk(tree, L)
        value = _logsumexp(p) + _logsumexp(q)
        if value > best:
            best, best_pq = value, (p, q)
        slack = L - p[:, None] - q[None, :]
        # For each tree edge: rows off and columns on row a's side once it is removed, and the entering
        # cell, the first slack minimum in row-major order over that rows x columns block.
        on_side = (np.array(sides)[:, None] & bits) != 0
        rows_out, cols_in = ~on_side[:, :na], on_side[:, na:]
        blocks = rows_out[:, :, None] & cols_in[:, None, :]
        entering_cells = np.where(blocks, slack, np.inf).reshape(len(edges), -1).argmin(axis=1)
        for edge, has_block, cell in zip(edges, blocks.any(axis=(1, 2)), entering_cells.tolist(), strict=True):
            if not has_block:
                continue
            entering = divmod(cell, nb)
            nxt = (tree - {edge}) | {entering}
            if nxt not in seen:
                if len(seen) >= max_vertices:
                    return None
                seen.add(nxt)
                queue.append(nxt)
    p, q = best_pq
    value = min(float(M.sum()), float(np.exp(best)))
    return Mu1Result(value, True, np.exp(p - _logsumexp(p)), np.exp(q - _logsumexp(q)))


def mu1_optimize(M: np.ndarray, *, seed: int = 0) -> Mu1Result:
    """Certified ``mu_1`` via :func:`mu1_vertex`; uncertified softmax search when enumeration is too large."""
    exact = mu1_vertex(M)
    if exact is not None:
        return exact
    return mu1_softmax(M, seed=seed)


def mu_caps(
    M: np.ndarray,
    *,
    k: int = 2,
    seed: int = 0,
    mu1: Mu1Result | None = None,
) -> tuple[tuple[float, ...], bool]:
    r"""Certified nondecreasing caps on ``mu_1, ..., mu_k`` for :func:`coverage_entropy_lower`.

    ``cap_j = min(mass, j mu_1, spectral_j, cap_i + cap_{j-i})`` with :func:`mu_k_spectral_cap`
    (``j mu_1`` only when ``mu_1`` is certified; splitting ``j`` terms into two groups gives the
    subadditive term), followed by a suffix minimum since ``mu_j`` is nondecreasing in ``j``.
    When ``min(M.shape) == 3``, ``cap_2`` also uses :func:`mu_hyperplane_cap`, which is exact there.
    Every term is a valid upper bound, so the caps are always certified.
    """
    M = np.asarray(M, dtype=float)
    mu1 = mu1 if mu1 is not None else mu1_optimize(M, seed=seed)
    mass = min(1.0, float(M.sum()))
    hyperplane = mu_hyperplane_cap(M).value if min(M.shape) == 3 and k >= 2 else None
    raw: list[float] = []
    for j in range(1, k + 1):
        c = min(mass, mu_k_spectral_cap(M, j))
        if mu1.certified:
            c = min(c, j * mu1.value)
        if j == 2 and hyperplane is not None:
            c = min(c, hyperplane)
        for i in range(1, j):
            c = min(c, raw[i - 1] + raw[j - i - 1])
        raw.append(c)
    caps = list(raw)
    for j in range(len(caps) - 2, -1, -1):
        caps[j] = min(caps[j], caps[j + 1])
    return tuple(caps), True


@dataclass(frozen=True)
class CoverageLowerResult:
    """Coverage entropy lower bound from ``mu_k`` caps."""

    value: float
    caps: tuple[float, ...]
    certified: bool
    mu1: float
    k_used: int


def coverage_bound(
    M: np.ndarray,
    *,
    k_max: int = 8,
    seed: int = 0,
) -> CoverageLowerResult:
    """Coverage lower bound from certified caps on ``mu_1, ..., mu_k``, ``k = min(k_max, na, nb)``.

    ``mu_k`` equals the total mass once ``k >= min(na, nb)`` (one rank-1 term per row), so larger
    ``k`` adds nothing.
    """
    M = np.asarray(M, dtype=float)
    mu1 = mu1_optimize(M, seed=seed)
    k_used = max(1, min(int(k_max), *M.shape))
    caps, certified = mu_caps(M, k=k_used, seed=seed, mu1=mu1)
    return CoverageLowerResult(
        coverage_entropy_lower(caps),
        caps,
        certified,
        mu1.value,
        k_used,
    )


@dataclass(frozen=True)
class Mu1Witness2:
    """Witness for ``mu_1`` on a 2×2 matrix (grid search, not interval-certified)."""

    value: float
    u0: float
    v0: float


def _mu1_at_2x2_float(M: np.ndarray, u0: float, v0: float) -> float:
    u = (u0, 1.0 - u0)
    v = (v0, 1.0 - v0)
    vals = []
    for a in range(2):
        for b in range(2):
            denom = u[a] * v[b]
            if denom <= 0.0:
                return 0.0
            vals.append(float(M[a, b]) / denom)
    return min(vals)


def mu1_brute_2x2(M: np.ndarray, *, grid: int = 4000) -> Mu1Witness2:
    """Exhaustive grid on ``(u0, v0)`` for a 2×2 positive matrix (fast, high precision)."""
    M = np.asarray(M, dtype=float)
    best = Mu1Witness2(0.0, 0.5, 0.5)
    u0s = np.linspace(1.0 / grid, 1.0 - 1.0 / grid, grid)
    for u0 in u0s:
        u1 = 1.0 - u0
        for v0 in u0s:
            v1 = 1.0 - v0
            val = min(
                M[0, 0] / (u0 * v0),
                M[0, 1] / (u0 * v1),
                M[1, 0] / (u1 * v0),
                M[1, 1] / (u1 * v1),
            )
            if val > best.value:
                best = Mu1Witness2(float(val), float(u0), float(v0))
    return best


def mu1_product_lower(M: np.ndarray, witness: Mu1Witness2) -> float:
    """Feasible rank-1 mass on ``kron(M, M)`` from a 2×2 witness (valid lower bound on ``mu_1`` of the product)."""
    u = np.array([witness.u0, 1.0 - witness.u0])
    v = np.array([witness.v0, 1.0 - witness.v0])
    u2 = np.kron(u, u)
    v2 = np.kron(v, v)
    big = np.kron(M, M)
    return float(np.min(big / np.outer(u2, v2)))


def mu_k_bilinear(M: np.ndarray, k: int, *, starts: int = 80, seeds: tuple[np.ndarray, ...] = ()) -> float:
    """Heuristic upper estimate of ``mu_k`` via separable rank-``k`` factors (SLSQP, multistart)."""
    M = np.asarray(M, dtype=float)
    na, nb = M.shape
    n = k * (na + nb)
    rng = np.random.default_rng(0)

    def split(z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return z[: k * na].reshape(k, na), z[k * na :].reshape(k, nb)

    def total(z: np.ndarray) -> float:
        A, B = split(z)
        return -float((A.sum(1) * B.sum(1)).sum())

    def gtotal(z: np.ndarray) -> np.ndarray:
        A, B = split(z)
        return -np.concatenate(
            [(B.sum(1)[:, None] * np.ones((k, na))).ravel(), (A.sum(1)[:, None] * np.ones((k, nb))).ravel()]
        )

    def slack(z: np.ndarray) -> np.ndarray:
        A, B = split(z)
        return (M - np.einsum("ia,ib->ab", A, B)).ravel()

    def gslack(z: np.ndarray) -> np.ndarray:
        A, B = split(z)
        J = np.zeros((na, nb, n))
        for i in range(k):
            for a in range(na):
                J[a, :, i * na + a] = -B[i]
            for b in range(nb):
                J[:, b, k * na + i * nb + b] = -A[i]
        return J.reshape(na * nb, n)

    best = 0.0
    inits = list(seeds) + [rng.random(n) * 0.5 for _ in range(starts)]
    for z0 in inits:
        r = minimize(
            total,
            z0,
            jac=gtotal,
            method="SLSQP",
            bounds=[(0.0, None)] * n,
            constraints=[{"type": "ineq", "fun": slack, "jac": gslack}],
            options={"maxiter": 800, "ftol": 1e-12},
        )
        if np.min(slack(r.x)) > -1e-9:
            best = max(best, -float(r.fun))
    return best


__all__ = [
    "CoverageLowerResult",
    "Mu1Result",
    "Mu1Witness2",
    "MuKResult",
    "coverage_bound",
    "coverage_entropy_lower",
    "joint_matrix",
    "lp_rank_k_weights",
    "mu1_brute_2x2",
    "mu1_optimize",
    "mu1_product_lower",
    "mu1_rank1_mass",
    "mu1_search",
    "mu1_softmax",
    "mu1_vertex",
    "mu_caps",
    "mu_k_bilinear",
    "mu_k_certified_trivial_cap",
    "mu_k_lp_simplex",
    "mu_k_optimize",
    "mu_hyperplane_cap",
    "mu_k_spectral_cap",
]
