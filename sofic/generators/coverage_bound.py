r"""Partial-sum (coverage) lower bounds on exact common information from rank-1 decompositions.

For :math:`M = p(s^+, s^-)` and an exact decomposition into nonnegative rank-1 terms with weights
:math:`\lambda_w`, the :math:`k` largest weights sum to at most :math:`\mu_k(M)` — the maximum total
mass of :math:`k` rank-1 terms that fit entrywise under :math:`M`. Entropy is Schur-concave, so
:math:`\operatorname{G}(S^+ : S^-) \ge \operatorname{H}(\mu_1, \mu_2 - \mu_1, \ldots)` for any
certified **upper** bounds on the :math:`\mu_k`.
"""

from __future__ import annotations

import heapq
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
    """Entropy lower bound from nondecreasing upper caps ``mu_k`` on partial sums (bits)."""
    caps = list(mu_caps)
    if not caps:
        return 0.0
    inc = [min(1.0, max(0.0, caps[0]))]
    for k in range(1, len(caps)):
        inc.append(min(1.0, max(0.0, caps[k] - caps[k - 1])))
    tail = max(0.0, 1.0 - caps[-1])
    if tail > 0:
        inc.append(tail)
    inc = [x for x in inc if x > 0]
    inc = np.asarray(inc, float)
    inc = inc / inc.sum()
    with np.errstate(divide="ignore", invalid="ignore"):
        logp = np.log2(inc)
    logp[~np.isfinite(logp)] = 0.0
    return float(-(inc * logp).sum())


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
    Gs, hs = G[:, :, combos], h[:, :, combos]
    regular = np.abs(np.linalg.det(Gs)) > 1e-14
    x = np.linalg.solve(np.where(regular[..., None, None], Gs, np.eye(m)), hs[..., None])[..., 0]
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


def _tree_potentials(tree: frozenset, L: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Solve ``p_a + q_b = L_ab`` on the edges of a spanning tree of ``K_{na,nb}`` with ``p_0 = 0``."""
    na, nb = L.shape
    adj: list[list[int]] = [[] for _ in range(na + nb)]
    for a, b in tree:
        adj[a].append(na + b)
        adj[na + b].append(a)
    pot = np.full(na + nb, np.nan)
    pot[0] = 0.0
    stack = [0]
    while stack:
        x = stack.pop()
        for y in adj[x]:
            if np.isnan(pot[y]):
                a, b = (x, y - na) if x < na else (y, x - na)
                pot[y] = L[a, b] - pot[x]
                stack.append(y)
    return pot[:na], pot[na:]


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
    while queue:
        tree = queue.pop()
        p, q = _tree_potentials(tree, L)
        value = _logsumexp(p) + _logsumexp(q)
        if value > best:
            best, best_pq = value, (p, q)
        slack = L - p[:, None] - q[None, :]
        for edge in tree:
            comp = _components(na, nb, tree - {edge})
            side = comp[edge[0]]
            rows_out = np.array([comp[a] != side for a in range(na)])
            cols_in = np.array([comp[na + b] == side for b in range(nb)])
            if not rows_out.any() or not cols_in.any():
                continue
            sub = slack[np.ix_(rows_out, cols_in)]
            i, j = np.unravel_index(int(np.argmin(sub)), sub.shape)
            entering = (int(np.flatnonzero(rows_out)[i]), int(np.flatnonzero(cols_in)[j]))
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
