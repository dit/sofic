r"""Partial-sum (coverage) lower bounds on exact common information from rank-1 decompositions.

For :math:`M = p(s^+, s^-)` and an exact decomposition into nonnegative rank-1 terms with weights
:math:`\lambda_w`, the :math:`k` largest weights sum to at most :math:`\mu_k(M)` — the maximum total
mass of :math:`k` rank-1 terms that fit entrywise under :math:`M`. Entropy is Schur-concave, so
:math:`\operatorname{G}(S^+ : S^-) \ge \operatorname{H}(\mu_1, \mu_2 - \mu_1, \ldots)` for any
certified **upper** bounds on the :math:`\mu_k`.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

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


def mu_k_optimize(
    M: np.ndarray,
    k: int,
    *,
    seed: int = 0,
    starts: int = 80,
    mu1: Mu1Result | None = None,
) -> MuKResult:
    """Best-effort ``mu_k`` cap: certified ``k * mu_1`` when ``mu_1`` is certified, else LP / bilinear search."""
    if k <= 0:
        return MuKResult(0.0, True, k)
    if k == 1:
        r = mu1 if mu1 is not None else mu1_optimize(M, seed=seed)
        return MuKResult(r.value, r.certified, 1, feasible=r.value)
    mu1_res = mu1 if mu1 is not None else mu1_optimize(M, seed=seed)
    trivial = mu_k_certified_trivial_cap(k, mu1_res.value)
    if mu1_res.certified:
        return MuKResult(trivial, True, k)
    lp = mu_k_lp_simplex(M, k, starts=starts, seed=seed)
    bil = mu_k_bilinear(M, k, starts=min(starts, 120))
    feasible = max(lp.feasible or 0.0, bil)
    est = max(feasible, lp.value)
    return MuKResult(min(1.0, est), False, k, feasible=feasible)


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


def mu1_convex_full_support(M: np.ndarray, *, starts: int = 24, rng: np.random.Generator | None = None) -> Mu1Result:
    r"""Certified ``mu_1`` when ``M`` is entrywise positive (full support).

    Solves ``min_{alpha,beta} max_{a,b} (alpha_a + beta_b - log M_{ab})`` with
    ``logsumexp(alpha) = logsumexp(beta) = 0`` (simplex equality in log coordinates).
    Then ``mu_1 = exp(-optimum)``. Convex in ``(alpha, beta)`` for fixed support.
    """
    M = np.asarray(M, dtype=float)
    if np.any(M <= 0):
        raise ValueError("mu1_convex_full_support requires entrywise positive M")
    na, nb = M.shape
    L = np.log(M)
    rng = rng or np.random.default_rng(0)

    def objective(z: np.ndarray) -> float:
        alpha, beta = z[:na], z[na:]
        margin = float(np.max(alpha[:, None] + beta[None, :] - L))
        penalty = (_logsumexp(alpha) ** 2 + _logsumexp(beta) ** 2) * 1e6
        return margin + penalty

    best = np.inf
    best_z = None
    for _ in range(starts):
        z0 = np.concatenate([np.log(rng.dirichlet(np.ones(na))), np.log(rng.dirichlet(np.ones(nb)))])
        r = minimize(objective, z0, method="L-BFGS-B", options={"maxiter": 2000, "ftol": 1e-14})
        if r.success and r.fun < best:
            best, best_z = r.fun, r.x
    if best_z is None or not np.isfinite(best):
        return mu1_softmax(M, seed=int(rng.integers(0, 2**31)))
    alpha, beta = best_z[:na], best_z[na:]
    u = np.exp(alpha - _logsumexp(alpha))
    v = np.exp(beta - _logsumexp(beta))
    value = float(np.exp(-best))
    return Mu1Result(value, True, u, v)


def mu1_optimize(M: np.ndarray, *, seed: int = 0, grid: int = 6000) -> Mu1Result:
    """Best available ``mu_1`` upper bound: exact convex step on full support, else 2×2 brute or softmax."""
    M = np.asarray(M, dtype=float)
    na, nb = M.shape
    if na == 2 and nb == 2 and np.all(M > 0):
        w = mu1_brute_2x2(M, grid=grid)
        u = np.array([w.u0, 1.0 - w.u0])
        v = np.array([w.v0, 1.0 - w.v0])
        return Mu1Result(w.value, True, u, v)
    if np.all(M > 0):
        return mu1_convex_full_support(M, rng=np.random.default_rng(seed))
    return mu1_softmax(M, seed=seed)


def mu_caps(M: np.ndarray, *, k: int = 2, seed: int = 0, mu_starts: int = 80) -> tuple[tuple[float, ...], bool]:
    """Nondecreasing ``mu_k`` caps for :func:`coverage_entropy_lower`; ``certified`` if every term is certified."""
    mu1 = mu1_optimize(M, seed=seed)
    caps: list[float] = [mu1.value]
    certified = mu1.certified
    for j in range(2, k + 1):
        mk = mu_k_optimize(M, j, seed=seed, starts=mu_starts, mu1=mu1)
        caps.append(min(1.0, max(caps[-1], mk.value)))
        certified = certified and mk.certified
    return tuple(caps), certified


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
    "Mu1Result",
    "Mu1Witness2",
    "MuKResult",
    "coverage_entropy_lower",
    "joint_matrix",
    "lp_rank_k_weights",
    "mu1_brute_2x2",
    "mu1_convex_full_support",
    "mu1_optimize",
    "mu1_product_lower",
    "mu1_rank1_mass",
    "mu1_search",
    "mu1_softmax",
    "mu_caps",
    "mu_k_bilinear",
    "mu_k_certified_trivial_cap",
    "mu_k_lp_simplex",
    "mu_k_optimize",
]
