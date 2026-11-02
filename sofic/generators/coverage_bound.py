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
from scipy.optimize import differential_evolution, minimize


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


def mu1_search(M: np.ndarray, *, seed: int = 0, maxiter: int = 300) -> float:
    """Heuristic upper estimate of ``mu_1`` via differential evolution on simplex parameters."""
    M = np.asarray(M, dtype=float)
    na, nb = M.shape

    def neg(z: np.ndarray) -> float:
        u = _softmax(z[:na])
        v = _softmax(z[na:])
        return -float(np.min(M / np.outer(u, v)))

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
    return -float(r.fun)


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
    "Mu1Witness2",
    "coverage_entropy_lower",
    "joint_matrix",
    "mu1_brute_2x2",
    "mu1_product_lower",
    "mu1_search",
    "mu_k_bilinear",
]
