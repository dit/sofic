r"""Minimal-state-entropy generators without the lifting restriction.

A generator of a stationary process with :math:`k` hidden states is described,
jointly with the forward and reverse causal states, by a state distribution
:math:`\pi`, the state's beliefs :math:`Q_w(s^+) = \Pr(S^+ = s^+ \mid W = w)`
and :math:`R_w(s^-) = \Pr(S^- = s^- \mid W = w)`, and a kernel
:math:`K(x, w' \mid w)`. Its Markov property makes the joint law of
:math:`(S^+_0, W_0, X_0, W_1, S^-_1)` the product
:math:`\pi_w Q_w(s^+_0) K(x_0, w_1 \mid w_0) R_{w_1}(s^-_1)`, which must

* reproduce the edge law :math:`\Pr(S^+_0, X_0, S^-_1)` of the bidirectional
  ε-machine (Ellison, Mahoney & Crutchfield :cite:`Ellison2009`);
* be stationary on the past side: :math:`\pi_{w'} Q_{w'}(s^+_1) =
  \sum \pi_w Q_w(s^+_0) K(x_0, w' \mid w)` over :math:`s^+_1 = f(s^+_0, x_0)`;
* be consistent on the future side: :math:`R_w(s^-_0) =
  \sum K(x_0, w' \mid w) R_{w'}(s^-_1)` over :math:`s^-_0 = g(x_0, s^-_1)`.

Unlike the lifted search of
:mod:`~sofic.generators.lifted_common_information`, the kernel :math:`K` is free,
so consecutive states may share randomness. :func:`coupled_generative_model`
minimizes :math:`\H{\pi}` subject to these equations (penalty continuation,
analytic gradients) from the ε-machine, reverse ε-machine, bidirectional
machine, optional extra starts (e.g. a lifted optimum) and random starts, and
checks every candidate against the source process's word probabilities. The
result bounds the generative complexity :math:`C_g` of Löhr & Ay
:cite:`LohrAy2009` from above for the given number of states.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

import numpy as np
from scipy.optimize import minimize

from sofic.exceptions import StochasticValidationError
from sofic.generators.lifted_common_information import _LiftedProblem
from sofic.generators.mealy import MealyHMM
from sofic.generators.minimal_generative_model import _normalized_joint_distribution
from sofic.graph import ATTR_EMISSION, ATTR_PROB, TransitionGraph

if TYPE_CHECKING:
    from sofic.generators.bidirectional_epsilon_machine import BidirectionalEpsilonMachine

_LN2 = np.log(2.0)
_ALM_MU = 1e3
_ALM_MU_MAX = 1e7
_ALM_ROUNDS = 60
_ALM_TOL = 1e-9


class _GeneratorProblem:
    """Edge law and causal-state maps of a bidirectional ε-machine, for k-state generator equations."""

    def __init__(self, bidir: BidirectionalEpsilonMachine, *, cutoff: float = 0.0) -> None:
        joint = _normalized_joint_distribution(bidir, cutoff=cutoff)
        lifted = _LiftedProblem(bidir, joint)
        self.bidir = bidir
        self.lifted = lifted
        self.symbols = sorted(bidir.observation_alphabet, key=repr)
        plus, minus, edges = lifted.plus, lifted.minus, lifted.edges
        # Pe[a, x, b'] = Pr(S+_0 = a, X_0 = x, S-_1 = b')
        self.Pe = np.einsum("pa,pxq,qb->axb", plus, edges, minus)
        # F[a, x, a'] = 1 when s+_1 = a' follows s+_0 = a on x; G[x, b', b] = 1 when s-_0 = b precedes s-_1 = b'.
        self.F = (np.einsum("pa,pxq,qc->axc", plus, edges, plus) > 0).astype(float)
        self.G = (np.einsum("pb,pxq,qc->xcb", minus, edges, minus) > 0).astype(float)
        self.p_pair = lifted.p
        self.p_plus = plus.T @ lifted.p
        self.p_minus = minus.T @ lifted.p
        self.n_plus, self.n_minus, self.n_sym = plus.shape[1], minus.shape[1], len(self.symbols)

    # ---- parametrization ---------------------------------------------------------------------------
    def sizes(self, k: int) -> list[tuple[int, ...]]:
        return [(k,), (k, self.n_plus), (k, self.n_minus), (k, self.n_sym * k)]

    def unpack(self, theta: np.ndarray, k: int) -> tuple[np.ndarray, ...]:
        parts, i = [], 0
        for shape in self.sizes(k):
            n = int(np.prod(shape))
            z = theta[i : i + n].reshape(shape)
            z = np.exp(z - z.max(axis=-1, keepdims=True))
            parts.append(z / z.sum(axis=-1, keepdims=True))
            i += n
        pi, Q, R, K = parts
        return pi, Q, R, K.reshape(k, self.n_sym, k)

    @staticmethod
    def logits(pi: np.ndarray, Q: np.ndarray, R: np.ndarray, K: np.ndarray, floor: float = 1e-9) -> np.ndarray:
        return np.concatenate([np.log(np.maximum(a, floor)).ravel() for a in (pi, Q, R, K.reshape(len(pi), -1))])

    # ---- residuals ------------------------------------------------------------------------------------
    def residuals(self, pi, Q, R, K) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        A = pi[:, None] * Q
        M = np.einsum("wa,wxv->axv", A, K)
        rE = np.einsum("axv,vb->axb", M, R) - self.Pe
        rF = np.einsum("axv,axc->vc", M, self.F) - A
        N = np.einsum("wxv,vc,xcb->wb", K, R, self.G)
        rB = pi[:, None] * (N - R)
        return rE, rF, rB

    def residual(self, pi, Q, R, K) -> float:
        return float(max(np.abs(r).max() for r in self.residuals(pi, Q, R, K)))

    def loss_and_grad(
        self, theta: np.ndarray, k: int, mu: float, shifts: tuple[np.ndarray, ...] | None = None, weight: float = 1.0
    ) -> tuple[float, np.ndarray]:
        """``weight * H[pi] + mu/2 * |r + shift|^2`` and its gradient (augmented Lagrangian when shifts = lambda/mu)."""
        pi, Q, R, K = self.unpack(theta, k)
        A = pi[:, None] * Q
        M = np.einsum("wa,wxv->axv", A, K)
        rE = np.einsum("axv,vb->axb", M, R) - self.Pe
        rF = np.einsum("axv,axc->vc", M, self.F) - A
        N = np.einsum("wxv,vc,xcb->wb", K, R, self.G)
        D = N - R
        rB = pi[:, None] * D
        if shifts is not None:
            rE, rF, rB = rE + shifts[0], rF + shifts[1], rB + shifts[2]
        loss = 0.5 * float((rE**2).sum() + (rF**2).sum() + (rB**2).sum())

        # Gradients of the penalty in (pi, Q, R, K).
        gM = np.einsum("axb,vb->axv", rE, R) + np.einsum("vc,axc->axv", rF, self.F)
        gR = np.einsum("axb,axv->vb", rE, M)
        gA = np.einsum("wxv,axv->wa", K, gM) - rF
        gK = np.einsum("wa,axv->wxv", A, gM)
        gpi = (gA * Q).sum(axis=1) + (rB * D).sum(axis=1)
        gQ = gA * pi[:, None]
        wB = pi[:, None] * rB
        gK += np.einsum("wb,vc,xcb->wxv", wB, R, self.G)
        gR += np.einsum("wb,wxv,xcb->vc", wB, K, self.G) - wB

        # State entropy.
        logp = np.log2(np.maximum(pi, 1e-300))
        h = float(-(pi * logp).sum())
        gpi_total = -weight * (logp + 1.0 / _LN2) + mu * gpi

        def through_softmax(p: np.ndarray, g: np.ndarray) -> np.ndarray:
            return p * (g - (g * p).sum(axis=-1, keepdims=True))

        grads = [
            through_softmax(pi, gpi_total),
            through_softmax(Q, mu * gQ),
            through_softmax(R, mu * gR),
            through_softmax(K.reshape(k, -1), mu * gK.reshape(k, -1)),
        ]
        return weight * h + mu * loss, np.concatenate([g.ravel() for g in grads])

    # ---- seeds ----------------------------------------------------------------------------------------
    def epsilon_seed(self) -> tuple[np.ndarray, ...]:
        """W = S+: the forward ε-machine as a generator."""
        pa = self.p_plus
        joint_ab = self.lifted.plus.T @ (self.p_pair[:, None] * self.lifted.minus)
        R = joint_ab / np.maximum(pa[:, None], 1e-300)
        px_a = self.Pe.sum(axis=2) / np.maximum(pa[:, None], 1e-300)
        K = px_a[:, :, None] * self.F
        return pa, np.eye(self.n_plus), R, K

    def reverse_seed(self) -> tuple[np.ndarray, ...]:
        """W = S-: the reverse ε-machine run forward in time as a generator."""
        pb = self.p_minus
        joint_ab = self.lifted.plus.T @ (self.p_pair[:, None] * self.lifted.minus)
        Q = (joint_ab / np.maximum(pb[None, :], 1e-300)).T
        # Pr(X_0 = x, S-_1 = b' | S-_0 = b) for g(x, b') = b
        K = np.einsum("axc,xcb->bxc", self.Pe, self.G) / np.maximum(pb[:, None, None], 1e-300)
        return pb, Q, np.eye(self.n_minus), K

    def pair_seed(self) -> tuple[np.ndarray, ...]:
        """W = (S+, S-): the bidirectional machine."""
        return self.channel_seed(np.eye(len(self.p_pair)))

    def channel_seed(self, c: np.ndarray) -> tuple[np.ndarray, ...]:
        """The lift of a channel ``c[pair, w]`` (e.g. a lifted or unconstrained optimum)."""
        lifted = self.lifted
        joint = lifted.p[:, None] * c
        pi = joint.sum(axis=0)
        safe = np.maximum(pi, 1e-300)
        Q = (lifted.plus.T @ joint).T / safe[:, None]
        R = (lifted.minus.T @ joint).T / safe[:, None]
        K = np.einsum("pw,pxq,qv->wxv", c, lifted.edges, c) / safe[:, None, None]
        return pi, Q, R, K


def _as_problem(bidir: BidirectionalEpsilonMachine | _GeneratorProblem) -> _GeneratorProblem:
    return bidir if isinstance(bidir, _GeneratorProblem) else _GeneratorProblem(bidir)


def memory_lift(bidir: BidirectionalEpsilonMachine | _GeneratorProblem, nu: np.ndarray) -> tuple[np.ndarray, ...]:
    r"""Generator equations' variables for the memory-1 lift of a channel ``nu[pair, v]``.

    Draw :math:`V_t \sim \nu(\cdot \mid B_t)` independently at each time given the bidirectional states
    :math:`B_t`, and take :math:`W_t = (V_t, V_{t+1})`; consecutive states share :math:`V_{t+1}`. With
    :math:`m` values of :math:`V` this gives :math:`m^2` states, ordered ``(v, v') -> v * m + v'``.
    ``nu`` is indexed by the causal-state pairs in sorted order. Returns ``(pi, Q, R, K)``.
    """
    problem = _as_problem(bidir)
    E = problem.lifted.edges  # E[p, x, q] = Pr(B_0 = p, X_0 = x, B_1 = q)
    p_pair = problem.lifted.p
    m = nu.shape[1]
    # Pr(W_0 = (v, v'), B_0 = p) and Pr(B_1 = q -> B_2 = r) transition probabilities
    joint_wp = np.einsum("pxq,pv,qu->vup", E, nu, nu).reshape(m * m, -1)
    pi = joint_wp.sum(axis=1)
    safe = np.maximum(pi, 1e-300)
    Q = joint_wp @ problem.lifted.plus / safe[:, None]
    R = joint_wp @ problem.lifted.minus / safe[:, None]
    step = E.sum(axis=1) / np.maximum(p_pair[:, None], 1e-300)  # Pr(B_2 = r | B_1 = q)
    # Pr(W_0 = (v, v'), X_0 = x, W_1 = (v', v''))
    flow = np.einsum("pxq,qr,pv,qu,rw->vuxw", E, step, nu, nu, nu)
    K = np.zeros((m * m, problem.n_sym, m * m))
    for v in range(m):
        for u in range(m):
            K[v * m + u, :, u * m : (u + 1) * m] = flow[v, u]
    K /= safe[:, None, None]
    return pi, Q, R, K


def memory_lift_search(
    bidir: BidirectionalEpsilonMachine,
    m: int,
    *,
    starts: int = 10,
    maxiter: int = 400,
    rng: np.random.Generator | None = None,
) -> list[tuple[float, float, np.ndarray]]:
    """Minimize the state entropy of memory-1 lifts with ``m`` values of V subject to the generator equations.

    A penalty continuation over the channel's logits (numerical gradients; meant for small problems).
    Returns ``(entropy, residual, nu)`` for every start, best first.
    """
    rng = rng or np.random.default_rng()
    problem = _GeneratorProblem(bidir)
    n = len(problem.lifted.p)

    def channel(z: np.ndarray) -> np.ndarray:
        z = z.reshape(n, m)
        z = np.exp(z - z.max(axis=1, keepdims=True))
        return z / z.sum(axis=1, keepdims=True)

    def value(z: np.ndarray, mu: float) -> float:
        pi, Q, R, K = memory_lift(problem, channel(z))
        p = pi[pi > 1e-15]
        penalty = sum(float((r**2).sum()) for r in problem.residuals(pi, Q, R, K))
        return float(-(p * np.log2(p)).sum()) + mu * penalty

    out = []
    for _ in range(starts):
        z = rng.normal(scale=2.0, size=n * m)
        for mu in (1e2, 1e4, 1e6, 1e8):
            z = minimize(value, z, args=(mu,), method="L-BFGS-B", options={"maxiter": maxiter}).x
        nu = channel(z)
        pi, Q, R, K = memory_lift(problem, nu)
        p = pi[pi > 1e-15]
        out.append((float(-(p * np.log2(p)).sum()), problem.residual(pi, Q, R, K), nu))
    return sorted(out, key=lambda t: (t[1] > 1e-6, t[0]))


def generator_kernel(bidir: BidirectionalEpsilonMachine | _GeneratorProblem, c: np.ndarray) -> tuple[float, np.ndarray]:
    """Best kernel for a fixed channel ``c[pair, w]``: the generator equations are linear in ``K``.

    Solves the linear program minimizing the largest violation ``t`` over stochastic kernels; ``t = 0`` means
    the channel's variable is the state of a generator, whatever the coupling. ``c`` is indexed by the
    causal-state pairs in sorted order. Returns ``(t, K)``.
    """
    from scipy.optimize import linprog

    problem = _as_problem(bidir)

    pi, Q, R, _ = problem.channel_seed(c)
    k, X = len(pi), problem.n_sym
    A = pi[:, None] * Q
    nK = k * X * k
    idx = lambda w, x, v: (w * X + x) * k + v  # noqa: E731
    rows, rhs = [], []

    def add(terms, target: float) -> None:
        row = np.zeros(nK)
        for j, val in terms:
            row[j] += val
        rows.append(row)
        rhs.append(target)

    for a in range(problem.n_plus):  # edge law
        for x in range(X):
            for b in range(problem.n_minus):
                add(((idx(w, x, v), A[w, a] * R[v, b]) for w in range(k) for v in range(k)), problem.Pe[a, x, b])
    for v in range(k):  # past-side stationarity
        for c_ in range(problem.n_plus):
            add(
                (
                    (idx(w, x, v), A[w, a] * problem.F[a, x, c_])
                    for w in range(k)
                    for x in range(X)
                    for a in range(problem.n_plus)
                ),
                A[v, c_],
            )
    for w in range(k):  # future-side consistency
        for b in range(problem.n_minus):
            add(
                (
                    (idx(w, x, v), pi[w] * R[v, c_] * problem.G[x, c_, b])
                    for x in range(X)
                    for v in range(k)
                    for c_ in range(problem.n_minus)
                ),
                pi[w] * R[w, b],
            )
    m = len(rows)
    dense = np.array(rows)
    target = np.array(rhs)
    # |dense K - rhs| <= t ; rows of K sum to one; K >= 0
    A_ub = np.block([[dense, -np.ones((m, 1))], [-dense, -np.ones((m, 1))]])
    b_ub = np.concatenate([target, -target])
    A_eq = np.zeros((k, nK + 1))
    for w in range(k):
        A_eq[w, w * X * k : (w + 1) * X * k] = 1.0
    cost = np.zeros(nK + 1)
    cost[-1] = 1.0
    res = linprog(cost, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=np.ones(k), bounds=[(0, None)] * (nK + 1), method="highs")
    if not res.success:
        return np.inf, np.zeros((k, X, k))
    return float(res.x[-1]), res.x[:-1].reshape(k, X, k)


def polish_separating_channel(
    bidir: BidirectionalEpsilonMachine | _GeneratorProblem, c: np.ndarray, *, maxiter: int = 500
) -> np.ndarray:
    """Re-solve a separating channel's decomposition ``p(s+, s-) = sum_w lam_w Q_w(s+) R_w(s-)`` exactly.

    Starts from the channel's own ``(lam, Q, R)`` and minimizes ``H[lam]`` with SLSQP under the exact
    decomposition constraints (unconstrained optimizers such as dit's satisfy them only to a tolerance, which
    is enough to make an otherwise valid generator state fail :func:`generator_kernel`). Returns the polished
    channel ``c'[pair, w] = lam_w Q_w(s+) R_w(s-) / p(s+, s-)``.
    """
    problem = _as_problem(bidir)
    lifted = problem.lifted
    P = lifted.plus.T @ (lifted.p[:, None] * lifted.minus)  # p(s+, s-)
    pi, Q, R, _ = problem.channel_seed(c)
    keep = pi > 1e-12
    lam0, Q0, R0 = pi[keep], Q[keep], R[keep]
    k, na, nb = len(lam0), P.shape[0], P.shape[1]
    n = k + k * na + k * nb

    def split(x):
        return x[:k], x[k : k + k * na].reshape(k, na), x[k + k * na :].reshape(k, nb)

    def H(x):
        lam = np.maximum(x[:k], 1e-300)
        return float(-(lam * np.log2(lam)).sum())

    def dH(x):
        g = np.zeros(n)
        g[:k] = -(np.log2(np.maximum(x[:k], 1e-15)) + 1 / _LN2)
        return g

    def dec(x):
        lam, A, B = split(x)
        return (np.einsum("w,wa,wb->ab", lam, A, B) - P).ravel()[:-1]

    def ddec(x):
        lam, A, B = split(x)
        J = np.zeros((na, nb, n))
        J[:, :, :k] = np.einsum("wa,wb->abw", A, B)
        for w in range(k):
            for a in range(na):
                J[a, :, k + w * na + a] = lam[w] * B[w]
            for b in range(nb):
                J[:, b, k + k * na + w * nb + b] = lam[w] * A[w]
        return J.reshape(na * nb, n)[:-1]

    S = np.zeros((1 + 2 * k, n))
    S[0, :k] = 1
    for w in range(k):
        S[1 + w, k + w * na : k + (w + 1) * na] = 1
        S[1 + k + w, k + k * na + w * nb : k + k * na + (w + 1) * nb] = 1
    x0 = np.concatenate([lam0, Q0.ravel(), R0.ravel()])
    res = minimize(
        H,
        x0,
        jac=dH,
        method="SLSQP",
        bounds=[(0, 1)] * n,
        constraints=[
            {"type": "eq", "fun": dec, "jac": ddec},
            {"type": "eq", "fun": lambda x: S @ x - np.concatenate([[1.0], np.ones(2 * k)]), "jac": lambda x: S},
        ],
        options={"maxiter": maxiter, "ftol": 1e-14},
    )
    lam, A, B = split(np.clip(res.x, 0, None))
    joint = np.einsum("w,pa,pb,wa,wb->pw", lam, lifted.plus, lifted.minus, A, B)
    return joint / np.maximum(joint.sum(axis=1, keepdims=True), 1e-300)


def bilevel_search(
    bidir: BidirectionalEpsilonMachine,
    starts: Sequence[np.ndarray],
    *,
    weight: float = 1e3,
    maxiter: int = 300,
) -> list[tuple[float, float, np.ndarray]]:
    """Outer search over channels ``c[pair, w]`` scoring ``H[W] + weight * t(c)``, where ``t`` is the
    linear-programming violation of :func:`generator_kernel`, from the given starting channels (derivative-free;
    small problems). Returns ``(entropy, violation, channel)`` per start, best feasible first.
    """
    problem = _GeneratorProblem(bidir)
    n = len(problem.lifted.p)
    out = []
    for c0 in starts:
        k = c0.shape[1]

        def channel(z: np.ndarray, k: int = k) -> np.ndarray:
            z = z.reshape(n, k)
            z = np.exp(z - z.max(axis=1, keepdims=True))
            return z / z.sum(axis=1, keepdims=True)

        def score(z: np.ndarray, channel=channel) -> float:
            c = channel(z)
            pi = problem.lifted.p @ c
            p = pi[pi > 1e-15]
            return float(-(p * np.log2(p)).sum()) + weight * generator_kernel(problem, c)[0]

        z0 = np.log(np.maximum(c0, 1e-6)).ravel()
        z = minimize(score, z0, method="Powell", options={"maxiter": maxiter, "xtol": 1e-6, "ftol": 1e-10}).x
        c = channel(z)
        pi = problem.lifted.p @ c
        p = pi[pi > 1e-15]
        out.append((float(-(p * np.log2(p)).sum()), generator_kernel(problem, c)[0], c))
    return sorted(out, key=lambda t: (t[1] > 1e-7, t[0]))


Seed = tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]


def _pad(seed: Seed, k: int, rng: np.random.Generator, noise: float) -> Seed:
    """Fit a seed with fewer or more states into ``k`` states, adding small mass to empty states."""
    pi, Q, R, K = seed
    m = len(pi)
    if m > k:
        keep = np.argsort(-pi)[:k]
        pi, Q, R, K = pi[keep], Q[keep], R[keep], K[np.ix_(keep, np.arange(K.shape[1]), keep)]
        m = k
    pi2 = np.full(k, noise)
    pi2[:m] = pi
    Q2 = rng.dirichlet(np.ones(Q.shape[1]), size=k)
    Q2[:m] = Q
    R2 = rng.dirichlet(np.ones(R.shape[1]), size=k)
    R2[:m] = R
    K2 = rng.dirichlet(np.ones(K.shape[1] * k), size=k).reshape(k, K.shape[1], k) * noise
    K2[:m, :, :m] += K
    K2 /= K2.reshape(k, -1).sum(axis=1)[:, None, None]
    return pi2 / pi2.sum(), Q2, R2, K2


@dataclass
class CoupledSearchResult:
    """One optimized k-state generator: state law, beliefs, kernel, entropy and residuals."""

    pi: np.ndarray
    Q: np.ndarray
    R: np.ndarray
    K: np.ndarray
    entropy: float
    residual: float
    start: str


def _optimize(problem: _GeneratorProblem, theta: np.ndarray, k: int, maxiter: int) -> np.ndarray:
    """Feasibility first, then an augmented Lagrangian for min H[pi] subject to the generator equations."""
    theta = minimize(
        problem.loss_and_grad,
        theta,
        args=(k, 1.0, None, 0.0),
        jac=True,
        method="L-BFGS-B",
        options={"maxiter": maxiter},
    ).x
    mu = _ALM_MU
    lam = [np.zeros_like(r) for r in problem.residuals(*problem.unpack(theta, k))]
    for _ in range(_ALM_ROUNDS):
        shifts = tuple(l / mu for l in lam)
        theta = minimize(
            problem.loss_and_grad,
            theta,
            args=(k, mu, shifts),
            jac=True,
            method="L-BFGS-B",
            options={"maxiter": maxiter},
        ).x
        res = problem.residuals(*problem.unpack(theta, k))
        lam = [l + mu * r for l, r in zip(lam, res, strict=False)]
        if max(np.abs(r).max() for r in res) < _ALM_TOL:
            break
        mu = min(mu * 2.0, _ALM_MU_MAX)
    return theta


def coupled_search(
    bidir: BidirectionalEpsilonMachine,
    k: int,
    *,
    starts: int = 10,
    extra: Sequence[tuple[str, tuple[np.ndarray, ...]]] = (),
    maxiter: int = 3000,
    residual_atol: float = 1e-6,
    rng: np.random.Generator | None = None,
) -> list[CoupledSearchResult]:
    """Optimize k-state generators from the standard seeds, ``extra`` named seeds and random starts.

    Returns every candidate whose generator equations hold to ``residual_atol``, best first.
    """
    rng = rng or np.random.default_rng()
    problem = _GeneratorProblem(bidir)
    named = [("eps", problem.epsilon_seed()), ("reverse", problem.reverse_seed()), ("bidir", problem.pair_seed())]
    named += list(extra)
    thetas = [(name, problem.logits(*_pad(cast(Seed, seed), k, rng, 1e-3))) for name, seed in named]
    thetas += [
        (f"random{i}", rng.normal(scale=1.5, size=sum(int(np.prod(s)) for s in problem.sizes(k))))
        for i in range(starts)
    ]
    out = []
    for name, theta in thetas:
        theta = _optimize(problem, theta, k, maxiter)
        pi, Q, R, K = problem.unpack(theta, k)
        residual = problem.residual(pi, Q, R, K)
        if residual <= residual_atol:
            p = pi[pi > 1e-12]
            out.append(CoupledSearchResult(pi, Q, R, K, float(-(p * np.log2(p)).sum()), residual, name))
    return sorted(out, key=lambda r: r.entropy)


class CoupledGenerativeModel(MealyHMM):
    """A generator found by :func:`coupled_generative_model`; ``generative_complexity`` is its state entropy."""

    markov_residual: float

    def __init__(self, *, markov_residual: float = 0.0, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.markov_residual = float(markov_residual)

    def generative_complexity(self) -> float:
        return self.state_entropy()


def _model_from_kernel(
    problem: _GeneratorProblem, result: CoupledSearchResult, cutoff: float
) -> CoupledGenerativeModel:
    active = [w for w in range(len(result.pi)) if result.pi[w] > cutoff]
    names = {w: f"G{i}" for i, w in enumerate(active)}
    graph = TransitionGraph()
    for w in active:
        graph.add_state(names[w])
    for w in active:
        row = result.K[w][:, active]
        row = np.where(row > cutoff, row, 0.0)
        row /= row.sum()
        for xi, x in enumerate(problem.symbols):
            for j, v in enumerate(active):
                if row[xi, j] > 0:
                    graph.add_transition(names[w], names[v], **{ATTR_PROB: float(row[xi, j]), ATTR_EMISSION: x})
    mass = result.pi[active] / result.pi[active].sum()
    model = CoupledGenerativeModel(
        graph=graph,
        initial_distribution={names[w]: float(m) for w, m in zip(active, mass, strict=False)},
        observation_alphabet=problem.bidir.observation_alphabet,
        markov_residual=result.residual,
    )
    model.validate()
    return model


def coupled_generative_model(
    bidir: BidirectionalEpsilonMachine,
    *,
    states: int | Sequence[int] | None = None,
    starts: int = 10,
    extra: Sequence[tuple[str, tuple[np.ndarray, ...]]] = (),
    maxiter: int = 3000,
    residual_atol: float = 1e-6,
    reproduction_atol: float = 1e-5,
    cutoff: float = 1e-9,
    rng: np.random.Generator | None = None,
) -> CoupledGenerativeModel:
    """Least-state-entropy generator found over the given numbers of states (default 2 .. #causal-state pairs).

    Every candidate is checked against the source's word probabilities; raises if none reproduces the process.
    """
    from sofic.generators.minimal_generative_model import _reproduction_error, _reproduction_length

    rng = rng or np.random.default_rng()
    problem = _GeneratorProblem(bidir)
    if states is None:
        states = range(2, len(problem.p_pair) + 1)
    elif isinstance(states, int):
        states = [states]
    candidates = []
    for k in states:
        candidates += coupled_search(
            bidir, k, starts=starts, extra=extra, maxiter=maxiter, residual_atol=residual_atol, rng=rng
        )
    reference = bidir.forward_machine
    length = _reproduction_length(reference)
    for result in sorted(candidates, key=lambda r: r.entropy):
        model = _model_from_kernel(problem, result, cutoff)
        if _reproduction_error(model, reference, max_length=length) <= reproduction_atol:
            return model
    raise StochasticValidationError("coupled generative model: no candidate reproduced the source process")


__all__ = [
    "CoupledGenerativeModel",
    "CoupledSearchResult",
    "bilevel_search",
    "coupled_generative_model",
    "coupled_search",
    "generator_kernel",
    "memory_lift",
    "memory_lift_search",
    "polish_separating_channel",
]
