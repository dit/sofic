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
_PE_TOL = 1e-15


@dataclass(frozen=True)
class _ConstraintWeights:
    """Per-entry masks and scales for edge / forward / reverse residuals."""

    weight_edge: np.ndarray
    weight_forward: np.ndarray
    weight_reverse: np.ndarray


def _kernel_lp_beliefs(
    problem: _GeneratorProblem, pi: np.ndarray, Q: np.ndarray, R: np.ndarray
) -> tuple[float, np.ndarray]:
    """Minimize max equation violation over stochastic ``K`` with fixed ``(pi, Q, R)``."""
    from scipy.optimize import linprog

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

    for a in range(problem.n_plus):
        for x in range(X):
            for b in range(problem.n_minus):
                add(((idx(w, x, v), A[w, a] * R[v, b]) for w in range(k) for v in range(k)), problem.Pe[a, x, b])
    for v in range(k):
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
    for w in range(k):
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
    dense = np.array(rows)
    target = np.array(rhs)
    m = len(rows)
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
        self._weights: _ConstraintWeights | None = None

    def constraint_weights(self) -> _ConstraintWeights:
        if self._weights is None:
            scale_e = max(float(np.max(np.abs(self.Pe))), 1e-8)
            wE = np.where(
                np.abs(self.Pe) > _PE_TOL,
                1.0 / np.maximum(np.abs(self.Pe), 1e-8),
                1.0 / scale_e,
            )
            self._weights = _ConstraintWeights(
                weight_edge=wE,
                weight_forward=1.0 / np.maximum(self.p_plus, 1e-8),
                weight_reverse=1.0 / np.maximum(self.p_minus, 1e-8),
            )
        return self._weights

    @classmethod
    def from_edge_tensors(
        cls,
        *,
        Pe: np.ndarray,
        F: np.ndarray,
        G: np.ndarray,
        p_pair: np.ndarray,
        p_plus: np.ndarray,
        p_minus: np.ndarray,
        symbols: Sequence[Any],
        bidir: Any | None = None,
    ) -> _GeneratorProblem:
        """Construct from explicit tensors (truncated reverse); ``bidir`` optional for reproduction checks."""
        if bidir is None:
            raise StochasticValidationError(
                "_GeneratorProblem.from_edge_tensors requires bidir until word verification is tensor-native"
            )
        return cls(bidir)

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

    def belief_sizes(self, k: int) -> list[tuple[int, ...]]:
        return [(k,), (k, self.n_plus), (k, self.n_minus)]

    def unpack_beliefs(self, theta: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        parts, i = [], 0
        for shape in self.belief_sizes(k):
            n = int(np.prod(shape))
            z = theta[i : i + n].reshape(shape)
            z = np.exp(z - z.max(axis=-1, keepdims=True))
            parts.append(z / z.sum(axis=-1, keepdims=True))
            i += n
        return parts[0], parts[1], parts[2]

    @staticmethod
    def logits_beliefs(pi: np.ndarray, Q: np.ndarray, R: np.ndarray, floor: float = 1e-9) -> np.ndarray:
        return np.concatenate([np.log(np.maximum(a, floor)).ravel() for a in (pi, Q, R)])

    @staticmethod
    def logits(pi: np.ndarray, Q: np.ndarray, R: np.ndarray, K: np.ndarray, floor: float = 1e-9) -> np.ndarray:
        return np.concatenate(
            [
                _GeneratorProblem.logits_beliefs(pi, Q, R, floor=floor),
                np.log(np.maximum(K.reshape(len(pi), -1), floor)).ravel(),
            ]
        )

    def _weighted_block_loss(
        self,
        rE: np.ndarray,
        rF: np.ndarray,
        rB: np.ndarray,
        A: np.ndarray,
        *,
        shifts: tuple[np.ndarray, np.ndarray, np.ndarray] | None,
        block_mus: tuple[float, float, float],
    ) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
        cw = self.constraint_weights()
        wE = cw.weight_edge
        wF = np.broadcast_to(cw.weight_forward, rF.shape)
        wB = np.broadcast_to(cw.weight_reverse, rB.shape)
        if shifts is not None:
            rE, rF, rB = rE + shifts[0], rF + shifts[1], rB + shifts[2]
        rEw, rFw, rBw = wE * rE, wF * rF, wB * rB
        muE, muF, muB = block_mus
        loss = 0.5 * float(muE * (rEw**2).sum() + muF * (rFw**2).sum() + muB * (rBw**2).sum())
        return loss, rE, rF, rB

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
        self,
        theta: np.ndarray,
        k: int,
        mu: float,
        shifts: tuple[np.ndarray, ...] | None = None,
        weight: float = 1.0,
        block_mus: tuple[float, float, float] | None = None,
    ) -> tuple[float, np.ndarray]:
        """Augmented Lagrangian with optional block penalties and entrywise residual scaling."""
        pi, Q, R, K = self.unpack(theta, k)
        loss, grad = self._loss_and_grad_core(pi, Q, R, K, k, mu, shifts, weight, block_mus, grad_K=True)
        return loss, grad

    def loss_and_grad_beliefs(
        self,
        theta: np.ndarray,
        k: int,
        K: np.ndarray,
        mu: float,
        shifts: tuple[np.ndarray, ...] | None = None,
        weight: float = 1.0,
        block_mus: tuple[float, float, float] | None = None,
    ) -> tuple[float, np.ndarray]:
        """Same objective with ``K`` fixed (alternating kernel LP step)."""
        pi, Q, R = self.unpack_beliefs(theta, k)
        loss, grad_beliefs = self._loss_and_grad_core(pi, Q, R, K, k, mu, shifts, weight, block_mus, grad_K=False)
        return loss, grad_beliefs

    def _loss_and_grad_core(
        self,
        pi: np.ndarray,
        Q: np.ndarray,
        R: np.ndarray,
        K: np.ndarray,
        k: int,
        mu: float,
        shifts: tuple[np.ndarray, ...] | None,
        weight: float,
        block_mus: tuple[float, float, float] | None,
        *,
        grad_K: bool,
    ) -> tuple[float, np.ndarray]:
        block = block_mus if block_mus is not None else (mu, mu, mu)
        A = pi[:, None] * Q
        M = np.einsum("wa,wxv->axv", A, K)
        rE = np.einsum("axv,vb->axb", M, R) - self.Pe
        rF = np.einsum("axv,axc->vc", M, self.F) - A
        N = np.einsum("wxv,vc,xcb->wb", K, R, self.G)
        D = N - R
        rB = pi[:, None] * D
        shift_tuple = None
        if shifts is not None:
            shift_tuple = (shifts[0], shifts[1], shifts[2])
        penalty, rE, rF, rB = self._weighted_block_loss(rE, rF, rB, A, shifts=shift_tuple, block_mus=block)
        cw = self.constraint_weights()
        wE, wF_row, wB_row = cw.weight_edge, cw.weight_forward, cw.weight_reverse
        wF = np.broadcast_to(wF_row, A.shape)
        wB = np.broadcast_to(wB_row, R.shape)
        muE, muF, muB = block
        rEeff = wE * wE * muE * rE
        rFeff = wF * wF * muF * rF
        rBeff = wB * wB * muB * rB

        gM = np.einsum("axb,vb->axv", rEeff, R) + np.einsum("vc,axc->axv", rFeff, self.F)
        gR = np.einsum("axb,axv->vb", rEeff, M)
        gA = np.einsum("wxv,axv->wa", K, gM) - rFeff
        gK = np.einsum("wa,axv->wxv", A, gM) if grad_K else None
        gpi = (gA * Q).sum(axis=1) + (rBeff * D).sum(axis=1)
        gQ = gA * pi[:, None]
        wBpen = pi[:, None] * rBeff
        if grad_K:
            gK = gK + np.einsum("wb,vc,xcb->wxv", wBpen, R, self.G)
        gR = gR + np.einsum("wb,wxv,xcb->vc", wBpen, K, self.G) - wBpen

        logp = np.log2(np.maximum(pi, 1e-300))
        h = float(-(pi * logp).sum())
        gpi_total = -weight * (logp + 1.0 / _LN2) + gpi

        def through_softmax(p: np.ndarray, g: np.ndarray) -> np.ndarray:
            return p * (g - (g * p).sum(axis=-1, keepdims=True))

        grads = [
            through_softmax(pi, gpi_total),
            through_softmax(Q, gQ),
            through_softmax(R, gR),
        ]
        if grad_K:
            grads.append(through_softmax(K.reshape(k, -1), gK.reshape(k, -1)))
            return weight * h + penalty, np.concatenate([g.ravel() for g in grads])
        return weight * h + penalty, np.concatenate([g.ravel() for g in grads])

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
    problem = _as_problem(bidir)
    pi, Q, R, _ = problem.channel_seed(c)
    return _kernel_lp_beliefs(problem, pi, Q, R)


def _decompose(
    P: np.ndarray, lam0: np.ndarray, Q0: np.ndarray, R0: np.ndarray, maxiter: int = 500
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Minimize ``H[lam]`` over exact decompositions ``P = sum_w lam_w Q_w R_w^T`` with SLSQP from ``(lam0, Q0, R0)``.

    Returns ``(lam, Q, R, residual)``, the residual being the largest entry of ``|sum_w lam_w Q_w R_w^T - P|``.
    """
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
    ones = np.ones(1 + 2 * k)
    res = minimize(
        H,
        np.concatenate([lam0, Q0.ravel(), R0.ravel()]),
        jac=dH,
        method="SLSQP",
        bounds=[(0, 1)] * n,
        constraints=[
            {"type": "eq", "fun": dec, "jac": ddec},
            {"type": "eq", "fun": lambda x: S @ x - ones, "jac": lambda x: S},
        ],
        options={"maxiter": maxiter, "ftol": 1e-14},
    )
    lam, A, B = split(np.clip(res.x, 0, None))
    residual = float(np.abs(np.einsum("w,wa,wb->ab", lam, A, B) - P).max())
    return lam, A, B, residual


def _pair_joint(problem: _GeneratorProblem) -> np.ndarray:
    lifted = problem.lifted
    return lifted.plus.T @ (lifted.p[:, None] * lifted.minus)


def _channel_from_decomposition(problem: _GeneratorProblem, lam, A, B) -> np.ndarray:
    lifted = problem.lifted
    joint = np.einsum("w,pa,pb,wa,wb->pw", lam, lifted.plus, lifted.minus, A, B)
    return joint / np.maximum(joint.sum(axis=1, keepdims=True), 1e-300)


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
    pi, Q, R, _ = problem.channel_seed(c)
    keep = pi > 1e-12
    lam, A, B, _ = _decompose(_pair_joint(problem), pi[keep], Q[keep], R[keep], maxiter)
    return _channel_from_decomposition(problem, lam, A, B)


def separating_optima(
    bidir: BidirectionalEpsilonMachine | _GeneratorProblem,
    *,
    starts: int = 60,
    sizes: Sequence[int] | None = None,
    atol: float = 1e-7,
    rng: np.random.Generator | None = None,
) -> tuple[float, list[np.ndarray]]:
    """Distinct optima of the exact common information ``G[S+ : S-]`` found by multistart exact decomposition.

    Runs :func:`_decompose` (SLSQP with exact constraints) from ``starts`` random points for each number of terms in
    ``sizes`` (default ``2 .. min(|S+| |S-|, 8)``, within the Kumar-Li-El Gamal bound :cite:`Kumar2014`), keeps the
    solutions within ``atol`` of the best value, removes duplicates (equal sets of terms up to relabeling) and
    returns ``(best value, channels)``. The optimum need not be unique, and only some optima may be generator
    states; :func:`generator_kernel` decides which.
    """
    problem = _as_problem(bidir)
    rng = rng or np.random.default_rng()
    P = _pair_joint(problem)
    na, nb = P.shape
    sizes = sizes or range(2, min(na * nb, 8) + 1)
    found = []
    for k in sizes:
        for _ in range(starts):
            lam, A, B, residual = _decompose(
                P, rng.dirichlet(np.ones(k)), rng.dirichlet(np.ones(na), size=k), rng.dirichlet(np.ones(nb), size=k)
            )
            if residual <= 1e-10:
                live = lam > 1e-10
                p = lam[live]
                found.append((float(-(p * np.log2(p)).sum()), lam[live], A[live], B[live]))
    if not found:
        return np.inf, []
    best = min(f[0] for f in found)
    optima, keys = [], []
    for value, lam, A, B in sorted(found, key=lambda f: f[0]):
        if value > best + atol:
            break
        key = sorted(tuple(np.round(np.concatenate([[l], a, b]), 6)) for l, a, b in zip(lam, A, B, strict=True))
        if key not in keys:
            keys.append(key)
            optima.append(_channel_from_decomposition(problem, lam, A, B))
    return best, optima


def separating_decomposition_channels(
    bidir: BidirectionalEpsilonMachine | _GeneratorProblem,
    *,
    starts: int = 40,
    sizes: Sequence[int] | None = None,
    residual_atol: float = 1e-10,
    rng: np.random.Generator | None = None,
) -> list[tuple[float, np.ndarray]]:
    """Separating channels from multistart exact decompositions, sorted by ``H[W]`` (not only ``G`` optima).

    Each entry is ``(entropy, channel)`` for one distinct decomposition of ``p(s^+, s^-)`` found by
    :func:`_decompose`. Used to search for a **generating** separator with entropy below the current upper
    bound when every minimum-entropy optimum fails :func:`generator_kernel`.
    """
    problem = _as_problem(bidir)
    rng = rng or np.random.default_rng()
    P = _pair_joint(problem)
    na, nb = P.shape
    sizes = sizes or range(2, min(na * nb, 8) + 1)
    found: list[tuple[float, np.ndarray, tuple]] = []
    for k in sizes:
        for _ in range(starts):
            lam, A, B, residual = _decompose(
                P, rng.dirichlet(np.ones(k)), rng.dirichlet(np.ones(na), size=k), rng.dirichlet(np.ones(nb), size=k)
            )
            if residual > residual_atol:
                continue
            live = lam > 1e-10
            p = lam[live]
            lam, A, B = lam[live], A[live], B[live]
            key = tuple(
                sorted(tuple(np.round(np.concatenate([[l], a, b]), 6)) for l, a, b in zip(lam, A, B, strict=True))
            )
            channel = _channel_from_decomposition(problem, lam, A, B)
            found.append((float(-(p * np.log2(p)).sum()), channel, key))
    if not found:
        return []
    found.sort(key=lambda t: t[0])
    out: list[tuple[float, np.ndarray]] = []
    seen: list[tuple] = []
    for entropy, channel, key in found:
        if key in seen:
            continue
        seen.append(key)
        out.append((entropy, channel))
    return out


def first_generating_separator(
    bidir: BidirectionalEpsilonMachine | _GeneratorProblem,
    channels: Sequence[tuple[float, np.ndarray]],
    *,
    max_entropy: float | None = None,
    kernel_atol: float = 1e-9,
    polish: bool = True,
) -> tuple[float, np.ndarray, np.ndarray] | None:
    """First channel in ``channels`` (lowest entropy first) that passes the generator kernel LP."""
    problem = _as_problem(bidir)
    for entropy, channel in channels:
        if max_entropy is not None and entropy > max_entropy + 1e-9:
            break
        c = polish_separating_channel(problem, channel) if polish else channel
        violation, K = generator_kernel(problem, c)
        if violation <= kernel_atol:
            return entropy, c, K
    return None


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


def generator_state_cap(problem: _GeneratorProblem, *, hard_cap: int = 8) -> int:
    """Upper limit on hidden state count for coupled search (Carathéodory + support size)."""
    na, nb = len(problem.p_plus), len(problem.p_minus)
    n_cells = int(np.sum(problem.p_pair > 1e-14))
    carath = max(2, na + nb - 1)
    return min(hard_cap, len(problem.p_pair), n_cells, carath)


def systematic_coupled_seeds(
    problem: _GeneratorProblem,
    channels: Sequence[np.ndarray],
    *,
    max_permutations: int = 24,
) -> list[tuple[str, Seed]]:
    """Named coupled seeds from polished channels and label permutations (Phase B systematic restarts)."""
    import itertools

    out: list[tuple[str, Seed]] = []
    seen: set[tuple[float, ...]] = set()

    def register(name: str, seed: Seed) -> None:
        key = tuple(np.round(np.concatenate([seed[0], seed[1].ravel(), seed[2].ravel(), seed[3].ravel()]), 8))
        if key in seen:
            return
        seen.add(key)
        out.append((name, seed))

    for i, raw in enumerate(channels):
        c = polish_separating_channel(problem, raw)
        register(f"systematic_channel_{i}", problem.channel_seed(c))
        k = c.shape[1]
        if k > 6:
            continue
        n_perm = 0
        for perm in itertools.permutations(range(k)):
            if perm == tuple(range(k)):
                continue
            register(f"systematic_perm_{i}_{n_perm}", problem.channel_seed(c[:, perm]))
            n_perm += 1
            if n_perm >= max_permutations:
                break
    return out


def line_split_seeds(
    problem: _GeneratorProblem,
    channels: Sequence[np.ndarray],
    *,
    max_broad: int = 2,
) -> list[tuple[str, Seed]]:
    """Exact decompositions mixing broad terms with point-mass beliefs, as coupled seeds.

    For each separating channel, keep the ``j <= max_broad`` heaviest terms of its decomposition
    :math:`p(s^+, s^-) = \\sum_w \\pi_w Q_w R_w^\\top` and split the nonnegative remainder by columns
    (one state per reverse causal state, ``R = e_b``) or by rows (``Q = e_a``). Minimal generators
    found by random restarts often have this shape: broad states that nearly separate the causal
    states plus small states certain of one of them. Kernels come from the kernel LP.
    """
    P = problem.lifted.plus.T @ (problem.p_pair[:, None] * problem.lifted.minus)
    na, nb = P.shape
    out: list[tuple[str, Seed]] = []
    for i, c in enumerate(channels):
        pi, Q, R, _ = problem.channel_seed(c)
        order = np.argsort(-pi)
        for j in range(1, min(max_broad, int((pi > 1e-9).sum())) + 1):
            top = order[:j]
            rest = np.maximum(P - np.einsum("w,wa,wb->ab", pi[top], Q[top], R[top]), 0.0)
            for axis, name in ((0, "columns"), (1, "rows")):
                lines = rest.sum(axis=axis)
                live = np.flatnonzero(lines > 1e-12)
                if axis == 0:
                    Q_new, R_new = (rest[:, live] / lines[live]).T, np.eye(nb)[live]
                else:
                    Q_new, R_new = np.eye(na)[live], rest[live] / lines[live][:, None]
                weights = np.concatenate([pi[top], lines[live]])
                beliefs = (weights / weights.sum(), np.vstack([Q[top], Q_new]), np.vstack([R[top], R_new]))
                _violation, K = _kernel_lp_beliefs(problem, *beliefs)
                out.append((f"line split {i} (top {j} + {name})", (*beliefs, K)))
    return out


def _line_split_beliefs(
    P: np.ndarray, z: np.ndarray, broad: int, axis: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``broad`` rank-1 terms with softmax beliefs ``z``, each at its largest weight under the running
    remainder, then the remainder split into point-mass states by columns (``axis=0``) or rows."""
    from sofic.generators.coverage_bound import mu1_rank1_mass

    na, nb = P.shape
    rest = P.copy()
    weights, Qs, Rs = [], [], []
    for i in range(broad):
        zi = z[i * (na + nb) : (i + 1) * (na + nb)]
        q = np.exp(zi[:na] - zi[:na].max())
        r = np.exp(zi[na:] - zi[na:].max())
        q, r = q / q.sum(), r / r.sum()
        lam = mu1_rank1_mass(rest, q, r)
        rest = np.maximum(rest - lam * np.outer(q, r), 0.0)
        weights.append(lam)
        Qs.append(q)
        Rs.append(r)
    lines = rest.sum(axis=axis)
    live = np.flatnonzero(lines > 1e-12)
    if axis == 0:
        Q_new, R_new = (rest[:, live] / lines[live]).T, np.eye(nb)[live]
    else:
        Q_new, R_new = np.eye(na)[live], rest[live] / lines[live][:, None]
    w = np.concatenate([weights, lines[live]])
    return w / w.sum(), np.vstack([*Qs, Q_new]), np.vstack([*Rs, R_new])


def targeted_line_split_seeds(
    problem: _GeneratorProblem,
    channels: Sequence[np.ndarray],
    *,
    configs: Sequence[tuple[int, int]] = ((1, 0), (1, 1), (2, 0), (2, 1)),
    perturbations: int = 2,
    residual_atol: float = 1e-7,
    rng: np.random.Generator | None = None,
) -> list[tuple[str, Seed]]:
    """Generators found by searching the line-split family directly for kernel-LP feasibility.

    A member of the family is ``broad`` rank-1 terms, each at its largest weight under the
    remainder, plus the remainder split into point-mass states (:func:`line_split_seeds`), so it is
    an exact decomposition fixed by the broad terms' beliefs. From the heaviest terms of each channel's
    decomposition (and ``perturbations`` perturbations of them), Nelder-Mead reaches a decomposition
    whose kernel LP is feasible, then COBYLA minimizes its state entropy subject to staying feasible.
    Each result is polished by the coupled ALM from several logit floors; seeds that reach the
    generator equations to ``residual_atol`` are returned. ``configs`` lists ``(broad, axis)`` pairs.
    """
    rng = rng or np.random.default_rng()
    P = problem.lifted.plus.T @ (problem.p_pair[:, None] * problem.lifted.minus)
    out: list[tuple[str, Seed]] = []
    for i, c in enumerate(channels):
        pi0, Q0, R0, _ = problem.channel_seed(c)
        order = np.argsort(-pi0)
        for broad, axis in configs:
            if broad > int((pi0 > 1e-9).sum()):
                continue
            base = np.concatenate([np.log(np.maximum(np.concatenate([Q0[w], R0[w]]), 1e-6)) for w in order[:broad]])

            def violation(z: np.ndarray, broad: int = broad, axis: int = axis) -> float:
                return _kernel_lp_beliefs(problem, *_line_split_beliefs(P, z, broad, axis))[0]

            def entropy(z: np.ndarray, broad: int = broad, axis: int = axis) -> float:
                return _entropy(_line_split_beliefs(P, z, broad, axis)[0])

            starts = [base] + [base + rng.normal(scale=0.3, size=base.shape) for _ in range(perturbations)]
            for s, z in enumerate(starts):
                if violation(z) >= 1e-6:
                    res = minimize(
                        violation,
                        z,
                        method="Nelder-Mead",
                        options={"maxiter": 300 * len(z), "xatol": 1e-9, "fatol": 1e-13},
                    )
                    if res.fun >= 1e-6:
                        continue
                    z = res.x
                constraint = [{"type": "ineq", "fun": lambda y, v=violation: 1e-9 - v(y)}]
                for rho in (0.5, 0.1):
                    res = minimize(
                        entropy,
                        z,
                        method="COBYLA",
                        constraints=constraint,
                        options={"rhobeg": rho, "maxiter": 3000, "tol": 1e-10},
                    )
                    if violation(res.x) < 1e-6:
                        z = res.x
                beliefs = _line_split_beliefs(P, z, broad, axis)
                _violation, K = _kernel_lp_beliefs(problem, *beliefs)
                k = len(beliefs[0])
                lines = "columns" if axis == 0 else "rows"
                polished = []
                for floor in (1e-12, 1e-5, 1e-3):
                    theta = _optimize_coupled(
                        problem, problem.logits(*beliefs, K, floor=floor), k, 3000, residual_atol=residual_atol
                    )
                    seed = problem.unpack(theta, k)
                    if problem.residual(*seed) <= residual_atol:
                        polished.append((_entropy(seed[0]), seed))
                if polished:
                    seed = min(polished, key=lambda pair: pair[0])[1]
                    out.append((f"targeted line split {i}.{s} ({broad} broad + {lines})", cast(Seed, seed)))
    return out


def _pad(seed: Seed, k: int, rng: np.random.Generator, noise: float) -> Seed:
    """Fit a seed with fewer or more states into ``k`` states, adding small mass to empty states."""
    pi, Q, R, K = seed
    m = len(pi)
    if m == k:
        return pi / pi.sum(), Q, R, K
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


def _optimize(
    problem: _GeneratorProblem,
    theta: np.ndarray,
    k: int,
    maxiter: int,
    *,
    alternating: bool = False,
) -> np.ndarray:
    """Feasibility pass, then block ALM; joint beliefs+kernel by default."""
    block_mus = (_ALM_MU, _ALM_MU, _ALM_MU)
    pi, Q, R, K = problem.unpack(theta, k)
    if problem.residual(pi, Q, R, K) <= _ALM_TOL:
        return theta

    def _block_max(res: tuple[np.ndarray, ...]) -> tuple[float, float, float]:
        return float(np.abs(res[0]).max()), float(np.abs(res[1]).max()), float(np.abs(res[2]).max())

    if alternating:
        theta_b = problem.logits_beliefs(pi, Q, R)
        lam = [np.zeros_like(r) for r in problem.residuals(pi, Q, R, K)]
        theta_b = minimize(
            problem.loss_and_grad_beliefs,
            theta_b,
            args=(k, K, 1.0, None, 0.0, block_mus),
            jac=True,
            method="L-BFGS-B",
            options={"maxiter": maxiter},
        ).x
        pi, Q, R = problem.unpack_beliefs(theta_b, k)
        for _ in range(_ALM_ROUNDS):
            _viol, K = _kernel_lp_beliefs(problem, pi, Q, R)
            shifts = tuple(l / m for l, m in zip(lam, block_mus, strict=False))
            theta_b = minimize(
                problem.loss_and_grad_beliefs,
                theta_b,
                args=(k, K, 1.0, shifts, 0.0, block_mus),
                jac=True,
                method="L-BFGS-B",
                options={"maxiter": maxiter},
            ).x
            pi, Q, R = problem.unpack_beliefs(theta_b, k)
            res = problem.residuals(pi, Q, R, K)
            lam = [l + m * r for l, r, m in zip(lam, res, block_mus, strict=False)]
            mE, mF, mB = _block_max(res)
            if max(mE, mF, mB) < _ALM_TOL:
                break
            block_mus = tuple(
                min(m * (2.0 if v > _ALM_TOL else 1.0), _ALM_MU_MAX)
                for m, v in zip(block_mus, (mE, mF, mB), strict=False)
            )
        return problem.logits(pi, Q, R, K)

    theta = minimize(
        problem.loss_and_grad,
        theta,
        args=(k, 1.0, None, 0.0, block_mus),
        jac=True,
        method="L-BFGS-B",
        options={"maxiter": maxiter},
    ).x
    lam = [np.zeros_like(r) for r in problem.residuals(*problem.unpack(theta, k))]
    for _ in range(_ALM_ROUNDS):
        shifts = tuple(l / m for l, m in zip(lam, block_mus, strict=False))
        theta = minimize(
            problem.loss_and_grad,
            theta,
            args=(k, 1.0, shifts, 0.0, block_mus),
            jac=True,
            method="L-BFGS-B",
            options={"maxiter": maxiter},
        ).x
        res = problem.residuals(*problem.unpack(theta, k))
        lam = [l + m * r for l, r, m in zip(lam, res, block_mus, strict=False)]
        mE, mF, mB = _block_max(res)
        if max(mE, mF, mB) < _ALM_TOL:
            break
        block_mus = tuple(
            min(m * (2.0 if v > _ALM_TOL else 1.0), _ALM_MU_MAX) for m, v in zip(block_mus, (mE, mF, mB), strict=False)
        )
    return theta


def _optimize_coupled(
    problem: _GeneratorProblem,
    theta: np.ndarray,
    k: int,
    maxiter: int,
    *,
    alternating: bool = False,
    dual_strategy: bool = True,
    residual_atol: float = 1e-6,
) -> np.ndarray:
    """Run block ALM; optionally retry with alternating ``K`` LP if joint ALM stays infeasible."""
    pi, Q, R, K = problem.unpack(theta, k)
    if problem.residual(pi, Q, R, K) <= residual_atol:
        return theta
    if alternating:
        return _optimize(problem, theta, k, maxiter, alternating=True)
    theta_joint = _optimize(problem, theta, k, maxiter, alternating=False)
    r_joint = problem.residual(*problem.unpack(theta_joint, k))
    if r_joint <= residual_atol or not dual_strategy:
        return theta_joint
    theta_alt = _optimize(problem, theta, k, maxiter, alternating=True)
    r_alt = problem.residual(*problem.unpack(theta_alt, k))
    if r_alt + 1e-15 < r_joint:
        return theta_alt
    if r_joint + 1e-15 < r_alt:
        return theta_joint
    h_joint = _entropy(problem.unpack(theta_joint, k)[0])
    h_alt = _entropy(problem.unpack(theta_alt, k)[0])
    return theta_alt if h_alt < h_joint else theta_joint


def _entropy(pi: np.ndarray) -> float:
    p = pi[pi > 1e-15]
    return float(-(p * np.log2(p)).sum())


def coupled_search(
    bidir: BidirectionalEpsilonMachine,
    k: int,
    *,
    starts: int = 10,
    extra: Sequence[tuple[str, tuple[np.ndarray, ...]]] = (),
    maxiter: int = 3000,
    residual_atol: float = 1e-6,
    alternating: bool = False,
    dual_strategy: bool = True,
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
        pi, Q, R, K = problem.unpack(theta, k)
        init_res = problem.residual(pi, Q, R, K)
        if init_res > residual_atol:
            theta = _optimize_coupled(
                problem,
                theta,
                k,
                maxiter,
                alternating=alternating,
                dual_strategy=dual_strategy,
                residual_atol=residual_atol,
            )
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
    alternating: bool = False,
    dual_strategy: bool = True,
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
            bidir,
            k,
            starts=starts,
            extra=extra,
            maxiter=maxiter,
            residual_atol=residual_atol,
            alternating=alternating,
            dual_strategy=dual_strategy,
            rng=rng,
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
    "first_generating_separator",
    "generator_kernel",
    "generator_state_cap",
    "line_split_seeds",
    "memory_lift",
    "memory_lift_search",
    "polish_separating_channel",
    "separating_decomposition_channels",
    "separating_optima",
    "systematic_coupled_seeds",
    "targeted_line_split_seeds",
]
