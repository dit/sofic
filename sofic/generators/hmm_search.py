r"""Belief-free search for low-entropy generators: optimize an HMM's symbol matrices directly.

The coupled searches of :mod:`~sofic.generators.coupled_generators` parameterize a generator's states by their
causal-state beliefs :math:`(Q_w, R_w)`. That parameterization favours generators whose states' pasts and futures
are belief-mixtures of the causal states', and on some processes it misses lower-entropy generators that are not.
:func:`hmm_generator_search` instead optimizes the :math:`k`-state symbol matrices :math:`T^{(x)}` (softmax over
:math:`(x, w')` for each row), minimizing the state entropy :math:`H[\pi]` subject to exact word probabilities
for all words up to length :math:`k + n`, where :math:`n` is the number of forward causal states. Two weighted
automata with :math:`k` and :math:`n` states that agree on all words of length up to :math:`k + n - 1` agree on all
words :cite:`Tzeng1992`, so feasible points are generators. The constraints enter an augmented Lagrangian, first
without the entropy term (to reach a generator from a random start) and then with it. Gradients are analytic:
forward and backward vectors over the word tree, and the stationary distribution differentiated through the
fundamental matrix. Direct parameter search for minimal generators of binary Markov processes is the approach of
:cite:`Ruebeck2018`; the word-equality constraints and the use as a pipeline stage are ours.
"""

from __future__ import annotations

import itertools
import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.optimize import minimize

__all__ = ["HMMSearchResult", "coupled_seed_from_hmm", "hmm_generator_search"]

# Zero entries of a given start are raised to this before taking logits (log 0 is not a logit).
_INITIAL_FLOOR = 1e-8


@dataclass(frozen=True)
class HMMSearchResult:
    """A generator found by :func:`hmm_generator_search`: symbol matrices, stationary law, and fit."""

    matrices: tuple[np.ndarray, ...]
    pi: np.ndarray
    entropy: float
    word_residual: float
    k: int
    start: int


def _entropy(p: np.ndarray) -> float:
    p = p[p > 1e-15]
    return float(-(p * np.log2(p)).sum())


def _stationary(M: np.ndarray) -> np.ndarray:
    """A stationary law of the stochastic matrix ``M`` (least squares, so defined for reducible chains too)."""
    k = M.shape[0]
    A = np.vstack([M.T - np.eye(k), np.ones(k)])
    b = np.zeros(k + 1)
    b[-1] = 1.0
    return np.linalg.lstsq(A, b, rcond=None)[0]


def _fundamental(M: np.ndarray, pi: np.ndarray) -> np.ndarray:
    """``(I - M + 1 pi)^{-1}``, so that ``d pi = pi dM Z``; a pseudo-inverse when the chain is reducible."""
    A = np.eye(M.shape[0]) - M + np.outer(np.ones(M.shape[0]), pi)
    try:
        return np.linalg.inv(A)
    except np.linalg.LinAlgError:
        return np.linalg.pinv(A)


class _WordTree:
    """All words of length 1 .. L over ``ns`` symbols, in breadth-first order (parents before children)."""

    def __init__(self, ns: int, L: int) -> None:
        self.words = [w for n in range(1, L + 1) for w in itertools.product(range(ns), repeat=n)]
        index = {w: i for i, w in enumerate(self.words)}
        self.parent = np.array([index.get(w[:-1], -1) for w in self.words])
        self.last = np.array([w[-1] for w in self.words])

    def forward(self, T: list[np.ndarray], pi: np.ndarray) -> np.ndarray:
        """Row vectors pi T^{(w)} for every word."""
        alpha = np.empty((len(self.words), len(pi)))
        for i, (p, x) in enumerate(zip(self.parent, self.last, strict=True)):
            alpha[i] = (pi if p < 0 else alpha[p]) @ T[x]
        return alpha


def _target_words(machine: Any, tree: _WordTree) -> np.ndarray:
    from sofic.generators.matrices import symbol_matrices

    mats = symbol_matrices(machine.to_mealy() if hasattr(machine, "to_mealy") else machine)
    T = [np.asarray(mats[x], float) for x in sorted(mats)]
    pi = _stationary(sum(T))
    return tree.forward(T, pi).sum(axis=1)


def _logits(matrices: Sequence[np.ndarray], k: int, ns: int, floor: float) -> np.ndarray:
    """Softmax logits (the inverse of :func:`_unpack`) of ``k``-state symbol matrices, entries floored first."""
    T = [np.asarray(M, float) for M in matrices]
    if len(T) != ns or any(M.shape != (k, k) for M in T):
        raise ValueError(f"initial generator must have {ns} symbol matrices of shape ({k}, {k})")
    E = np.concatenate([np.maximum(M, floor) for M in T], axis=1)
    E /= E.sum(axis=1, keepdims=True)
    return np.log(E).ravel()


def _unpack(theta: np.ndarray, k: int, ns: int) -> tuple[list[np.ndarray], np.ndarray]:
    L = theta.reshape(k, ns * k)
    E = np.exp(L - L.max(axis=1, keepdims=True))
    E /= E.sum(axis=1, keepdims=True)
    return [E[:, x * k : (x + 1) * k] for x in range(ns)], E


def _objective(theta, k, ns, tree, target, lam, mu, weight):
    """Augmented Lagrangian value and gradient in the softmax logits."""
    T, E = _unpack(theta, k, ns)
    M = sum(T)
    pi = _stationary(M)
    Z = _fundamental(M, pi)
    alpha = tree.forward(T, pi)
    r = alpha.sum(axis=1) - target
    pi_c = np.clip(pi, 1e-300, None)
    h = float(-(pi_c * np.log2(pi_c)).sum())
    value = weight * h + float(lam @ r) + 0.5 * mu * float(r @ r)
    g_word = lam + mu * r  # d value / d P(w)
    # reverse-mode through alpha(w) = alpha(parent) T^{(last)}, P(w) = alpha(w) . 1; children come after parents
    adj = np.outer(g_word, np.ones(k))
    gT = [np.zeros((k, k)) for _ in range(ns)]
    g_pi = np.zeros(k)
    for i in range(len(tree.words) - 1, -1, -1):
        p, x = tree.parent[i], tree.last[i]
        prev = pi if p < 0 else alpha[p]
        gT[x] += np.outer(prev, adj[i])
        back = T[x] @ adj[i]
        if p < 0:
            g_pi += back
        else:
            adj[p] += back
    g_pi += weight * (-(np.log2(pi_c) + 1.0 / np.log(2)))
    # pi = stationary(M): d pi = pi dM Z, so d value / d M = pi^T (Z g_pi)^T
    gM = np.outer(pi, Z @ g_pi)
    G = np.concatenate([gT[x] + gM for x in range(ns)], axis=1)  # d value / d E
    grad = E * (G - (G * E).sum(axis=1, keepdims=True))
    return value, grad.ravel()


def hmm_generator_search(
    machine: Any,
    k: int,
    *,
    starts: int = 8,
    initial: Sequence[Sequence[np.ndarray]] = (),
    rng: np.random.Generator | None = None,
    deadline: float | None = None,
    rounds: int = 12,
    feasibility_tol: float = 1e-7,
) -> list[HMMSearchResult]:
    """Low-entropy ``k``-state generators of the process of ``machine`` (its forward ε-machine), best first.

    Each of ``starts`` random starts reaches a generator with no entropy term, then descends in :math:`H[\\pi]`
    under the augmented Lagrangian (``rounds`` outer updates). ``initial`` adds starts from given ``k``-state HMMs
    (each a sequence of symbol matrices in symbol order), run before the random ones. Such a start is an
    approximate generator to be polished, for instance the solution of
    :func:`~sofic.generators.global_bound.generator_global_bound`: its rounds keep the entropy term off, so it
    moves to a nearby generator instead of descending (from a near-generator, the descent stalls at an infeasible
    point). A result's ``start`` counts the initial starts first. Only end points whose largest word residual is at
    most ``feasibility_tol`` are returned; callers should still verify them exactly (the pipeline does).
    ``deadline`` is a :func:`time.monotonic` value checked between starts and rounds.
    """
    from sofic.generators.matrices import symbol_matrices

    rng = rng if rng is not None else np.random.default_rng()
    mats = symbol_matrices(machine.to_mealy() if hasattr(machine, "to_mealy") else machine)
    ns, n = len(mats), next(iter(mats.values())).shape[0]
    given = [_logits(T, k, ns, _INITIAL_FLOOR) for T in initial]
    tree = _WordTree(ns, k + n)
    target = _target_words(machine, tree)
    opts = {"maxiter": 3000, "ftol": 1e-16, "gtol": 1e-12}
    out = []
    for start in range(len(given) + starts):
        if deadline is not None and time.monotonic() >= deadline:
            break
        theta = given[start] if start < len(given) else rng.normal(size=k * ns * k)
        weight = 0.0 if start < len(given) else 1.0
        lam = np.zeros(len(target))
        args = (k, ns, tree, target)
        theta = minimize(_objective, theta, args=(*args, lam, 10.0, 0.0), jac=True, method="L-BFGS-B", options=opts).x
        mu = 1e3
        for _ in range(rounds):
            if deadline is not None and time.monotonic() >= deadline:
                break
            theta = minimize(
                _objective, theta, args=(*args, lam, mu, weight), jac=True, method="L-BFGS-B", options=opts
            ).x
            T, _ = _unpack(theta, k, ns)
            pi = _stationary(sum(T))
            lam = lam + mu * (tree.forward(T, pi).sum(axis=1) - target)
            mu = min(mu * 3.0, 1e7)
        T, _ = _unpack(theta, k, ns)
        pi = _stationary(sum(T))
        resid = float(np.abs(tree.forward(T, pi).sum(axis=1) - target).max())
        if resid <= feasibility_tol:
            out.append(HMMSearchResult(tuple(T), pi, _entropy(pi), resid, k, start))
    return sorted(out, key=lambda r: r.entropy)


def coupled_seed_from_hmm(problem: Any, matrices: tuple[np.ndarray, ...]) -> tuple[np.ndarray, ...]:
    """The coupled form ``(pi, Q, R, K)`` of an HMM generator relative to ``problem``'s causal states.

    ``Q_w = Pr(S+_0 | W_0 = w)`` from the stationary law of the forward product chain (S+, W), ``R_w`` from the
    reverse-time product chain (S-, W), ``K[w, x, w'] = T^{(x)}[w, w']``. States of zero stationary weight are
    dropped.
    """
    T = [np.asarray(M, float) for M in matrices]
    pi = _stationary(sum(T))
    keep = np.flatnonzero(pi > 1e-12)
    T = [M[np.ix_(keep, keep)] / 1.0 for M in T]
    rows = sum(T).sum(axis=1, keepdims=True)
    T = [M / rows for M in T]
    pi = _stationary(sum(T))
    k = len(pi)
    na, nb = problem.n_plus, problem.n_minus
    Pf = np.zeros((na * k, na * k))
    for a, x, a2 in zip(*np.nonzero(problem.F), strict=True):
        Pf[a * k : (a + 1) * k, a2 * k : (a2 + 1) * k] += T[x]
    J = _stationary(Pf).reshape(na, k)
    Q = (J / np.maximum(J.sum(axis=0), 1e-300)).T
    Pr = np.zeros((nb * k, nb * k))
    for x, c, b in zip(*np.nonzero(problem.G), strict=True):
        Pr[c * k : (c + 1) * k, b * k : (b + 1) * k] += (pi[:, None] * T[x] / np.maximum(pi[None, :], 1e-300)).T
    J = _stationary(Pr).reshape(nb, k)
    R = (J / np.maximum(J.sum(axis=0), 1e-300)).T
    K = np.stack(T, axis=1)
    return pi, np.clip(Q, 0, None), np.clip(R, 0, None), K
