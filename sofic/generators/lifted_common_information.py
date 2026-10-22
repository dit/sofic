r"""Wyner and exact common information restricted to lifted generator states.

The Wyner (:cite:`Wyner1975`) and exact (:cite:`Kumar2014`) common informations
of the forward and reverse causal states optimize a channel
:math:`p(g \mid s^+, s^-)` such that :math:`G` renders :math:`S^+` and
:math:`S^-` conditionally independent. Because the past and future are
conditionally independent given each causal state (Ellison, Mahoney &
Crutchfield :cite:`Ellison2009`), any such :math:`G` also separates the past
from the future. It need not be the hidden state of a generator, however.

Applying the channel at both ends of every edge of the bidirectional
ε-machine (:math:`S^\pm_0 \to X_0 \to S^\pm_1`) and marginalizing gives a
stationary kernel :math:`p(x_0, g_1 \mid g_0)`: the *lifted* HMM. It
generates the source process, with :math:`G_0` jointly distributed with the
past and future exactly as the channel prescribes, if and only if

.. math::

   \op{I}{G_0 X_0 : S^-_1 \mid G_1} = 0
   \qquad\text{and}\qquad
   \op{I}{S^+_0 : X_0 G_1 \mid G_0} = 0.

The first condition says that :math:`G_1` screens what has been generated so
far from the reverse causal state :math:`S^-_1`, which carries everything the
future needs (sufficiency follows by induction on word length). The second is
its time reverse.
:func:`~sofic.generators.minimal_generative_model.wyner_generative_model` and
:func:`~sofic.generators.minimal_generative_model.minimal_generative_model`
bound both conditional mutual informations, together with
:math:`\op{I}{S^+ : S^- \mid G}`, by a small tolerance and minimize
:math:`\op{I}{(S^+, S^-) : G}` (Wyner) or :math:`\H{G}` (exact) with SLSQP. The
starts are the generator states :math:`S^+`, :math:`S^-` and
:math:`(S^+, S^-)`, the unconstrained optimum, and random channels. From each
start the closure tolerance is tightened in stages, beginning with the
unconstrained problem.
:func:`~sofic.generators.minimal_generative_model.functional_generative_model`
imposes the same conditions on deterministic channels, which can be checked
exactly, by a search over partitions of the joint states.

Every feasible point is a generator state, so the lifted exact value bounds
the generative complexity :math:`C_g` of Löhr & Ay :cite:`LohrAy2009` from
above. Both lifted values bound the corresponding unconstrained common
informations from above; they coincide for first-order Markov processes.

Lifted generators are a proper subclass of generators. The lift draws
:math:`G_0` and :math:`G_1` independently given the bidirectional edge; a
general generator may couple them. For
:math:`Z_t = Z_{t-2} \oplus \mathrm{Bernoulli}(0.1)`, the interleaved product of
two first-order generators has state :math:`G_t = (U_t, U_{t+1})`, sharing
:math:`U_{t+1}` with :math:`G_{t+1}`. It is a generator state of entropy 1.982,
essentially :math:`\op{G}{S^+ : S^-}`, yet violates both lifting conditions by
0.019 bits, so the lifted search cannot reach it. For a deterministic channel
the coupling is forced, so lifting is no restriction: the functional search
optimizes over all deterministic generator states.

The lifted values can also exceed the optimum over lifted generators because the
optimizer is local.
"""

from __future__ import annotations

from collections.abc import Hashable
from typing import TYPE_CHECKING

import numpy as np
from scipy.optimize import minimize

from sofic.exceptions import StochasticValidationError
from sofic.generators.minimal_generative_model import (
    MinimalGenerativeModel,
    WynerGenerativeModel,
    _auxiliary_state_channel,
    _matching_support_channel,
    _model_from_channel,
    _normalized_joint_distribution,
    _optimize_exact_common_information,
    _optimize_wyner_common_information,
    _reproduction_error,
    _reproduction_length,
)
from sofic.graph import ATTR_EMISSION, ATTR_PROB

if TYPE_CHECKING:
    from sofic.generators.bidirectional_epsilon_machine import BidirectionalEpsilonMachine

_LN2 = np.log(2.0)
_COARSE_TOL = 5e-8
_REFINE = 5
_SUPPORT_CUTOFF = 1e-8
_REFINE_TOL = 1e-12
_EXACT_ATOL = 1e-10
_EXHAUSTIVE_PAIRS = 9


class _LiftedProblem:
    """Joint-state tensors of a bidirectional ε-machine and the lifted constraints on a channel ``c[pair, g]``."""

    def __init__(self, bidir: BidirectionalEpsilonMachine, joint: dict[tuple[Hashable, Hashable], float]) -> None:
        self.pairs = sorted(joint, key=repr)
        plus = sorted({pair[0] for pair in self.pairs}, key=repr)
        minus = sorted({pair[1] for pair in self.pairs}, key=repr)
        symbols = sorted(bidir.observation_alphabet, key=repr)
        index = {pair: i for i, pair in enumerate(self.pairs)}
        n = len(self.pairs)
        self.p = np.array([joint[pair] for pair in self.pairs])
        self.plus = np.zeros((n, len(plus)))
        self.minus = np.zeros((n, len(minus)))
        for i, (alpha, gamma) in enumerate(self.pairs):
            self.plus[i, plus.index(alpha)] = 1.0
            self.minus[i, minus.index(gamma)] = 1.0
        self.edges = np.zeros((n, len(symbols), n))
        for pair in self.pairs:
            for transition in bidir.graph.out_transitions(pair):
                if transition.target in index:
                    x = symbols.index(transition.data.get(ATTR_EMISSION))
                    self.edges[index[pair], x, index[transition.target]] += joint[pair] * float(
                        transition.data.get(ATTR_PROB, 0.0)
                    )
        self.edges /= self.edges.sum()

    def tensors(self, c: np.ndarray) -> tuple[np.ndarray, ...]:
        """``p(s+, s-, g)``, ``p(g_0, x_0, s-_1, g_1)`` and ``p(s+_0, g_0, x_0, g_1)``."""
        joint_g = self.p[:, None] * c
        screen = np.einsum("pa,pb,pg->abg", self.plus, self.minus, joint_g)
        forward = np.einsum("pxq,qv->pxv", self.edges, c)
        future = np.einsum("pg,pxq,qs,qv->gxsv", c, self.edges, self.minus, c)
        past = np.einsum("pa,pg,pxv->agxv", self.plus, c, forward)
        return screen, future, past

    def constraint_informations(self, c: np.ndarray) -> np.ndarray:
        """``I[S+ : S- | G]``, ``I[G_0 X_0 : S-_1 | G_1]`` and ``I[S+_0 : X_0 G_1 | G_0]`` in bits."""
        screen, future, past = self.tensors(c)
        return np.array(
            [
                _conditional_mutual_information(screen),
                _conditional_mutual_information(future.reshape(-1, *future.shape[2:])),
                _conditional_mutual_information(np.moveaxis(past, 1, -1).reshape(past.shape[0], -1, past.shape[1])),
            ]
        )

    def constraint_jacobian(self, c: np.ndarray) -> np.ndarray:
        """Gradients of :meth:`constraint_informations` in ``c``, shape ``(3, *c.shape)``."""
        screen, future, past = self.tensors(c)
        g_screen = _cmi_gradient(screen)
        g_future = _cmi_gradient(future.reshape(-1, *future.shape[2:])).reshape(future.shape)
        moved = np.moveaxis(past, 1, -1)
        g_past = np.moveaxis(_cmi_gradient(moved.reshape(past.shape[0], -1, past.shape[1])).reshape(moved.shape), -1, 1)
        jac = np.empty((3, *c.shape))
        jac[0] = self.p[:, None] * np.einsum("pa,pb,abg->pg", self.plus, self.minus, g_screen)
        jac[1] = np.einsum("gxsv,pxq,qs,qv->pg", g_future, self.edges, self.minus, c) + np.einsum(
            "gxsv,pg,pxq,qs->qv", g_future, c, self.edges, self.minus
        )
        jac[2] = np.einsum("agxv,pa,pxq,qv->pg", g_past, self.plus, self.edges, c) + np.einsum(
            "agxv,pa,pg,pxq->qv", g_past, self.plus, c, self.edges
        )
        return jac

    def markov_residual(self, c: np.ndarray) -> float:
        """Largest constraint conditional mutual information (bits)."""
        return float(self.constraint_informations(c).max())

    def objective(self, c: np.ndarray, kind: str) -> tuple[float, np.ndarray]:
        """``I[(S+, S-) : G]`` or ``H[G]`` and its gradient in ``c``."""
        joint_g = self.p[:, None] * c
        p_g = joint_g.sum(axis=0)
        log_g = np.log2(np.maximum(p_g, 1e-300))
        h_g = float(-(p_g * log_g).sum())
        grad = -self.p[:, None] * (log_g[None, :] + 1.0 / _LN2)
        if kind == "exact":
            return h_g, grad
        log_joint = np.log2(np.maximum(joint_g, 1e-300))
        h_joint = float(-(joint_g * log_joint).sum())
        h_pairs = float(-(self.p * np.log2(self.p)).sum())
        grad += self.p[:, None] * (log_joint + 1.0 / _LN2)
        return h_g + h_pairs - h_joint, grad

    def seeds(self, k: int) -> list[np.ndarray]:
        """Channels of the generator states ``S+``, ``S-`` and ``(S+, S-)`` that fit in ``k`` states."""
        out = []
        for labels in (self.plus, self.minus, np.eye(len(self.pairs))):
            if labels.shape[1] <= k:
                c = np.zeros((len(self.pairs), k))
                c[:, : labels.shape[1]] = labels
                out.append(c)
        return out


def _entropy_bits(p: np.ndarray) -> float:
    p = p[p > 0.0]
    return float(-(p * np.log2(p)).sum())


def _conditional_mutual_information(joint: np.ndarray) -> float:
    """``I[a : b | c]`` in bits for a joint indexed ``[a, b, c]``."""
    return max(
        0.0,
        _entropy_bits(joint.sum(axis=1))
        + _entropy_bits(joint.sum(axis=0))
        - _entropy_bits(joint)
        - _entropy_bits(joint.sum(axis=(0, 1))),
    )


def _cmi_gradient(joint: np.ndarray) -> np.ndarray:
    """Gradient of ``I[a : b | c]`` (bits) in the entries of a joint indexed ``[a, b, c]``."""

    def log(x: np.ndarray) -> np.ndarray:
        return np.log2(np.maximum(x, 1e-300))

    return (
        log(joint)
        + log(joint.sum(axis=(0, 1)))[None, None, :]
        - log(joint.sum(axis=1))[:, None, :]
        - log(joint.sum(axis=0))[None, :, :]
    )


def _unconstrained_channel(
    problem: _LiftedProblem,
    joint: dict[tuple[Hashable, Hashable], float],
    kind: str,
    *,
    niter: int | None,
    maxiter: int,
    polish: float | bool,
    backend: str,
    cutoff: float,
    rng: np.random.Generator,
) -> np.ndarray | None:
    """Channel of the ordinary (dit) Wyner or exact optimum, as a start; ``None`` if that optimizer fails."""
    optimizer = _optimize_wyner_common_information if kind == "wyner" else _optimize_exact_common_information
    try:
        channel, _value = _auxiliary_state_channel(
            joint,
            optimizer=optimizer,
            optimizer_name=f"{kind} common information",
            bound=None,
            niter=niter,
            maxiter=maxiter,
            polish=polish,
            backend=backend,
            cutoff=cutoff,
            rng=rng,
        )
    except StochasticValidationError:
        return None
    states = sorted({state for row in channel.values() for state in row}, key=repr)
    return np.array([[channel[pair].get(state, 0.0) for state in states] for pair in problem.pairs])


def _optimize(
    problem: _LiftedProblem,
    kind: str,
    k: int,
    starts: int,
    maxiter: int,
    tol: float,
    rng: np.random.Generator,
    extra: list[np.ndarray],
):
    n = len(problem.pairs)
    shape = (n, k)
    row_sums = {
        "type": "eq",
        "fun": lambda z: z.reshape(shape).sum(axis=1) - 1.0,
        "jac": lambda z: np.kron(np.eye(n), np.ones((1, k))),
    }

    def fun(z: np.ndarray) -> tuple[float, np.ndarray]:
        value, grad = problem.objective(z.reshape(shape), kind)
        return value, grad.ravel()

    def solve(c0: np.ndarray, closure_tol: float, screen_tol: float, support: np.ndarray | None = None) -> np.ndarray:
        bounds_vec = np.array([screen_tol, closure_tol, closure_tol])
        bounds = [
            (0.0, 1.0) if keep else (0.0, 0.0) for keep in (np.ones(c0.size, bool) if support is None else support)
        ]
        informations = {
            "type": "ineq",
            "fun": lambda z: bounds_vec - problem.constraint_informations(z.reshape(shape)),
            "jac": lambda z: -problem.constraint_jacobian(z.reshape(shape)).reshape(3, -1),
        }
        result = minimize(
            fun,
            c0.ravel(),
            jac=True,
            method="SLSQP",
            bounds=bounds,
            constraints=[informations, row_sums],
            options={"maxiter": maxiter, "ftol": 1e-12},
        )
        c = np.clip(result.x.reshape(shape), 0.0, None)
        return c / c.sum(axis=1, keepdims=True)

    # Continuation to a coarse tolerance: the plain common-information problem first, then ever tighter closure.
    coarse = max(tol, _COARSE_TOL)
    schedule = [1e3, *np.geomspace(1e-1, coarse, max(2, int(np.ceil(np.log10(1e-1 / coarse) / 2)) + 1))]
    fixed = problem.seeds(k) + extra
    candidates = [(problem.objective(c0, kind)[0], problem.markov_residual(c0), c0) for c0 in fixed]
    for c0 in fixed + [rng.dirichlet(np.ones(k), size=n) for _ in range(starts)]:
        c = c0
        for closure_tol in schedule:
            c = solve(c, closure_tol, coarse)
        candidates.append((problem.objective(c, kind)[0], problem.markov_residual(c), c))

    # Refine the extra seeds and the best coarse candidates to the final tolerance on their own support
    # (entries near zero make the log-gradients ill-conditioned); values are reported at the refined channels.
    coarse_feasible = sorted((cand for cand in candidates if cand[1] <= 2 * coarse), key=lambda cand: cand[0])
    pool = [(problem.markov_residual(c0), c0) for c0 in extra] + [(r, c) for _v, r, c in coarse_feasible[:_REFINE]]
    refined = []
    for residual, c in pool:
        # Step well below the acceptance threshold so platform-level rounding cannot tip a candidate over it.
        for target in sorted({tol, tol / 10, min(tol, _REFINE_TOL)}, reverse=True):
            if residual <= 2 * target:
                continue
            support = c > _SUPPORT_CUTOFF
            trimmed = np.where(support, c, 0.0)
            tightened = solve(trimmed / trimmed.sum(axis=1, keepdims=True), target, target, support.ravel())
            if problem.markov_residual(tightened) < residual:
                c, residual = tightened, problem.markov_residual(tightened)
        refined.append((problem.objective(c, kind)[0], residual, c))
    return refined, candidates


def lifted_model(
    bidir: BidirectionalEpsilonMachine,
    *,
    objective: str,
    bound: int | None,
    starts: int,
    niter: int | None,
    maxiter: int,
    polish: float | bool,
    backend: str,
    markov_atol: float,
    reproduction_atol: float,
    cutoff: float,
    rng: np.random.Generator | None,
) -> WynerGenerativeModel | MinimalGenerativeModel:
    """Lifted Wyner (``objective="wyner"``) or exact (``"exact"``) generative model; see the module docstring.

    Backs ``wyner_generative_model`` and ``minimal_generative_model``.
    """
    if cutoff < 0.0:
        raise ValueError("cutoff must be nonnegative")
    model_cls = WynerGenerativeModel if objective == "wyner" else MinimalGenerativeModel
    measure = "wyner_common_information" if objective == "wyner" else "exact_common_information"
    name = f"{'Wyner' if objective == 'wyner' else 'minimal'} generative model"

    joint = _normalized_joint_distribution(bidir, cutoff=cutoff)
    problem = _LiftedProblem(bidir, joint)
    if problem.plus.shape[1] <= 1 or problem.minus.shape[1] <= 1:
        channel = {pair: {"G0": 1.0} for pair in joint}
        value, residual = 0.0, 0.0
    elif (matching := _matching_support_channel(joint)) is not None:
        channel = matching
        value, residual = _entropy_bits(problem.p), 0.0
    else:
        rng = rng or np.random.default_rng()
        unconstrained = _unconstrained_channel(
            problem,
            joint,
            objective,
            niter=niter,
            maxiter=maxiter,
            polish=polish,
            backend=backend,
            cutoff=cutoff,
            rng=rng,
        )
        k = int(bound) if bound is not None else len(problem.pairs)
        refined, candidates = _optimize(problem, objective, k, starts, maxiter, markov_atol / 2, rng, [])
        if unconstrained is not None:
            # The unconstrained optimum gets its own pass at its own size (capped by an explicit bound).
            seed = unconstrained
            if bound is not None and seed.shape[1] > k:
                seed = seed[:, np.sort(np.argsort(-(problem.p @ seed))[:k])] + 1e-12
                seed /= seed.sum(axis=1, keepdims=True)
            k_seed = max(seed.shape[1], k if bound is not None else 0)
            seed = np.pad(seed, ((0, 0), (0, k_seed - seed.shape[1])))
            seed_refined, seed_candidates = _optimize(
                problem, objective, k_seed, 0, maxiter, markov_atol / 2, rng, [seed]
            )
            refined += seed_refined
            candidates += seed_candidates
        # Prefer refined channels: looser candidates can undercut the true optimum by about sqrt(residual).
        feasible = [cand for cand in refined if cand[1] <= markov_atol] or [
            cand for cand in candidates if cand[1] <= markov_atol
        ]
        if not feasible:
            raise StochasticValidationError(
                f"{name}: no start reached the lifted constraints within {markov_atol:g} bits "
                f"(best residual {min(cand[1] for cand in candidates):.3g})"
            )
        value, residual, c = min(feasible, key=lambda cand: cand[0])
        channel = _channel_dict(problem, c, cutoff)

    model = _model_from_channel(
        bidir,
        joint,
        channel,
        model_cls=model_cls,
        model_name=name,
        measure_kwargs={measure: value, "markov_residual": residual},
        cutoff=cutoff,
    )
    reference = bidir.forward_machine
    error = _reproduction_error(model, reference, max_length=_reproduction_length(reference))
    if error > reproduction_atol:
        raise StochasticValidationError(
            f"{name} satisfies the lifted constraints to {residual:.3g} bits but misses the source "
            f"process by {error:.3g} in word probability"
        )
    return model


def _channel_dict(
    problem: _LiftedProblem, c: np.ndarray, cutoff: float
) -> dict[tuple[Hashable, Hashable], dict[Hashable, float]]:
    active = np.flatnonzero((problem.p[:, None] * c).sum(axis=0) > cutoff)
    channel = {}
    for i, pair in enumerate(problem.pairs):
        row = {f"G{rank}": float(c[i, g]) for rank, g in enumerate(active) if c[i, g] > cutoff}
        total = sum(row.values())
        channel[pair] = {state: prob / total for state, prob in row.items()}
    return channel


def partition_residual(
    bidir: BidirectionalEpsilonMachine,
    joint: dict[tuple[Hashable, Hashable], float],
    channel: dict[tuple[Hashable, Hashable], dict[Hashable, float]],
) -> float:
    """Largest constraint conditional mutual information (bits) of a channel given as a dict."""
    problem = _LiftedProblem(bidir, joint)
    states = sorted({state for row in channel.values() for state in row}, key=repr)
    c = np.array([[channel[pair].get(state, 0.0) for state in states] for pair in problem.pairs])
    return problem.markov_residual(c)


def lifted_partition_channel(
    bidir: BidirectionalEpsilonMachine,
    joint: dict[tuple[Hashable, Hashable], float],
) -> tuple[dict[tuple[Hashable, Hashable], dict[Hashable, float]], float]:
    """Least-entropy deterministic ``W = f(S+, S-)`` satisfying all three constraints, and ``H[W]``.

    Exhaustive over set partitions of the joint states when there are at most
    ``_EXHAUSTIVE_PAIRS`` of them; otherwise greedy, merging blocks of the
    identity partition (always feasible: its lift is the bidirectional machine)
    while the constraints hold, taking the largest entropy reduction each step.
    """
    problem = _LiftedProblem(bidir, joint)
    n = len(problem.pairs)

    def entropy(labels: np.ndarray) -> float:
        return _entropy_bits(np.bincount(labels, weights=problem.p))

    def feasible(labels: np.ndarray) -> bool:
        return problem.markov_residual(np.eye(labels.max() + 1)[labels]) <= _EXACT_ATOL

    if n <= _EXHAUSTIVE_PAIRS:
        best = min(
            (labels for labels in _set_partitions(n) if feasible(labels)),
            key=entropy,
        )
    else:
        best = np.arange(n)
        while True:
            merges = []
            for a in range(best.max() + 1):
                for b in range(a + 1, best.max() + 1):
                    merged = np.where(best == b, a, best)
                    merged = np.unique(merged, return_inverse=True)[1]
                    if feasible(merged):
                        merges.append((entropy(merged), merged))
            if not merges:
                break
            best = min(merges, key=lambda item: item[0])[1]

    channel = {pair: {f"G{int(best[i])}": 1.0} for i, pair in enumerate(problem.pairs)}
    return channel, entropy(best)


def _set_partitions(n: int):
    """Restricted-growth strings of length ``n``: every set partition of ``range(n)`` once."""
    labels = np.zeros(n, dtype=int)

    def extend(i: int, blocks: int):
        if i == n:
            yield labels.copy()
            return
        for block in range(blocks + 1):
            labels[i] = block
            yield from extend(i + 1, max(blocks, block + 1))

    yield from extend(1, 1) if n else iter(())


__all__: list[str] = []
