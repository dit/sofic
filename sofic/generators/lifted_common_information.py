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

from collections.abc import Hashable, Sequence
from typing import TYPE_CHECKING

import numpy as np
from scipy.optimize import minimize
from scipy.special import entr

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
# I[a : b | c] = H[a, c] + H[b, c] - H[a, b, c] - H[c]
_CMI_SIGNS = np.array([1.0, 1.0, -1.0, -1.0])
_COARSE_TOL = 5e-8
_REFINE = 5
_SUPPORT_CUTOFF = 1e-8
_REFINE_TOL = 1e-12
_EXACT_ATOL = 1e-10
_PERTURB_SCALES = (0.05, 0.2)
_PERTURB_COPIES = 2
_MIX_WEIGHTS = (0.5, 0.8)
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
        # The edge law flattened both ways, and p(s+, s- | pair), for the matrix products below.
        self._edges_rows = self.edges.reshape(n * len(symbols), n)  # [(p, x), q]
        self._edges_cols = self.edges.reshape(n, len(symbols) * n)  # [p, (x, q)]
        self._pair_labels = (self.plus[:, :, None] * self.minus[:, None, :]).reshape(n, -1)  # [p, (a, b)]
        self._cached: tuple[tuple[int, ...], bytes, dict[str, np.ndarray]] | None = None

    def _products(self, c: np.ndarray) -> dict[str, np.ndarray]:
        """The tensors of :meth:`tensors` and the intermediate products, for the last ``c`` seen.

        SLSQP evaluates the constraints and their Jacobian at the same point, so the last one is kept.
        """
        key = c.tobytes()
        if self._cached is not None and self._cached[0] == c.shape and self._cached[1] == key:
            return self._cached[2]
        n, k = c.shape
        n_plus, n_minus, n_sym = self.plus.shape[1], self.minus.shape[1], self.edges.shape[1]
        plus_c = (self.plus[:, :, None] * c[:, None, :]).reshape(n, n_plus * k)  # [p, (a, g)]
        minus_c = (self.minus[:, :, None] * c[:, None, :]).reshape(n, n_minus * k)  # [q, (s, v)]
        forward = self._edges_rows @ c  # [(p, x), v]
        past = (plus_c.T @ forward.reshape(n, n_sym * k)).reshape(n_plus, k, n_sym * k)  # [a, g, (x, v)]
        products = {
            "plus_c": plus_c,
            "minus_c": minus_c,
            "forward": forward,
            # The three joints of constraint_informations, each indexed [a, b, c] for I[a : b | c].
            "screen": (self._pair_labels.T @ (self.p[:, None] * c)).reshape(n_plus, n_minus, k),
            "future": (c.T @ (self._edges_rows @ minus_c).reshape(n, -1)).reshape(k * n_sym, n_minus, k),
            "past": past.transpose(0, 2, 1),
        }
        self._cached = (c.shape, key, products)
        return products

    def tensors(self, c: np.ndarray) -> tuple[np.ndarray, ...]:
        """``p(s+, s-, g)``, ``p(g_0, x_0, s-_1, g_1)`` and ``p(s+_0, g_0, x_0, g_1)``."""
        products = self._products(c)
        k = c.shape[1]
        n_plus, n_minus, n_sym = products["screen"].shape[0], products["screen"].shape[1], self.edges.shape[1]
        future = products["future"].reshape(k, n_sym, n_minus, k)
        past = products["past"].transpose(0, 2, 1).reshape(n_plus, k, n_sym, k)
        return products["screen"], future, past

    def constraint_informations(self, c: np.ndarray) -> np.ndarray:
        """``I[S+ : S- | G]``, ``I[G_0 X_0 : S-_1 | G_1]`` and ``I[S+_0 : X_0 G_1 | G_0]`` in bits."""
        products = self._products(c)
        return _conditional_mutual_informations((products["screen"], products["future"], products["past"]))

    def constraint_jacobian(self, c: np.ndarray) -> np.ndarray:
        """Gradients of :meth:`constraint_informations` in ``c``, shape ``(3, *c.shape)``."""
        products = self._products(c)
        screen = products["screen"]
        n, k = c.shape
        n_plus, n_minus, n_sym = screen.shape[0], screen.shape[1], self.edges.shape[1]
        g_screen = _cmi_gradient(screen)
        g_future = _cmi_gradient(products["future"]).reshape(k * n_sym, n_minus * k)  # [(g, x), (s, v)]
        g_past = _cmi_gradient(products["past"]).transpose(0, 2, 1).reshape(n_plus * k, n_sym * k)  # [(a, g), (x, v)]
        jac = np.empty((3, n, k))
        jac[0] = self.p[:, None] * (self._pair_labels @ g_screen.reshape(-1, k))
        # d/dc[p, g] through the c_0 factor of the future, then d/dc[q, v] through the c_1 factor.
        through_target = (g_future @ products["minus_c"].T).reshape(k, n_sym * n)  # [g, (x, q)]
        through_source = ((c.T @ self._edges_cols).reshape(k * n_sym, n).T @ g_future).reshape(n, n_minus, k)
        jac[1] = self._edges_cols @ through_target.T + np.einsum("qs,qsv->qv", self.minus, through_source)
        # Likewise for the past, whose c_1 factor sits inside the forward map.
        through_forward = (products["forward"].reshape(n, n_sym * k) @ g_past.T).reshape(n, n_plus, k)
        through_plus = (products["plus_c"] @ g_past).reshape(n * n_sym, k)
        jac[2] = np.einsum("pa,pag->pg", self.plus, through_forward) + self._edges_rows.T @ through_plus
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
    return float(entr(np.maximum(p, 0.0)).sum()) / _LN2


def _conditional_mutual_informations(joints: Sequence[np.ndarray]) -> np.ndarray:
    """``I[a : b | c]`` in bits for each joint indexed ``[a, b, c]``, from one pass over all the entropies."""
    parts = []
    for joint in joints:
        ac = joint.sum(axis=1)
        parts += [ac.ravel(), joint.sum(axis=0).ravel(), joint.ravel(), ac.sum(axis=0)]
    starts = np.cumsum([0] + [part.size for part in parts[:-1]])
    entropies = np.add.reduceat(entr(np.maximum(np.concatenate(parts), 0.0)), starts).reshape(-1, 4)
    return np.maximum(entropies @ _CMI_SIGNS / _LN2, 0.0)


def _cmi_gradient(joint: np.ndarray) -> np.ndarray:
    """Gradient of ``I[a : b | c]`` (bits) in the entries of a joint indexed ``[a, b, c]``."""

    def log(x: np.ndarray) -> np.ndarray:
        return np.log2(np.maximum(x, 1e-300))

    ac = joint.sum(axis=1)
    return log(joint) + log(ac.sum(axis=0))[None, None, :] - log(ac)[:, None, :] - log(joint.sum(axis=0))[None, :, :]


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
    more_starts: Sequence[np.ndarray] = (),
    generator_seeds: bool = True,
):
    """Continuation from the generator seeds (if ``generator_seeds``), the ``extra`` channels, ``more_starts``
    and ``starts`` random channels; the ``extra`` channels are also kept and refined as they are."""
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
    fixed = (problem.seeds(k) if generator_seeds else []) + extra
    candidates = [(problem.objective(c0, kind)[0], problem.markov_residual(c0), c0) for c0 in fixed]
    for c0 in fixed + list(more_starts) + [rng.dirichlet(np.ones(k), size=n) for _ in range(starts)]:
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
            perturbed = [
                (1 - eps) * seed + eps * rng.dirichlet(np.ones(k_seed), size=len(problem.pairs))
                for eps in _PERTURB_SCALES
                for _ in range(_PERTURB_COPIES)
            ]
            seed_refined, seed_candidates = _optimize(
                problem, objective, k_seed, 0, maxiter, markov_atol / 2, rng, [seed], perturbed, generator_seeds=False
            )
            refined += seed_refined
            candidates += seed_candidates
            # Time-sharing mixtures: an independent coin picks the unconstrained channel or an eps-machine
            # channel; disjoint state labels keep the mixture separating.
            for machine in (problem.plus, problem.minus, np.eye(len(problem.pairs))):
                width = k_seed + machine.shape[1]
                if bound is not None and width > k:
                    continue
                mixtures = [np.hstack([lam * seed, (1 - lam) * machine]) for lam in _MIX_WEIGHTS]
                mix_refined, mix_candidates = _optimize(
                    problem, objective, width, 0, maxiter, markov_atol / 2, rng, [], mixtures, generator_seeds=False
                )
                refined += mix_refined
                candidates += mix_candidates
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
