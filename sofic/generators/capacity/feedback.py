r"""Feedback capacity of finite unifilar channels (ε-transducers).

An ε-transducer is a unifilar finite-state channel: ``(s, x, y)`` determines
the next causal state. With feedback the encoder sees past outputs, hence knows
the causal state, and the feedback capacity is the limit of normalized directed
information :cite:`Permuter2008,Yang2005`. Feedback capacity of finite-state
channels is not computable in general :cite:`Grigorescu2024`, so every method
here returns a bracket:

* :func:`qgraph_bounds` -- for a Q-graph, the single-letter upper bound
  :math:`\max I(X, S; Y \mid Q)` over stationary joints :cite:`Sabag2017`
  (a convex program :cite:`Sabag2020`) and the same objective restricted to
  BCJR-invariant inputs as an achievable lower bound.
* :func:`value_iteration_estimate` -- the belief-state dynamic program of
  :cite:`Permuter2008,Yang2005` on a gridded simplex (an uncertified estimate).
* :func:`duality_upper_bound` -- the duality bound with a Q-graph test
  distribution, which for unifilar channels is a finite Markov decision
  process :cite:`Huleihel2025`.
"""

from __future__ import annotations

import warnings
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from itertools import product
from math import log
from typing import Any, Literal

import networkx as nx
import numpy as np
from scipy.optimize import minimize

from sofic.generators.capacity._belief import (
    SimplexGrid,
    relative_value_iteration,
    transition_operator,
)
from sofic.generators.capacity._channel import ChannelTensors, channel_tensors, require_strongly_connected
from sofic.generators.capacity.qgraph import QGraph, coupled_kernel
from sofic.generators.capacity.results import CapacityBounds

_LN2 = log(2.0)
_TINY = 1e-300

FeedbackMethod = Literal["qgraph", "value_iteration", "duality", "all"]


def as_tensors(channel: Any) -> ChannelTensors:
    """Return ``channel`` as :class:`ChannelTensors`, checking strong connectivity."""
    tensors = channel if isinstance(channel, ChannelTensors) else channel_tensors(channel)
    require_strongly_connected(tensors)
    return tensors


# -- Q-graph bounds -----------------------------------------------------------


@dataclass(frozen=True, eq=False)
class _QGraphProgram:
    """Linear maps of the joint ``d[s, q, x]`` used by both Q-graph bounds."""

    n_s: int
    n_q: int
    n_x: int
    n_y: int
    table: np.ndarray
    allowed: np.ndarray
    stationarity: np.ndarray
    occupancy: np.ndarray
    outputs: np.ndarray
    context: np.ndarray
    context_rep: np.ndarray
    noise: np.ndarray
    successor: np.ndarray

    @classmethod
    def build(cls, tensors: ChannelTensors, qgraph: QGraph) -> _QGraphProgram:
        table = qgraph.aligned(tensors.outputs)
        n_s, n_x, n_y = tensors.P.shape
        n_q = table.shape[0]
        nv = n_s * n_q * n_x
        kernel = coupled_kernel(tensors, table).reshape(nv, n_s * n_q)
        occupancy = np.zeros((n_s * n_q, nv))
        for s, q, x in product(range(n_s), range(n_q), range(n_x)):
            occupancy[s * n_q + q, (s * n_q + q) * n_x + x] = 1.0
        outputs = np.zeros((n_q * n_y, nv))
        successor = np.zeros((n_s * n_q * n_y, nv))
        for s, q, x, y in product(range(n_s), range(n_q), range(n_x), range(n_y)):
            v = (s * n_q + q) * n_x + x
            outputs[q * n_y + y, v] = tensors.P[s, x, y]
            if tensors.P[s, x, y] > 0:
                successor[(tensors.F[s, x, y] * n_q + q) * n_y + y, v] = tensors.P[s, x, y]
        context = occupancy.reshape(n_s, n_q, nv).sum(axis=0)
        noise = np.repeat(tensors.conditional_entropy()[:, None, :], n_q, axis=1).ravel()
        allowed = np.repeat(tensors.allowed[:, None, :], n_q, axis=1).ravel()
        return cls(
            n_s,
            n_q,
            n_x,
            n_y,
            table,
            allowed,
            kernel.T - occupancy,
            occupancy,
            outputs,
            context,
            np.repeat(context, n_y, axis=0),
            noise,
            successor,
        )

    @property
    def n_vars(self) -> int:
        return self.n_s * self.n_q * self.n_x

    def information(self, d: np.ndarray) -> float:
        """Return ``I(X, S; Y | Q) = H(Y | Q) - H(Y | X, S)`` in bits."""
        n = self.outputs @ d
        m = self.context_rep @ d
        with np.errstate(divide="ignore", invalid="ignore"):
            terms = np.where(n > 0, n * np.log2(np.maximum(n, _TINY) / np.maximum(m, _TINY)), 0.0)
        return float(-terms.sum() - self.noise @ d)

    def gradient(self, d: np.ndarray) -> np.ndarray:
        n = np.maximum(self.outputs @ d, 1e-15)
        m = np.maximum(self.context_rep @ d, 1e-15)
        return -(self.outputs.T @ np.log2(n / m)) - self.noise

    def bcjr_residual(self, d: np.ndarray) -> np.ndarray:
        """Return ``P(s+, q, y) P(q+) - P(q, y) P(s+, q+)`` with ``q+ = g(q, y)``."""
        b = (self.successor @ d).reshape(self.n_s, self.n_q, self.n_y)
        n = (self.outputs @ d).reshape(self.n_q, self.n_y)
        pi = (self.occupancy @ d).reshape(self.n_s, self.n_q)
        pi_q = pi.sum(axis=0)
        return (b * pi_q[self.table][None] - n[None] * pi[:, self.table]).ravel()

    def policy(self, d: np.ndarray) -> np.ndarray:
        """Return ``P(x | s, q)``; unvisited ``(s, q)`` get the uniform admissible input."""
        joint = d.reshape(self.n_s, self.n_q, self.n_x)
        mask = self.allowed.reshape(self.n_s, self.n_q, self.n_x).astype(float)
        mass = joint.sum(axis=2, keepdims=True)
        uniform = mask / mask.sum(axis=2, keepdims=True)
        return np.where(mass > 1e-12, joint / np.maximum(mass, _TINY), uniform)

    def random_start(self, rng: np.random.Generator, tensors: ChannelTensors) -> np.ndarray:
        mask = self.allowed.reshape(self.n_s, self.n_q, self.n_x)
        policy = rng.dirichlet(np.ones(self.n_x), size=(self.n_s, self.n_q)) * mask
        policy /= policy.sum(axis=2, keepdims=True)
        pi = _stationary(self._chain(policy, tensors))
        return (pi.reshape(self.n_s, self.n_q, 1) * policy).ravel()

    def _chain(self, policy: np.ndarray, tensors: ChannelTensors) -> np.ndarray:
        kernel = coupled_kernel(tensors, self.table)
        n = self.n_s * self.n_q
        return np.einsum("sqx,sqxtr->sqtr", policy, kernel).reshape(n, n)


def _stationary(matrix: np.ndarray) -> np.ndarray:
    from sofic.generators.stationary import stationary_distribution_from_transition

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return stationary_distribution_from_transition(matrix)


def _has_cvxpy() -> bool:
    try:
        import cvxpy  # noqa: F401
    except ImportError:
        return False
    return True


def qgraph_upper_bound(
    tensors: ChannelTensors,
    qgraph: QGraph,
    *,
    solver: Literal["auto", "cvxpy", "scipy"] = "auto",
) -> tuple[float, np.ndarray]:
    r"""Return :math:`\max I(X, S; Y \mid Q)` over stationary joints and the optimal joint.

    For any Q-graph this upper-bounds the feedback capacity of a strongly
    connected unifilar channel :cite:`Sabag2017`. Optimizing over the joint
    ``d[s, q, x]`` with linear stationarity constraints makes the problem
    convex :cite:`Sabag2020`. ``solver="auto"`` uses cvxpy when installed;
    its solution, plus random stationary starts, is polished with scipy's
    SLSQP and the best feasible point is kept. The value is accurate to the
    optimizer's tolerance.
    """
    program = _QGraphProgram.build(tensors, qgraph)
    use_cvxpy = solver == "cvxpy" or (solver == "auto" and _has_cvxpy())
    starts = [_solve_upper_cvxpy(program)] if use_cvxpy else []
    d = _solve_upper_scipy(program, tensors, starts)
    d = np.where(program.allowed, np.maximum(d, 0.0), 0.0)
    d /= d.sum()
    return program.information(d), d


def _solve_upper_cvxpy(program: _QGraphProgram) -> np.ndarray:
    import cvxpy as cp

    d = cp.Variable(program.n_vars, nonneg=True)
    constraints = [program.stationarity @ d == 0, cp.sum(d) == 1]
    if not program.allowed.all():
        constraints.append(d[np.flatnonzero(~program.allowed)] == 0)
    conditional_output = -cp.sum(cp.rel_entr(program.outputs @ d, program.context_rep @ d)) / _LN2
    problem = cp.Problem(cp.Maximize(conditional_output - program.noise @ d), constraints)
    problem.solve()
    if d.value is None:
        raise RuntimeError(f"cvxpy failed to solve the Q-graph upper bound ({problem.status})")
    return np.asarray(d.value, dtype=float)


def _independent_rows(matrix: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return an equivalent full-row-rank system for ``matrix @ d == target``."""
    u, sigma, vt = np.linalg.svd(matrix, full_matrices=False)
    rank = int((sigma > 1e-10 * sigma[0]).sum())
    return vt[:rank], (u[:, :rank].T @ target) / sigma[:rank]


def _solve_upper_scipy(program: _QGraphProgram, tensors: ChannelTensors, starts: list[np.ndarray]) -> np.ndarray:
    nv = program.n_vars
    equality = np.vstack([program.stationarity, np.ones((1, nv))])
    target = np.zeros(equality.shape[0])
    target[-1] = 1.0
    equality, target = _independent_rows(equality, target)
    bounds = list(zip(np.zeros(nv), np.where(program.allowed, 1.0, 0.0), strict=True))
    rng = np.random.default_rng(0)
    starts = [np.clip(start, 0.0, 1.0) for start in starts] + [program.random_start(rng, tensors) for _ in range(3)]
    best, best_value = starts[0], -np.inf
    for start in starts:
        result = minimize(
            lambda d: -program.information(d),
            start,
            jac=lambda d: -program.gradient(d),
            method="SLSQP",
            bounds=bounds,
            constraints=[{"type": "eq", "fun": lambda d: equality @ d - target, "jac": lambda d: equality}],
            options={"ftol": 1e-15, "maxiter": 3000},
        )
        value = -float(result.fun)
        if value > best_value:
            best, best_value = np.asarray(result.x, dtype=float), value
    return best


def _encoder_kernel(tensors: ChannelTensors, table: np.ndarray) -> np.ndarray:
    """Return ``K[s, q, x, y, s', q']`` for an encoder that sees ``(s, q)``."""
    n_s, n_x, n_y = tensors.P.shape
    n_q = table.shape[0]
    kernel = np.zeros((n_s, n_q, n_x, n_y, n_s, n_q))
    for s, x, y in zip(*np.nonzero(tensors.P > 0), strict=True):
        kernel[s, np.arange(n_q), x, y, tensors.F[s, x, y], table[:, y]] = tensors.P[s, x, y]
    return kernel


def encoder_rate_bounds(
    tensors: ChannelTensors, qgraph: QGraph, policy: np.ndarray, *, block: int = 16
) -> tuple[float, float] | None:
    r"""Bracket the directed-information rate of the graph-based encoder ``policy[s, q, x]``.

    With feedback the encoder knows the causal state ``s`` and the context
    ``q``, so any ``P(x | s, q)`` is a valid causal input and its rate
    :math:`\bar h(Y) - \mathbb{E}_\pi H(Y \mid X, S)` is achievable
    :cite:`Permuter2008,Sabag2020`. The lower end uses
    :math:`H[Y_n \mid Y_{0:n}, Z_0]` with ``n = block``, so it is a certified
    lower bound on the feedback capacity. Returns ``None`` when the encoder's
    ``(S, Q)`` chain has more than one closed class.
    """
    from sofic.generators.capacity._rates import driven_rate_bounds

    table = qgraph.aligned(tensors.outputs)
    n_z = tensors.n_states * qgraph.n_nodes
    kernel = _encoder_kernel(tensors, table)
    symbol_tensors = np.einsum("sqx,sqxytr->ysqtr", policy, kernel).reshape(tensors.n_outputs, n_z, n_z)
    noise = np.einsum("sqx,sx->sq", policy, tensors.conditional_entropy()).ravel()
    return driven_rate_bounds(symbol_tensors, noise, block)


def _masked_softmax(logits: np.ndarray, mask: np.ndarray) -> np.ndarray:
    shifted = np.where(mask, logits, -np.inf)
    shifted = shifted - shifted.max(axis=-1, keepdims=True)
    weights = np.where(mask, np.exp(shifted), 0.0)
    return weights / weights.sum(axis=-1, keepdims=True)


def qgraph_lower_bound(
    tensors: ChannelTensors,
    qgraph: QGraph,
    *,
    start_joint: np.ndarray | None = None,
    restarts: int = 4,
    seed: int | None = 0,
    block: int | None = None,
    final_block: int | None = None,
    bcjr_tol: float = 1e-7,
) -> tuple[float, np.ndarray | None]:
    r"""Return a certified achievable feedback rate for an encoder on ``qgraph``.

    Optimizes the graph-based encoder :math:`P(x \mid s, q)` (L-BFGS-B on
    softmax logits, started from ``start_joint`` -- typically the upper-bound
    optimizer -- and ``restarts`` random policies) for the lower end of
    :func:`encoder_rate_bounds` at ``block``, then re-evaluates the winner at
    ``final_block`` (by default the largest ``n`` with :math:`|Y|^n \le 2^8`
    and :math:`2^{16}`, respectively). The upper-bound optimizer's own policy
    is evaluated too. If the winning input is also aperiodic and
    BCJR-invariant, :math:`S^+ - Q^+ - (Q, Y)`, the single-letter rate
    :math:`I(X, S; Y \mid Q)` is achievable as well :cite:`Sabag2017,Sabag2020`
    and the larger value is returned. Returns ``(rate, policy)``.
    """
    from sofic.generators.capacity._rates import default_block

    block = default_block(tensors.n_outputs, 8) if block is None else block
    final_block = default_block(tensors.n_outputs, 16) if final_block is None else final_block
    program = _QGraphProgram.build(tensors, qgraph)
    mask = program.allowed.reshape(program.n_s, program.n_q, program.n_x)
    rng = np.random.default_rng(seed)

    def objective(z: np.ndarray, n: int) -> float:
        bounds = encoder_rate_bounds(tensors, qgraph, _masked_softmax(z.reshape(mask.shape), mask), block=n)
        return 1.0 if bounds is None else -bounds[0]

    starts = []
    if start_joint is not None:
        starts.append(np.log(np.maximum(program.policy(np.maximum(start_joint, 0.0)), 1e-9)).ravel())
    starts += [rng.normal(size=mask.size) for _ in range(restarts)]
    best_z, best_value = None, np.inf
    for z0 in starts:
        result = minimize(objective, z0, args=(block,), method="L-BFGS-B", options={"maxiter": 500})
        if result.fun < best_value:
            best_z, best_value = result.x, float(result.fun)
    if best_z is None:
        return 0.0, None
    candidates = [_masked_softmax(best_z.reshape(mask.shape), mask)]
    if start_joint is not None:
        candidates.append(_clean_policy(program.policy(np.maximum(start_joint, 0.0))))
    best_rate, best_policy = 0.0, None
    for policy in candidates:
        final = encoder_rate_bounds(tensors, qgraph, policy, block=final_block)
        rate = 0.0 if final is None else final[0]
        single_letter = _bcjr_rate(program, tensors, policy, bcjr_tol)
        if single_letter is not None:
            rate = max(rate, single_letter)
        if best_policy is None or rate > best_rate:
            best_rate, best_policy = rate, policy
    return max(best_rate, 0.0), best_policy


def _clean_policy(policy: np.ndarray, threshold: float = 1e-7) -> np.ndarray:
    """Zero solver noise below ``threshold`` so near-deterministic inputs become deterministic."""
    cleaned = np.where(policy < threshold, 0.0, policy)
    return cleaned / cleaned.sum(axis=-1, keepdims=True)


def _bcjr_rate(program: _QGraphProgram, tensors: ChannelTensors, policy: np.ndarray, tol: float) -> float | None:
    """Return ``I(X, S; Y | Q)`` if ``policy`` is aperiodic and BCJR-invariant, else ``None``."""
    from sofic.generators.capacity._rates import unique_closed_class

    chain = program._chain(policy, tensors)
    members = unique_closed_class(chain)
    if members is None:
        return None
    support = nx.DiGraph()
    support.add_edges_from(zip(*np.nonzero(chain[np.ix_(members, members)] > 1e-12), strict=True))
    if not nx.is_aperiodic(support):
        return None
    joint = (_stationary(chain).reshape(program.n_s, program.n_q, 1) * policy).ravel()
    if np.max(np.abs(program.bcjr_residual(joint))) > tol:
        return None
    return program.information(joint)


@dataclass(frozen=True, eq=False)
class _QGraphOutcome:
    qgraph: QGraph
    upper: float
    upper_joint: np.ndarray
    lower: float
    lower_policy: np.ndarray | None


def qgraph_bounds(
    channel: Any,
    *,
    qgraphs: Iterable[QGraph] | None = None,
    max_order: int = 2,
    solver: Literal["auto", "cvxpy", "scipy"] = "auto",
    restarts: int = 4,
    seed: int | None = 0,
    block: int | None = None,
    final_block: int | None = None,
) -> CapacityBounds:
    """Bracket the feedback capacity with Q-graph bounds.

    Tries the Markov Q-graphs of orders ``0, ..., max_order`` plus any
    ``qgraphs`` and keeps the smallest upper and largest lower bound
    :cite:`Sabag2017,Sabag2020`.
    """
    tensors = as_tensors(channel)
    candidates = [QGraph.markov(tensors.outputs, k) for k in range(max_order + 1)]
    candidates += list(qgraphs or [])
    outcomes = []
    for qgraph in candidates:
        upper, upper_joint = qgraph_upper_bound(tensors, qgraph, solver=solver)
        lower, policy = qgraph_lower_bound(
            tensors,
            qgraph,
            start_joint=upper_joint,
            restarts=restarts,
            seed=seed,
            block=block,
            final_block=final_block,
        )
        outcomes.append(_QGraphOutcome(qgraph, upper, upper_joint, lower, policy))
    best_upper = min(outcomes, key=lambda o: o.upper)
    best_lower = max(outcomes, key=lambda o: o.lower)
    upper = min(best_upper.upper, tensors.trivial_upper_bound())
    return CapacityBounds(
        lower=min(max(best_lower.lower, 0.0), upper),
        upper=upper,
        method="qgraph",
        feedback=True,
        details={
            "upper_qgraph": best_upper.qgraph,
            "upper_joint": best_upper.upper_joint,
            "lower_qgraph": best_lower.qgraph,
            "lower_policy": best_lower.lower_policy,
            "per_qgraph": [(o.qgraph, o.lower, o.upper) for o in outcomes],
        },
    )


# -- belief-state dynamic program --------------------------------------------


def _input_grid(n_inputs: int, allowed: np.ndarray, resolution: int) -> np.ndarray:
    """Return all input laws on the admissible inputs with masses in ``1/resolution`` steps."""
    support = np.flatnonzero(allowed)
    grid = SimplexGrid.build(len(support), resolution).points
    laws = np.zeros((grid.shape[0], n_inputs))
    laws[:, support] = grid
    return laws


def value_iteration_estimate(
    channel: Any,
    *,
    belief_resolution: int = 100,
    action_resolution: int = 10,
    tol: float = 1e-9,
    max_iter: int = 20_000,
    max_grid_points: int = 100_000,
) -> CapacityBounds:
    r"""Estimate the feedback capacity by value iteration on the belief simplex.

    The dynamic program of :cite:`Permuter2008,Yang2005` has state
    :math:`\beta = P(S \mid y^t)`, action :math:`u(x \mid s)` and reward
    :math:`I(X, S; Y)` under :math:`\beta u P`. The belief simplex and the
    action set are gridded, so the returned bracket bounds the gain of the
    *discretized* problem only and is marked ``certified=False``; it converges
    to the capacity as both resolutions grow. ``details["policy"]`` holds the
    greedy input law at every grid belief.
    """
    tensors = as_tensors(channel)
    n_s, n_x, n_y = tensors.P.shape
    grid = SimplexGrid.build(n_s, belief_resolution, max_points=max_grid_points)
    per_state = [_input_grid(n_x, tensors.allowed[s], action_resolution) for s in range(n_s)]
    actions = np.array([np.stack(choice) for choice in product(*per_state)])
    n_b, n_a = grid.size, actions.shape[0]
    if n_b * n_a * n_s * n_x * n_y > 5e7:
        raise ValueError("belief/action grid too large; lower belief_resolution or action_resolution")

    noise = tensors.conditional_entropy()
    joint = np.einsum("bs,asx,sxy->basxy", grid.points, actions, tensors.P)
    p_y = joint.sum(axis=(2, 3))
    with np.errstate(divide="ignore", invalid="ignore"):
        h_y = -np.where(p_y > 0, p_y * np.log2(np.maximum(p_y, _TINY)), 0.0).sum(axis=2)
    rewards = h_y - np.einsum("bs,asx,sx->ba", grid.points, actions, noise)

    onehot = np.zeros((n_s, n_x, n_y, n_s))
    s_idx, x_idx, y_idx = np.nonzero(tensors.P > 0)
    onehot[s_idx, x_idx, y_idx, tensors.F[s_idx, x_idx, y_idx]] = 1.0
    unnormalized = np.einsum("basxy,sxyt->bayt", joint, onehot)
    queries = unnormalized / np.maximum(p_y[..., None], _TINY)
    queries[p_y <= 0] = 1.0 / n_s
    indices, weights = grid.interpolation(queries.reshape(-1, n_s))
    rows = (np.arange(n_a)[None, :, None] * n_b + np.arange(n_b)[:, None, None]) * np.ones((1, 1, n_y), dtype=int)
    operator = transition_operator(n_a * n_b, n_b, rows.ravel(), indices, weights, p_y.ravel())

    result = relative_value_iteration(rewards, operator, tol=tol, max_iter=max_iter)
    return CapacityBounds(
        lower=max(result.gain_lower, 0.0),
        upper=min(result.gain_upper, tensors.trivial_upper_bound()),
        method="value_iteration",
        feedback=True,
        certified=False,
        details={
            "beliefs": grid.points,
            "policy": actions[result.policy],
            "bias": result.bias,
            "iterations": result.iterations,
        },
    )


# -- duality upper bound ------------------------------------------------------


def _divergence_rewards(tensors: ChannelTensors, test: np.ndarray) -> np.ndarray:
    """Return ``D(P(. | x, s) || R(. | q))`` in bits as an ``(S, Q, X)`` array."""
    P = tensors.P[:, None, :, :]
    R = test[None, :, None, :]
    with np.errstate(divide="ignore", invalid="ignore"):
        terms = np.where(P > 0, P * np.log2(np.where(P > 0, P, 1.0) / np.maximum(R, _TINY)), 0.0)
    rewards = terms.sum(axis=3)
    return np.where(tensors.allowed[:, None, :], rewards, -np.inf)


def _duality_operator(tensors: ChannelTensors, table: np.ndarray):
    n_s, n_x, _n_y = tensors.P.shape
    n_q = table.shape[0]
    n = n_s * n_q
    kernel = coupled_kernel(tensors, table).reshape(n, n_x, n)
    from scipy.sparse import csr_matrix

    return csr_matrix(kernel.transpose(1, 0, 2).reshape(n_x * n, n))


def _softmax_rows(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    weights = np.exp(shifted)
    return weights / weights.sum(axis=1, keepdims=True)


def _test_from_joint(program_table: np.ndarray, tensors: ChannelTensors, joint: np.ndarray | None) -> np.ndarray:
    n_q, n_y = program_table.shape[0], tensors.n_outputs
    if joint is None:
        return np.full((n_q, n_y), 1.0 / n_y)
    d = joint.reshape(tensors.n_states, n_q, tensors.n_inputs)
    outputs = np.einsum("sqx,sxy->qy", d, tensors.P)
    mass = outputs.sum(axis=1, keepdims=True)
    test = np.where(mass > 1e-12, outputs / np.maximum(mass, _TINY), 1.0 / n_y)
    return 0.999 * test + 0.001 / n_y


def duality_gain(tensors: ChannelTensors, qgraph: QGraph, test: np.ndarray, *, tol: float = 1e-10) -> float:
    r"""Return the duality upper bound for the test distribution ``test[q, y]``.

    The bound is the optimal average reward of the finite MDP with state
    ``(s, q)``, action ``x``, reward :math:`D(P_{Y \mid x, s} \| R_{Y \mid q})`
    and transitions ``s' = f(s, x, y)``, ``q' = g(q, y)`` :cite:`Huleihel2025`;
    the span bound ``max(Th - h)`` keeps it a valid upper bound at any accuracy.
    """
    table = qgraph.aligned(tensors.outputs)
    rewards = _divergence_rewards(tensors, test).reshape(-1, tensors.n_inputs)
    result = relative_value_iteration(rewards, _duality_operator(tensors, table), tol=tol)
    return result.gain_upper


def duality_upper_bound(
    channel: Any,
    *,
    qgraphs: Iterable[QGraph] | None = None,
    max_order: int = 2,
    tests: Sequence[np.ndarray] | None = None,
    optimize: bool = True,
    solver: Literal["auto", "cvxpy", "scipy"] = "auto",
) -> CapacityBounds:
    """Upper-bound the feedback capacity by the Q-graph duality bound.

    For each Q-graph the test distribution starts at the output law of the
    Q-graph upper-bound optimizer (or ``tests``, aligned with ``qgraphs``) and,
    with ``optimize=True``, is refined by Nelder--Mead on softmax logits.
    Every test distribution yields a valid bound :cite:`Huleihel2025`.
    """
    tensors = as_tensors(channel)
    candidates = [QGraph.markov(tensors.outputs, k) for k in range(max_order + 1)]
    user = list(qgraphs or [])
    candidates += user
    user_tests = list(tests or [])
    best_value, best_graph, best_test = np.inf, None, None
    for index, qgraph in enumerate(candidates):
        table = qgraph.aligned(tensors.outputs)
        user_index = index - (len(candidates) - len(user))
        if 0 <= user_index < len(user_tests):
            start = np.asarray(user_tests[user_index], dtype=float)
        else:
            _value, joint = qgraph_upper_bound(tensors, qgraph, solver=solver)
            start = _test_from_joint(table, tensors, joint)
        value, test = duality_gain(tensors, qgraph, start), start
        if optimize and tensors.n_outputs > 1:
            value, test = _minimize_test(lambda r, g=qgraph: duality_gain(tensors, g, r, tol=1e-9), start, value)
            value = duality_gain(tensors, qgraph, test)
        if value < best_value:
            best_value, best_graph, best_test = value, qgraph, test
    return CapacityBounds(
        lower=0.0,
        upper=min(float(best_value), tensors.trivial_upper_bound()),
        method="duality",
        feedback=True,
        details={"qgraph": best_graph, "test_distribution": best_test},
    )


def _minimize_test(objective, start: np.ndarray, start_value: float) -> tuple[float, np.ndarray]:
    logits0 = np.log(np.maximum(start, 1e-12))
    shape = start.shape
    result = minimize(
        lambda z: objective(_softmax_rows(z.reshape(shape))),
        logits0.ravel(),
        method="Nelder-Mead",
        options={"xatol": 1e-7, "fatol": 1e-10, "maxiter": 400 * logits0.size},
    )
    test = _softmax_rows(result.x.reshape(shape))
    if result.fun < start_value:
        return float(result.fun), test
    return start_value, start


# -- entry point --------------------------------------------------------------


def feedback_capacity(channel: Any, method: FeedbackMethod = "qgraph", **kwargs: Any) -> CapacityBounds:
    """Bracket the feedback capacity of a finite unifilar channel.

    ``method`` selects :func:`qgraph_bounds`, :func:`value_iteration_estimate`,
    :func:`duality_upper_bound`, or ``"all"``, which intersects the certified
    brackets of the Q-graph and duality bounds and reports the value-iteration
    estimate in ``details``. Keyword arguments go to the selected method(s).
    """
    if method == "qgraph":
        return qgraph_bounds(channel, **kwargs)
    if method == "value_iteration":
        return value_iteration_estimate(channel, **kwargs)
    if method == "duality":
        return duality_upper_bound(channel, **kwargs)
    if method == "all":
        tensors = as_tensors(channel)
        qgraph_keys = ("qgraphs", "max_order", "solver", "restarts", "seed", "block", "final_block")
        qgraph_kwargs = {k: kwargs[k] for k in qgraph_keys if k in kwargs}
        duality_kwargs = {k: kwargs[k] for k in ("qgraphs", "max_order", "solver", "tests", "optimize") if k in kwargs}
        vi_kwargs = {k: kwargs[k] for k in ("belief_resolution", "action_resolution", "max_grid_points") if k in kwargs}
        q = qgraph_bounds(tensors, **qgraph_kwargs)
        dual = duality_upper_bound(tensors, **duality_kwargs)
        vi = value_iteration_estimate(tensors, **vi_kwargs)
        return CapacityBounds(
            lower=q.lower,
            upper=min(q.upper, dual.upper),
            method="all",
            feedback=True,
            details={"qgraph": q, "duality": dual, "value_iteration": vi},
        )
    raise ValueError(f"unknown feedback-capacity method {method!r}")
