r"""Capacity bounds for finite unifilar channels (ε-transducers) without feedback.

The capacity of a finite-state channel without feedback has no known
computable characterization, even for the trapdoor channel. This module
brackets it:

* :func:`markov_input_lower_bound` -- the information rate of an optimized
  order-``k`` Markov input, the quantity maximized by the generalized
  Blahut--Arimoto algorithm :cite:`Vontobel2008`, certified from below by
  block entropy bounds :cite:`Cover2006`.
* :func:`duality_upper_bound_no_feedback` -- the dual capacity bound with a
  Q-graph test distribution, which for unifilar channels is a deterministic
  average-reward dynamic program over :math:`P(S_t, Q_t \mid x^t)`
  :cite:`Huleihel2021`.

Both assume an indecomposable channel (the initial state is forgotten), so
that the capacity does not depend on it. Inputs must be admissible in every
state, since an encoder without feedback cannot see the state.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from itertools import product
from typing import Any, Literal

import numpy as np
from scipy.optimize import minimize

from sofic.exceptions import StochasticValidationError
from sofic.generators.capacity._belief import SimplexGrid, relative_value_iteration, transition_operator
from sofic.generators.capacity._channel import ChannelTensors
from sofic.generators.capacity._rates import default_block, driven_rate_bounds
from sofic.generators.capacity.feedback import (
    _divergence_rewards,
    _minimize_test,
    _softmax_rows,
    as_tensors,
)
from sofic.generators.capacity.qgraph import QGraph, coupled_kernel
from sofic.generators.capacity.results import CapacityBounds

NoFeedbackMethod = Literal["markov_input", "duality", "all"]


def _require_unconstrained(tensors: ChannelTensors) -> None:
    if not tensors.allowed.all():
        raise StochasticValidationError("no-feedback bounds need every input admissible in every state")


def _markov_input_tensors(tensors: ChannelTensors, policy: np.ndarray, order: int) -> tuple[np.ndarray, np.ndarray]:
    """Return the ``(context, state)`` chain's symbol tensors and per-state noise for ``policy[c, x]``."""
    n_s, n_x, n_y = tensors.P.shape
    contexts = list(product(range(n_x), repeat=order))
    index = {c: i for i, c in enumerate(contexts)}
    n_c = len(contexts)
    symbol = np.zeros((n_y, n_c, n_s, n_c, n_s))
    for c, context in enumerate(contexts):
        for x in range(n_x):
            target = index[(*context, x)[1:]] if order else 0
            for s, y in zip(*np.nonzero(tensors.P[:, x, :] > 0), strict=True):
                symbol[y, c, s, target, tensors.F[s, x, y]] += policy[c, x] * tensors.P[s, x, y]
    n_z = n_c * n_s
    noise = (policy @ tensors.conditional_entropy().T).ravel()
    return symbol.reshape(n_y, n_z, n_z), noise


def markov_input_lower_bound(
    channel: Any,
    *,
    order: int = 1,
    restarts: int = 4,
    seed: int | None = 0,
    block: int | None = None,
    final_block: int | None = None,
) -> CapacityBounds:
    r"""Lower-bound the no-feedback capacity with an optimized Markov input.

    An order-``k`` Markov input with transition law :math:`\theta(x \mid c)`
    drives the channel; the pair (input context, channel state) is a finite
    chain, and because ``(s, x, y)`` determines the next state the
    information rate is :math:`\bar h(Y) - \mathbb{E}_\pi H(Y \mid X, S)`.
    The entropy rate is bounded below by :math:`H[Y_n \mid Y_{0:n}, Z_0]`
    :cite:`Cover2006`, so the result is a certified lower bound. ``theta`` is
    optimized with L-BFGS-B on softmax logits at ``block`` and re-evaluated at
    ``final_block`` (defaults: largest ``n`` with :math:`|Y|^n \le 2^8`,
    :math:`2^{16}`). This is the objective of the generalized Blahut--Arimoto
    algorithm :cite:`Vontobel2008,Arnold2006`; the optimizer here is direct.
    """
    tensors = as_tensors(channel)
    _require_unconstrained(tensors)
    block = default_block(tensors.n_outputs, 8) if block is None else block
    final_block = default_block(tensors.n_outputs, 16) if final_block is None else final_block
    n_x = tensors.n_inputs
    shape = (n_x**order, n_x)
    rng = np.random.default_rng(seed)

    def rate(z: np.ndarray, n: int) -> tuple[float, float] | None:
        symbol, noise = _markov_input_tensors(tensors, _softmax_rows(z.reshape(shape)), order)
        return driven_rate_bounds(symbol, noise, n)

    def objective(z: np.ndarray) -> float:
        bounds = rate(z, block)
        return 1.0 if bounds is None else -bounds[0]

    starts = [np.zeros(int(np.prod(shape)))] + [rng.normal(size=int(np.prod(shape))) for _ in range(restarts)]
    best_z, best_value = starts[0], np.inf
    for z0 in starts:
        result = minimize(objective, z0, method="L-BFGS-B", options={"maxiter": 500})
        if result.fun < best_value:
            best_z, best_value = result.x, float(result.fun)
    final = rate(best_z, final_block)
    lower = 0.0 if final is None else max(final[0], 0.0)
    return CapacityBounds(
        lower=lower,
        upper=tensors.trivial_upper_bound(),
        method="markov_input",
        feedback=False,
        details={
            "order": order,
            "input_policy": _softmax_rows(best_z.reshape(shape)),
            "rate_estimate_upper": None if final is None else final[1],
        },
    )


def _no_feedback_dp(tensors: ChannelTensors, table: np.ndarray, resolution: int, max_grid_points: int):
    """Grid and interpolated transition operator for the belief ``z = P(S, Q | x^t)``."""
    n_s, n_x, _n_y = tensors.P.shape
    n_q = table.shape[0]
    n = n_s * n_q
    grid = SimplexGrid.build(n, resolution, max_points=max_grid_points)
    kernel = coupled_kernel(tensors, table).reshape(n, n_x, n)
    rows, indices, weights = [], [], []
    for x in range(n_x):
        queries = grid.points @ kernel[:, x, :]
        idx, w = grid.interpolation(queries)
        rows.append(x * grid.size + np.arange(grid.size))
        indices.append(idx)
        weights.append(w)
    operator = transition_operator(
        n_x * grid.size,
        grid.size,
        np.concatenate(rows),
        np.vstack(indices),
        np.vstack(weights),
        np.ones(n_x * grid.size),
    )
    return grid, operator


def _no_feedback_gain(tensors: ChannelTensors, grid: SimplexGrid, operator, test: np.ndarray, tol: float) -> float:
    rewards = grid.points @ _divergence_rewards(tensors, test).reshape(-1, tensors.n_inputs)
    return relative_value_iteration(rewards, operator, tol=tol).gain_upper


def duality_upper_bound_no_feedback(
    channel: Any,
    *,
    qgraphs: Iterable[QGraph] | None = None,
    max_order: int = 1,
    tests: Sequence[np.ndarray] | None = None,
    belief_resolution: int = 20,
    max_grid_points: int = 100_000,
    optimize: bool = True,
) -> CapacityBounds:
    r"""Upper-bound the no-feedback capacity by the Q-graph duality bound.

    For a test distribution :math:`R(y \mid q)` on a Q-graph,
    :math:`C \le \lim \frac1n \max_{x^n, s_0, q_0} D(P_{Y^n \mid x^n, s_0} \| R_{Y^n})`,
    which for a unifilar channel is the deterministic dynamic program with
    state :math:`z = P(S, Q \mid x^t)`, action ``x``, reward
    :math:`\sum_{s,q} z(s,q) D(P_{Y \mid x,s} \| R_{Y \mid q})` and linear
    dynamics :cite:`Huleihel2021`. Its value function is convex in ``z``, so
    linear interpolation on a simplex grid over-estimates it and the span
    bound ``max(Th - h)`` of the gridded problem is still a valid upper bound
    at any resolution. The test distribution starts uniform (or ``tests``,
    aligned with ``qgraphs``) and is refined by Nelder--Mead when
    ``optimize`` is true.
    """
    tensors = as_tensors(channel)
    _require_unconstrained(tensors)
    candidates = [QGraph.markov(tensors.outputs, k) for k in range(max_order + 1)]
    user = list(qgraphs or [])
    candidates += user
    user_tests = list(tests or [])
    best_value, best_graph, best_test = np.inf, None, None
    for index, qgraph in enumerate(candidates):
        table = qgraph.aligned(tensors.outputs)
        grid, operator = _no_feedback_dp(tensors, table, belief_resolution, max_grid_points)
        user_index = index - (len(candidates) - len(user))
        if 0 <= user_index < len(user_tests):
            start = np.asarray(user_tests[user_index], dtype=float)
        else:
            start = np.full((qgraph.n_nodes, tensors.n_outputs), 1.0 / tensors.n_outputs)
        value, test = _no_feedback_gain(tensors, grid, operator, start, 1e-10), start
        if optimize and tensors.n_outputs > 1:
            value, test = _minimize_test(
                lambda r, g=grid, op=operator: _no_feedback_gain(tensors, g, op, r, 1e-8), start, value
            )
            value = _no_feedback_gain(tensors, grid, operator, test, 1e-10)
        if value < best_value:
            best_value, best_graph, best_test = value, qgraph, test
    return CapacityBounds(
        lower=0.0,
        upper=min(float(best_value), tensors.trivial_upper_bound()),
        method="duality",
        feedback=False,
        details={"qgraph": best_graph, "test_distribution": best_test, "belief_resolution": belief_resolution},
    )


def capacity(channel: Any, method: NoFeedbackMethod = "all", **kwargs: Any) -> CapacityBounds:
    """Bracket the capacity without feedback of a finite unifilar channel.

    ``method`` selects :func:`markov_input_lower_bound`,
    :func:`duality_upper_bound_no_feedback`, or ``"all"`` for both. Keyword
    arguments go to the selected method(s).
    """
    if method == "markov_input":
        return markov_input_lower_bound(channel, **kwargs)
    if method == "duality":
        return duality_upper_bound_no_feedback(channel, **kwargs)
    if method == "all":
        tensors = as_tensors(channel)
        lower_keys = ("order", "restarts", "seed", "block", "final_block")
        upper_keys = ("qgraphs", "max_order", "tests", "belief_resolution", "max_grid_points", "optimize")
        low = markov_input_lower_bound(tensors, **{k: kwargs[k] for k in lower_keys if k in kwargs})
        high = duality_upper_bound_no_feedback(tensors, **{k: kwargs[k] for k in upper_keys if k in kwargs})
        return CapacityBounds(
            lower=min(low.lower, high.upper),
            upper=high.upper,
            method="all",
            feedback=False,
            details={"markov_input": low, "duality": high},
        )
    raise ValueError(f"unknown capacity method {method!r}")
