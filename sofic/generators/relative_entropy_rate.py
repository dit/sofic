"""Relative entropy (Kullback-Leibler divergence) rate between stationary processes.

For stationary processes ``P`` and ``Q`` over a finite alphabet the relative
entropy rate, in bits per symbol, is

.. math::

   D(P \\| Q) = \\lim_{n \\to \\infty} \\tfrac{1}{n} D(P_{0:n} \\| Q_{0:n}),

where ``P_{0:n}`` is the law of the length-``n`` block :cite:`Gray1990,Cover2006`.
Both processes are taken in their stationary laws (each generator's
:meth:`~sofic.generators.base.StochasticModel.stationary_distribution`).
"""

from __future__ import annotations

import math
from collections.abc import Hashable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import networkx as nx
import numpy as np

from sofic.generators.support import support_includes, support_nfa

if TYPE_CHECKING:
    from sofic.generators.base import HiddenMarkovModel

_TOL = 1e-12


@dataclass(frozen=True)
class RelativeEntropyRateBounds:
    """Bounds on a relative entropy rate in bits per symbol.

    ``lower <= D(P || Q) <= upper``. ``block_estimate`` is
    ``D(P_{0:n} || Q_{0:n}) / n`` and ``conditional_estimate`` is
    ``D(P_{0:n} || Q_{0:n}) - D(P_{0:n-1} || Q_{0:n-1})``, the expected
    divergence of the next-symbol predictions given the length ``n - 1`` past.
    """

    lower: float
    upper: float
    block_length: int
    block_estimate: float
    conditional_estimate: float


@dataclass(frozen=True)
class _Presentation:
    states: tuple[Hashable, ...]
    pi: np.ndarray
    matrices: dict[Any, np.ndarray]


def _presentation(model: HiddenMarkovModel) -> _Presentation:
    from sofic.generators.matrices import symbol_matrices

    mealy = model.to_mealy()
    states = tuple(mealy.reindex().states)
    pi = np.asarray(mealy.stationary_distribution(), dtype=float)
    matrices = {symbol: np.asarray(matrix, dtype=float) for symbol, matrix in symbol_matrices(mealy).items()}
    return _Presentation(states, pi, matrices)


def _aligned(p: _Presentation, q: _Presentation) -> tuple[list[Any], dict[Any, np.ndarray]]:
    symbols = sorted(set(p.matrices) | set(q.matrices), key=repr)
    zeros = np.zeros((len(q.states), len(q.states)))
    return symbols, {symbol: q.matrices.get(symbol, zeros) for symbol in symbols}


_support_nfa = support_nfa
_support_includes = support_includes


def _unifilar_successors(matrices: dict[Any, np.ndarray]) -> dict[Any, dict[int, int]] | None:
    successors: dict[Any, dict[int, int]] = {}
    for symbol, matrix in matrices.items():
        row_targets: dict[int, int] = {}
        for i, row in enumerate(matrix):
            targets = np.flatnonzero(row > _TOL)
            if len(targets) > 1:
                return None
            if len(targets) == 1:
                row_targets[i] = int(targets[0])
        successors[symbol] = row_targets
    return successors


def relative_entropy_rate(p: HiddenMarkovModel, q: HiddenMarkovModel) -> float:
    """Return the relative entropy rate ``D(P || Q)`` in bits per symbol, exactly.

    The result is :data:`math.inf` when some word of positive ``P`` probability
    has ``Q`` probability zero, decided by
    :func:`~sofic.generators.support.support_includes`.

    Otherwise ``q`` must be unifilar. Writing ``h_μ(P)`` for the entropy rate of
    ``p``,

    .. math::

       D(P \\| Q) = -h_\\mu(P) - \\lim_{n \\to \\infty}
       \\mathbb{E}_P\\left[\\log_2 Q(X_n \\mid X_{0:n})\\right],

    which for a ``k``-step Markov ``Q`` is Gray's formula
    :math:`-h_\\mu(P) - \\mathbb{E}_P[\\log_2 Q(X_k \\mid X_{0:k})]`
    (:cite:`Gray1990`, Lemma 3.10, stated there in nats). For a unifilar ``q``
    the past determines the set of ``q`` states consistent with it. The
    expectation is computed on the finite Markov chain of pairs
    ``(p state, set of q states)`` started from ``p``'s stationary law and the
    stationary support of ``q``: its Cesàro-limit law is built from the
    absorption probabilities into, and stationary laws of, its closed classes.
    In each closed class the ``q`` set must be a single state (``q`` is exactly
    synchronized by ``P``-typical pasts), and ``Q(x | past)`` is then that
    state's emission probability.

    ``h_μ(P)`` comes from :meth:`~sofic.generators.base.HiddenMarkovModel.entropy_rate`
    (exact for unifilar ``p``). ``p`` should have a unique stationary law
    (ergodic).

    Raises
    ------
    NotImplementedError
        If ``q`` is not unifilar, or is not exactly synchronized by ``P``-typical
        pasts. Use :func:`relative_entropy_rate_bounds` instead.
    """
    if not support_includes(p, q):
        return math.inf
    pp, qp = _presentation(p), _presentation(q)
    symbols, q_matrices = _aligned(pp, qp)
    successors = _unifilar_successors(q_matrices)
    if successors is None:
        raise NotImplementedError("exact relative entropy rate needs a unifilar q; use relative_entropy_rate_bounds")

    start_set = frozenset(int(i) for i in np.flatnonzero(qp.pi > _TOL))
    nodes: dict[tuple[int, frozenset[int]], int] = {}
    edges: list[tuple[int, int, float]] = []
    frontier = [(int(i), start_set) for i in np.flatnonzero(pp.pi > _TOL)]
    for node in frontier:
        nodes[node] = len(nodes)
    while frontier:
        state, subset = frontier.pop()
        for symbol in symbols:
            matrix = pp.matrices.get(symbol)
            if matrix is None:
                continue
            following = frozenset(successors[symbol][r] for r in subset if r in successors[symbol])
            for target in np.flatnonzero(matrix[state] > _TOL):
                if not following:
                    return math.inf
                node = (int(target), following)
                if node not in nodes:
                    nodes[node] = len(nodes)
                    frontier.append(node)
                edges.append((nodes[(state, subset)], nodes[node], float(matrix[state, target])))

    size = len(nodes)
    chain = np.zeros((size, size))
    for source, target, prob in edges:
        chain[source, target] += prob
    initial = np.zeros(size)
    for i in np.flatnonzero(pp.pi > _TOL):
        initial[nodes[(int(i), start_set)]] = pp.pi[i]
    limit = _cesaro_limit(chain, initial / initial.sum())

    labels = {index: node for node, index in nodes.items()}
    emit_p = {symbol: pp.matrices[symbol].sum(axis=1) for symbol in pp.matrices}
    emit_q = {symbol: q_matrices[symbol].sum(axis=1) for symbol in symbols}
    cross = 0.0
    for index in np.flatnonzero(limit > _TOL):
        state, subset = labels[int(index)]
        if len(subset) != 1:
            raise NotImplementedError(
                "q is not exactly synchronized by typical pasts of p; use relative_entropy_rate_bounds"
            )
        (r,) = subset
        for symbol, probs in emit_p.items():
            if probs[state] <= _TOL:
                continue
            if emit_q[symbol][r] <= _TOL:
                return math.inf
            cross -= limit[index] * probs[state] * math.log2(emit_q[symbol][r])
    return max(cross - float(p.entropy_rate()), 0.0)


def _cesaro_limit(chain: np.ndarray, initial: np.ndarray) -> np.ndarray:
    graph = nx.DiGraph()
    graph.add_nodes_from(range(len(chain)))
    graph.add_edges_from(zip(*np.nonzero(chain), strict=True))
    condensation = nx.condensation(graph)
    closed = [sorted(condensation.nodes[c]["members"]) for c in condensation.nodes if condensation.out_degree(c) == 0]
    recurrent = {state for members in closed for state in members}
    transient = [state for state in range(len(chain)) if state not in recurrent]

    if transient:
        sub = chain[np.ix_(transient, transient)]
        visits = np.linalg.solve((np.eye(len(transient)) - sub).T, initial[transient])
    limit = np.zeros(len(chain))
    for members in closed:
        mass = float(initial[members].sum())
        if transient:
            mass += float(visits @ chain[np.ix_(transient, members)].sum(axis=1))
        block = chain[np.ix_(members, members)]
        system = np.vstack([block.T - np.eye(len(members)), np.ones(len(members))])
        rhs = np.zeros(len(members) + 1)
        rhs[-1] = 1.0
        stationary = np.linalg.lstsq(system, rhs, rcond=None)[0]
        limit[members] = mass * stationary
    return limit


def relative_entropy_rate_bounds(
    p: HiddenMarkovModel, q: HiddenMarkovModel, block_length: int
) -> RelativeEntropyRateBounds:
    """Return bounds on ``D(P || Q)`` from exact enumeration of length-``n`` words.

    Valid for any finite HMMs ``p`` and ``q`` (``q`` need not be unifilar). With
    ``Q_s`` the law of ``q`` started in state ``s`` and ``s`` ranging over the
    stationary support of ``q``, let

    .. math::

       L_n = \\mathbb{E}_P[-\\log_2 \\max_s Q_s(X_{0:n})], \\qquad
       U_n = \\mathbb{E}_P[-\\log_2 \\min_s Q_s(X_{0:n})].

    Because ``Q(uv)`` lies between ``Q(u) \\min_s Q_s(v)`` and
    ``Q(u) \\max_s Q_s(v)``, ``L_n`` is superadditive and ``U_n`` subadditive,
    so the cross-entropy rate ``c = \\lim -\\tfrac{1}{n}\\mathbb{E}_P \\log_2 Q(X_{0:n})``
    satisfies ``L_n / n \\le c \\le U_n / n``, and ``L_n / n`` increases to
    ``c`` when ``q``'s stationary law has full support on its recurrent states.
    ``U_n`` is infinite whenever some recurrent ``q`` state cannot emit a
    ``P``-typical word; it is finite, and ``U_n / n`` decreases to ``c``, when
    every ``q`` state can emit every word.

    The entropy rate is bracketed by the hidden-Markov bounds
    ``H[X_{n-1} | X_{0:n-1}, S_0] \\le h_μ(P) \\le H[X_{n-1} | X_{0:n-1}]``
    (:cite:`Cover2006`, Theorem 4.5.1), which converge from both sides. So
    ``lower = max(0, L_n/n - H[X_{n-1} | X_{0:n-1}])`` and
    ``upper = U_n/n - H[X_{n-1} | X_{0:n-1}, S_0]``.

    Both bounds are :data:`math.inf` when ``P`` is not absolutely continuous
    with respect to ``Q`` on finite words, in which case the rate is infinite.
    Enumeration costs ``O(|A|^n)`` words.
    """
    if block_length < 1:
        raise ValueError("block_length must be positive")
    n = block_length
    if not support_includes(p, q):
        return RelativeEntropyRateBounds(math.inf, math.inf, n, math.inf, math.inf)
    pp, qp = _presentation(p), _presentation(q)
    symbols, q_matrices = _aligned(pp, qp)
    q_live = np.flatnonzero(qp.pi > _TOL)
    q_pi = qp.pi[q_live]
    q_sub = {symbol: matrix[np.ix_(q_live, q_live)] for symbol, matrix in q_matrices.items()}

    block = np.zeros(n + 1)
    cross = np.zeros(n + 1)
    low = np.zeros(n + 1)
    high = np.zeros(n + 1)

    def walk(alpha: np.ndarray, product: np.ndarray, length: int) -> None:
        prob = float(alpha.sum())
        if length:
            from_states = product.sum(axis=1)
            block[length] -= prob * math.log2(prob)
            cross[length] -= prob * math.log2(float(q_pi @ from_states))
            low[length] -= prob * math.log2(float(from_states.max()))
            smallest = float(from_states.min())
            high[length] = math.inf if smallest <= 0.0 else high[length] - prob * math.log2(smallest)
        if length == n:
            return
        for symbol in symbols:
            matrix = pp.matrices.get(symbol)
            if matrix is None:
                continue
            following = alpha @ matrix
            if following.sum() > _TOL:
                walk(following, product @ q_sub[symbol], length + 1)

    walk(pp.pi, np.eye(len(q_live)), 0)

    conditioned = np.zeros(n + 1)
    for i in np.flatnonzero(pp.pi > _TOL):
        start = np.zeros(len(pp.pi))
        start[i] = 1.0
        conditioned += pp.pi[i] * _block_entropies(start, pp.matrices, n)

    divergence = cross - block
    h_upper = block[n] - block[n - 1]
    h_lower = conditioned[n] - conditioned[n - 1]
    return RelativeEntropyRateBounds(
        lower=max(0.0, float(low[n] / n - h_upper)),
        upper=float(high[n] / n - h_lower),
        block_length=n,
        block_estimate=float(divergence[n] / n),
        conditional_estimate=float(divergence[n] - divergence[n - 1]),
    )


def _block_entropies(start: np.ndarray, matrices: dict[Any, np.ndarray], n: int) -> np.ndarray:
    entropies = np.zeros(n + 1)

    def walk(alpha: np.ndarray, length: int) -> None:
        if length:
            prob = float(alpha.sum())
            entropies[length] -= prob * math.log2(prob)
        if length == n:
            return
        for matrix in matrices.values():
            following = alpha @ matrix
            if following.sum() > _TOL:
                walk(following, length + 1)

    walk(start, 0)
    return entropies
