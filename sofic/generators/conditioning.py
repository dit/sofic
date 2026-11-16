"""Conditioning a stationary process on a regular constraint.

Given an HMM with symbol matrices ``T^(x)`` and a DFA, the words whose every
prefix the DFA accepts are those whose run stays in accepting DFA states. On
the product of HMM states and accepting DFA states this is the substochastic
matrix

.. math::

   M^{(x)}_{(i, q), (j, \\delta(q, x))} = T^{(x)}_{ij}
   \\quad \\text{whenever } \\delta(q, x) \\text{ is accepting.}

Conditioning on surviving ``n`` steps and letting ``n -> infinity`` gives
the Doob ``h``-transform of ``M`` by its right Perron eigenvector ``h``,

.. math::

   P'(x, j \\mid i) = \\frac{M^{(x)}_{ij} h_j}{\\lambda h_i},

the construction of Parry's measure of maximal entropy :cite:`Parry1964`
(:cite:`LindMarcus1995`, §13.3) applied to a weighted rather than a 0-1 matrix.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Hashable
from typing import TYPE_CHECKING, Any

import networkx as nx
import numpy as np

from sofic.exceptions import NonDeterministicError
from sofic.graph import ATTR_SYMBOL

if TYPE_CHECKING:
    from sofic.automata.dfa import DFA
    from sofic.generators.base import HiddenMarkovModel
    from sofic.generators.mealy import MealyHMM

_TOL = 1e-12
_RTOL = 1e-9


def condition_on_language(hmm: HiddenMarkovModel, dfa: DFA) -> MealyHMM:
    """Return the stationary process ``hmm`` conditioned to stay in ``dfa``'s language forever.

    The conditioning event for length ``n`` is that every prefix of
    ``X_{0:n}`` is accepted by ``dfa``, i.e. its run never leaves the accepting
    states. For a prefix-closed language this is just ``X_{0:n} ∈ L``, and for a
    factorial one (the language of a shift, such as an SFT) it is moreover
    shift invariant, so the result is the ``n -> infinity`` limit of the law of
    windows ``X_{k:k+m}`` given ``X_{0:n} ∈ L`` with ``k`` and ``n - k`` both
    large. A non-prefix-closed ``dfa`` is effectively replaced by the largest
    prefix-closed sublanguage of its language.

    ``dfa`` is completed over the union of both alphabets (the trap state is
    rejecting), and its product with the stationary-positive states of ``hmm``
    is restricted to accepting DFA states. Surviving paths of length ``n``
    concentrate on the strongly connected component of the restricted matrix
    with the largest Perron root ``λ``; this component is required to be
    unique. On it the result is the Doob ``h``-transform

    .. math::

       P'(x, (j, q') \\mid (i, q)) = \\frac{T^{(x)}_{ij} h_{(j, q')}}{\\lambda h_{(i, q)}},

    with ``h`` the right Perron eigenvector, started in its stationary law
    ``u_s h_s`` (``u`` the left Perron eigenvector). This is the Parry
    construction :cite:`Parry1964` (:cite:`LindMarcus1995`, §13.3): conditioning
    the uniform i.i.d. process on an irreducible SFT returns its measure of
    maximal entropy. When the component is periodic the window limit holds in
    the Cesàro sense.

    States of the returned HMM are pairs ``(hmm state, dfa state)``; its
    observation alphabet is that of ``hmm``.

    Raises
    ------
    NonDeterministicError
        If ``dfa`` is not deterministic.
    ValueError
        If no infinite word survives the constraint, or several components
        share the maximal Perron root, so the limit depends on how the
        constraint is entered.
    """
    from sofic.automata.algorithms import complete
    from sofic.generators.matrices import emission_tensors
    from sofic.generators.mealy import MealyHMM
    from sofic.shifts.parry_construction import _perron_pair

    if not dfa.is_deterministic():
        raise NonDeterministicError("condition_on_language needs a deterministic automaton")
    mealy = hmm.to_mealy()
    states = tuple(mealy.reindex().states)
    pi, raw = emission_tensors(mealy, policy="stationary")
    matrices = {symbol: np.asarray(matrix, dtype=float) for symbol, matrix in raw.items()}
    pi = np.asarray(pi, dtype=float)
    dfa = complete(dfa, frozenset(matrices) | dfa.input_alphabet)
    delta = {(t.source, t.data[ATTR_SYMBOL]): t.target for t in dfa.transitions()}
    accepting = dfa.accepting_states
    (start,) = dfa.initial_states
    if start not in accepting:
        raise ValueError("the empty word is rejected, so no word survives the constraint")

    nodes: dict[tuple[int, Hashable], int] = {}
    frontier: deque[tuple[int, Hashable]] = deque()
    for i in np.flatnonzero(pi > _TOL):
        nodes[(int(i), start)] = len(nodes)
        frontier.append((int(i), start))
    edges: list[tuple[int, int, Any, float]] = []
    while frontier:
        i, q = frontier.popleft()
        for symbol, matrix in matrices.items():
            target = delta[(q, symbol)]
            if target not in accepting:
                continue
            for j in np.flatnonzero(matrix[i] > _TOL):
                node = (int(j), target)
                if node not in nodes:
                    nodes[node] = len(nodes)
                    frontier.append(node)
                edges.append((nodes[(i, q)], nodes[node], symbol, float(matrix[i, j])))

    chain = np.zeros((len(nodes), len(nodes)))
    for source, target, _symbol, prob in edges:
        chain[source, target] += prob
    graph = nx.DiGraph()
    graph.add_nodes_from(range(len(nodes)))
    graph.add_edges_from((source, target) for source, target, _symbol, _prob in edges)
    radii = []
    for members in nx.strongly_connected_components(graph):
        component = sorted(members)
        block = chain[np.ix_(component, component)]
        radii.append((float(np.max(np.abs(np.linalg.eigvals(block)))), component))
    lam = max((radius for radius, _ in radii), default=0.0)
    if lam <= _TOL:
        raise ValueError("no infinite word survives the constraint")
    maximal = [component for radius, component in radii if radius >= lam * (1.0 - _RTOL)]
    if len(maximal) > 1:
        raise ValueError(
            f"{len(maximal)} components share the maximal growth rate; the conditioned limit is not unique"
        )
    (component,) = maximal
    position = {node: k for k, node in enumerate(component)}
    lam, left, right = _perron_pair(chain[np.ix_(component, component)])
    stationary = left * right
    stationary /= stationary.sum()

    labels = {index: (states[i], q) for (i, q), index in nodes.items()}
    result = MealyHMM(
        initial_distribution={labels[node]: float(stationary[position[node]]) for node in component},
        observation_alphabet=frozenset(mealy.observation_alphabet),
    )
    for node in component:
        result.graph.add_state(labels[node])
    for source, target, symbol, prob in edges:
        if source in position and target in position:
            weight = prob * right[position[target]] / (lam * right[position[source]])
            result.add_transition(labels[source], labels[target], symbol, weight)
    return result
