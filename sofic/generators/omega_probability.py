"""Probabilities of regular and ω-regular properties of HMM output.

Both quantities are computed on the product of the HMM with a deterministic
automaton: the automaton state after reading ``X_{0:t}`` is a function of the
output, so the pair (HMM state, automaton state) is again a finite Markov chain
:cite:`BaierKatoen2008`.
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
    from sofic.automata.base import LabeledAutomaton
    from sofic.automata.buchi import BuchiAutomaton
    from sofic.automata.dfa import DFA
    from sofic.generators.base import HiddenMarkovModel
    from sofic.generators.matrices import StartSpec

_TOL = 1e-12
_DEAD = object()


def _tensors(hmm: HiddenMarkovModel, start: StartSpec) -> tuple[np.ndarray, dict[Any, np.ndarray]]:
    from sofic.generators.matrices import start_vector, symbol_matrices

    mealy = hmm.to_mealy()
    pi = np.asarray(start_vector(mealy, start, policy="stationary"), dtype=float)
    matrices = {symbol: np.asarray(matrix, dtype=float) for symbol, matrix in symbol_matrices(mealy).items()}
    return pi, matrices


def _transition_table(aut: LabeledAutomaton) -> tuple[Hashable, dict[tuple[Hashable, Any], Hashable]]:
    if not aut.is_deterministic():
        raise NonDeterministicError(f"{type(aut).__name__} must be deterministic")
    (initial,) = aut.initial_states
    return initial, {(t.source, t.data[ATTR_SYMBOL]): t.target for t in aut.transitions()}


def omega_probability(hmm: HiddenMarkovModel, buchi: BuchiAutomaton, *, start: StartSpec = None) -> float:
    """Return the probability that the output ``X_0 X_1 ...`` of ``hmm`` is accepted by ``buchi``.

    ``buchi`` must be a deterministic Büchi automaton (one initial state, no
    ε-moves, at most one successor per state and symbol); a missing transition
    rejects. The HMM starts in its stationary law, or in ``start`` (a state, a
    state-to-mass mapping, or a vector in ``hmm.to_mealy().reindex()`` order).

    The product chain on (HMM state, automaton state), with an absorbing
    rejecting state for missing transitions, almost surely enters a bottom
    strongly connected component and then visits each of its states infinitely
    often. The run is therefore accepted iff that component contains an
    accepting automaton state, and the probability is the mass absorbed into
    such components :cite:`BaierKatoen2008`.

    Deterministic Büchi automata cannot express every ω-regular property
    (``finitely many 1s`` is the standard counterexample); its probability is
    one minus that of the DBA ``infinitely many 1s``.

    Raises
    ------
    NonDeterministicError
        If ``buchi`` is not deterministic.
    """
    pi, matrices = _tensors(hmm, start)
    initial, delta = _transition_table(buchi)

    nodes: dict[Any, int] = {_DEAD: 0}
    frontier: deque[tuple[int, Hashable]] = deque()
    for i in np.flatnonzero(pi > _TOL):
        nodes[(int(i), initial)] = len(nodes)
        frontier.append((int(i), initial))
    edges: list[tuple[int, int, float]] = [(0, 0, 1.0)]
    while frontier:
        i, q = frontier.popleft()
        for symbol, matrix in matrices.items():
            target = delta.get((q, symbol))
            for j in np.flatnonzero(matrix[i] > _TOL):
                node = _DEAD if target is None else (int(j), target)
                if node not in nodes:
                    nodes[node] = len(nodes)
                    frontier.append(node)
                edges.append((nodes[(i, q)], nodes[node], float(matrix[i, j])))

    size = len(nodes)
    chain = np.zeros((size, size))
    for source, target, prob in edges:
        chain[source, target] += prob
    graph = nx.DiGraph()
    graph.add_nodes_from(range(size))
    graph.add_edges_from((source, target) for source, target, _ in edges)
    condensation = nx.condensation(graph)
    accepting = {index for node, index in nodes.items() if node is not _DEAD and node[1] in buchi.accepting_states}
    good: set[int] = set()
    for component in condensation.nodes:
        members = condensation.nodes[component]["members"]
        if condensation.out_degree(component) == 0 and members & accepting:
            good |= members
    if not good:
        return 0.0

    reaching = set(good)
    for node in good:
        reaching |= nx.ancestors(graph, node)
    transient = sorted(reaching - good)
    absorbed = np.zeros(size)
    absorbed[sorted(good)] = 1.0
    if transient:
        sub = chain[np.ix_(transient, transient)]
        into = chain[np.ix_(transient, sorted(good))].sum(axis=1)
        absorbed[transient] = np.linalg.solve(np.eye(len(transient)) - sub, into)

    total = sum(pi[i] * absorbed[nodes[(int(i), initial)]] for i in np.flatnonzero(pi > _TOL))
    return float(min(max(total, 0.0), 1.0))


def regular_language_probability(hmm: HiddenMarkovModel, dfa: DFA, n: int, *, start: StartSpec = None) -> float:
    """Return ``P(X_{0:n} ∈ L(dfa))`` exactly.

    Dynamic programming over the product of HMM states and DFA states: the
    forward vector of each DFA state is pushed through ``T^(x)`` along
    ``δ(q, x)`` for ``n`` steps, and the mass in accepting DFA states is summed
    :cite:`HopcroftUllman1979,Rabiner1989`. Cost is ``O(n |Q| |A| |S|^2)``.
    ``start`` is as in :func:`omega_probability`, defaulting to the stationary
    law. A missing DFA transition rejects.

    Raises
    ------
    NonDeterministicError
        If ``dfa`` is not deterministic.
    ValueError
        If ``n`` is negative.
    """
    if n < 0:
        raise ValueError("n must be nonnegative")
    pi, matrices = _tensors(hmm, start)
    initial, delta = _transition_table(dfa)
    alpha: dict[Hashable, np.ndarray] = {initial: pi}
    for _ in range(n):
        following: dict[Hashable, np.ndarray] = {}
        for q, vector in alpha.items():
            for symbol, matrix in matrices.items():
                target = delta.get((q, symbol))
                if target is None:
                    continue
                pushed = vector @ matrix
                following[target] = following[target] + pushed if target in following else pushed
        alpha = following
    return float(sum(vector.sum() for q, vector in alpha.items() if q in dfa.accepting_states))
