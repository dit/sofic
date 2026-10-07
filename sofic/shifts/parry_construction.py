"""Parry measure construction for topological Markov chains."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Hashable
from typing import Any

import numpy as np

from sofic.generators.mealy import MealyHMM
from sofic.graph import ATTR_EMISSION, ATTR_MULTIPLICITY, ATTR_PROB, ATTR_SYMBOL
from sofic.shifts.algorithms import adjacency_matrix
from sofic.shifts.tmc import TopologicalMarkovChain


def _perron_pair(matrix: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """Return (λ, left, right) Perron eigenvectors with left^T right = 1."""
    eigenvalues, right_vecs = np.linalg.eig(matrix)
    index = int(np.argmax(np.real(eigenvalues)))
    lam = float(np.real(eigenvalues[index]))
    right = np.real(right_vecs[:, index])
    if right[np.argmax(np.abs(right))] < 0.0:
        right = -right
    right = np.maximum(right, 1e-300)

    left_eigenvalues, left_vecs = np.linalg.eig(matrix.T)
    left_index = int(np.argmax(np.real(left_eigenvalues)))
    left = np.real(left_vecs[:, left_index])
    if left[np.argmax(np.abs(left))] < 0.0:
        left = -left
    left = np.maximum(left, 1e-300)

    scale = float(left @ right)
    if scale < 0.0:
        left = -left
        scale = -scale
    if scale <= 0.0:
        raise ValueError("failed to normalize Perron eigenvectors")
    left /= scale
    return lam, left, right


def parry_measure(tmc: TopologicalMarkovChain) -> MealyHMM:
    """Parry measure (measure of maximal entropy) of an edge shift :cite:`Parry1964`.

    Every edge ``i -> j`` -- each of the ``multiplicity`` parallel copies
    counting separately -- gets probability ``v_j / (lambda v_i)`` from the
    right Perron vector ``v``, with stationary distribution ``pi ~ u * v``.

    A TMC is an edge shift, so parallel edges are distinct points. When every
    edge's symbol is unique among the edges leaving its source, edges emit
    their symbols. Otherwise every edge emits ``(symbol, k)``, where ``k`` counts
    the edges leaving the same source with the same symbol (``k = 0, 1, ...``,
    ordered by target and then copy), so a ``2`` in the adjacency matrix yields
    ``(symbol, 0)`` and ``(symbol, 1)``. Either way the HMM is unifilar and its
    entropy rate equals
    :meth:`~sofic.shifts.tmc.TopologicalMarkovChain.topological_entropy`.
    """
    matrix, states = adjacency_matrix(tmc)
    if matrix.size == 0:
        return MealyHMM(initial_distribution={}, observation_alphabet=frozenset())

    lam, left, right = _perron_pair(matrix)
    n = len(states)
    index = {state: i for i, state in enumerate(states)}

    stationary = left * right
    stationary /= stationary.sum()

    copies: dict[tuple[Hashable, Any], list[tuple[Hashable, Hashable]]] = defaultdict(list)
    for transition_edge in sorted(tmc.transitions(), key=lambda t: (index[t.source], index[t.target], t.key)):
        symbol = transition_edge.data.get(ATTR_EMISSION)
        if symbol is None:
            symbol = transition_edge.data.get(ATTR_SYMBOL)
        multiplicity = int(transition_edge.data.get(ATTR_MULTIPLICITY, 1))
        copies[(transition_edge.source, symbol)].extend(
            [(transition_edge.source, transition_edge.target)] * multiplicity
        )

    relabel = any(len(group) > 1 for group in copies.values())
    edges: list[tuple[Hashable, Hashable, Any, float]] = []
    for (_source, symbol), group in copies.items():
        for k, (source, target) in enumerate(group):
            label = (symbol, k) if relabel else symbol
            weight = right[index[target]] / (lam * right[index[source]])
            edges.append((source, target, label, float(weight)))

    labels = frozenset(label for _, _, label, _ in edges)
    hmm = MealyHMM(
        initial_distribution={states[i]: float(stationary[i]) for i in range(n)},
        observation_alphabet=labels if relabel else tmc.symbol_alphabet | labels,
    )
    for state in states:
        hmm.graph.add_state(state)
    for source, target, label, weight in edges:
        hmm.graph.add_transition(source, target, **{ATTR_PROB: weight, ATTR_EMISSION: label})
    hmm.validate()
    return hmm
