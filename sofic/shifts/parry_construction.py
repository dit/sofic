"""Parry measure construction for topological Markov chains."""

from __future__ import annotations

import numpy as np

from sofic.generators.mealy import MealyHMM
from sofic.graph import ATTR_EMISSION, ATTR_MULTIPLICITY, ATTR_PROB
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
    matrix, states = adjacency_matrix(tmc)
    if matrix.size == 0:
        return MealyHMM(initial_distribution={}, observation_alphabet=frozenset())

    lam, left, right = _perron_pair(matrix)
    n = len(states)
    transition = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(n):
            if matrix[i, j] > 0.0:
                transition[i, j] = matrix[i, j] * right[j] / (lam * right[i])

    stationary = left * right
    stationary /= stationary.sum()

    hmm = MealyHMM(
        initial_distribution={states[i]: float(stationary[i]) for i in range(n)},
        observation_alphabet=tmc.symbol_alphabet,
    )
    for state in states:
        hmm.graph.add_state(state)

    for transition_edge in tmc.transitions():
        i = states.index(transition_edge.source)
        j = states.index(transition_edge.target)
        multiplicity = float(transition_edge.data.get(ATTR_MULTIPLICITY, 1))
        total = matrix[i, j]
        if total <= 0.0:
            continue
        symbol = transition_edge.data.get(ATTR_EMISSION)
        if symbol is None:
            from sofic.graph import ATTR_SYMBOL

            symbol = transition_edge.data.get(ATTR_SYMBOL)
        weight = transition[i, j] * (multiplicity / total)
        hmm.graph.add_transition(
            transition_edge.source,
            transition_edge.target,
            **{ATTR_PROB: float(weight), ATTR_EMISSION: symbol},
        )
    hmm.validate()
    return hmm
