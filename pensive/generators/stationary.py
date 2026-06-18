"""Stationary distribution for hidden Markov models."""

from __future__ import annotations

import numpy as np

from pensive.generators.base import HiddenMarkovModel
from pensive.graph import ATTR_PROB


def stationary_distribution_hmm(hmm: HiddenMarkovModel) -> np.ndarray:
    idx = hmm.reindex()
    n = len(idx)
    if n == 0:
        return np.array([], dtype=float)

    transition = np.zeros((n, n), dtype=float)
    for state in idx.states:
        i = idx.index(state)
        for edge in hmm.graph.out_transitions(state):
            j = idx.index(edge.target)
            transition[i, j] += float(edge.data.get(ATTR_PROB, 0.0))

    distribution = np.zeros(n, dtype=float)
    for state, mass in hmm.initial_distribution.items():
        distribution[idx.index(state)] = float(mass)
    if distribution.sum() <= 0.0:
        distribution = np.full(n, 1.0 / n, dtype=float)

    for _ in range(10_000):
        updated = distribution @ transition
        if np.allclose(updated, distribution, rtol=1e-10, atol=1e-12):
            distribution = updated
            break
        distribution = updated

    total = distribution.sum()
    if total <= 0.0:
        return np.full(n, 1.0 / n, dtype=float)
    return distribution / total
