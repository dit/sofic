"""Matrix operations for quasi-stochastic generators."""

from __future__ import annotations

from collections.abc import Hashable, Sequence
from typing import Any

import numpy as np

from pensive.exceptions import QuasiStochasticValidationError
from pensive.generators.base import QuasiStochasticModel
from pensive.graph import ATTR_EMISSION, ATTR_QUASIPROB


def transition_matrices(model: QuasiStochasticModel) -> dict[Any, np.ndarray]:
    idx = model.reindex()
    n = len(idx)
    matrices: dict[Any, np.ndarray] = {}
    for transition in model.transitions():
        emission = transition.data.get(ATTR_EMISSION)
        if emission is None:
            continue
        matrix = matrices.setdefault(emission, np.zeros((n, n), dtype=float))
        i = idx.index(transition.source)
        j = idx.index(transition.target)
        matrix[i, j] += float(transition.data.get(ATTR_QUASIPROB, 0.0))
    return matrices


def stationary_quasidistribution(model: QuasiStochasticModel) -> np.ndarray:
    idx = model.reindex()
    n = len(idx)
    if n == 0:
        return np.array([], dtype=float)

    combined = np.zeros((n, n), dtype=float)
    for matrix in transition_matrices(model).values():
        combined += matrix

    distribution = np.zeros(n, dtype=float)
    for state, mass in model.initial_quasidistribution.items():
        distribution[idx.index(state)] = float(mass)

    if distribution.sum() <= 0.0:
        distribution = np.full(n, 1.0 / n, dtype=float)

    for _ in range(10_000):
        updated = distribution @ combined
        if np.allclose(updated, distribution, rtol=1e-10, atol=1e-12):
            distribution = updated
            break
        distribution = updated

    total = distribution.sum()
    if abs(total) <= 0.0:
        raise QuasiStochasticValidationError("failed to compute stationary quasidistribution")
    return distribution / total


def word_probability(model: QuasiStochasticModel, word: Sequence[Any]) -> float:
    idx = model.reindex()
    n = len(idx)
    pi = np.zeros(n, dtype=float)
    for state, mass in model.initial_quasidistribution.items():
        pi[idx.index(state)] = float(mass)
    ones = np.ones(n, dtype=float)
    matrices = transition_matrices(model)
    result = pi
    for symbol in word:
        matrix = matrices.get(symbol)
        if matrix is None:
            return 0.0
        result = result @ matrix
    return float(result @ ones)
