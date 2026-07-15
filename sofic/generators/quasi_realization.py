"""Matrix-native quasi-realization (pi, D, tau)."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from sofic.exceptions import QuasiStochasticValidationError
from sofic.generators.base import QuasiStochasticModel


class QuasiRealization(QuasiStochasticModel):
    """GPT quasi-realization quadruple (pi, D, tau)."""

    pi: np.ndarray
    tau: np.ndarray
    symbol_maps: dict[Any, np.ndarray]

    def __init__(
        self,
        pi: np.ndarray,
        tau: np.ndarray,
        symbol_maps: dict[Any, np.ndarray],
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.pi = np.asarray(pi, dtype=float)
        self.tau = np.asarray(tau, dtype=float)
        self.symbol_maps = {k: np.asarray(v, dtype=float) for k, v in symbol_maps.items()}

    def validate_quasistochastic(self) -> None:
        if not np.isclose(self.pi.sum(), 1.0):
            raise QuasiStochasticValidationError(f"pi sums to {self.pi.sum()}, not 1")

    def transition_matrices(self) -> dict[Any, np.ndarray]:
        return dict(self.symbol_maps)

    def stationary_quasidistribution(self) -> np.ndarray:
        if self.pi.size == 0:
            return np.array([], dtype=float)
        combined = np.zeros((self.pi.size, self.pi.size), dtype=float)
        for matrix in self.symbol_maps.values():
            combined += matrix
        return _stationary_left_quasivector(combined)

    def word_probability(self, word: Sequence[Any]) -> float:
        result = self.pi.copy()
        for symbol in word:
            matrix = self.symbol_maps.get(symbol)
            if matrix is None:
                return 0.0
            result = result @ matrix
        return float(result @ self.tau)

    @classmethod
    def from_nmachine(cls, nm: Any) -> QuasiRealization:
        from sofic.generators.conversions import quasi_realization_from_nmachine

        return quasi_realization_from_nmachine(nm)

    def to_nmachine(self) -> Any:
        from sofic.generators.conversions import nmachine_from_quasi_realization

        return nmachine_from_quasi_realization(self)


def _stationary_left_quasivector(transition: np.ndarray) -> np.ndarray:
    matrix = np.asarray(transition, dtype=float)
    n = matrix.shape[0]
    if matrix.shape != (n, n):
        raise ValueError("transition matrix must be square")

    eigenvalues, eigenvectors = np.linalg.eig(matrix.T)
    candidates = sorted(range(n), key=lambda i: abs(eigenvalues[i] - 1.0))
    for index in candidates:
        if not np.isclose(eigenvalues[index], 1.0, rtol=1e-9, atol=1e-10):
            continue
        vector = np.real_if_close(eigenvectors[:, index], tol=1000)
        if np.iscomplexobj(vector):
            continue
        distribution = np.asarray(vector, dtype=float)
        total = float(distribution.sum())
        if np.isclose(total, 0.0, atol=1e-12):
            continue
        distribution = distribution / total
        if np.allclose(distribution @ matrix, distribution, rtol=1e-8, atol=1e-10):
            return distribution

    augmented = np.vstack([matrix.T - np.eye(n), np.ones(n)])
    target = np.zeros(n + 1, dtype=float)
    target[-1] = 1.0
    solution, *_ = np.linalg.lstsq(augmented, target, rcond=None)
    if not np.allclose(solution @ matrix, solution, rtol=1e-8, atol=1e-10):
        raise QuasiStochasticValidationError("failed to compute an invariant stationary quasidistribution")
    return solution
