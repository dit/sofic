"""Matrix-native quasi-realization (pi, D, tau)."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Self

import numpy as np

from pensive.generators.base import QuasiStochasticModel


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
            from pensive.exceptions import QuasiStochasticValidationError

            raise QuasiStochasticValidationError(f"pi sums to {self.pi.sum()}, not 1")

    def transition_matrices(self) -> dict[Any, np.ndarray]:
        return dict(self.symbol_maps)

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
        from pensive.generators.conversions import quasi_realization_from_nmachine

        return quasi_realization_from_nmachine(nm)

    def to_nmachine(self) -> Any:
        from pensive.generators.conversions import nmachine_from_quasi_realization

        return nmachine_from_quasi_realization(self)
