"""Shared stochastic row/column validation helpers."""

from __future__ import annotations

import numpy as np

from pensive.exceptions import StochasticValidationError


def normalize_row_weights(weights: dict[tuple, float], *, atol: float = 1e-9) -> dict[tuple, float]:
    """Return ``weights`` scaled to sum to 1 when the total is positive."""
    total = sum(weights.values())
    if total <= 0.0:
        return {}
    if np.isclose(total, 1.0, atol=atol):
        return dict(weights)
    return {key: value / total for key, value in weights.items()}


def assert_stochastic_rows(matrix: np.ndarray, *, atol: float = 1e-9) -> None:
    """Raise if any row of ``matrix`` does not sum to 1."""
    row_sums = matrix.sum(axis=1)
    if not np.allclose(row_sums, 1.0, atol=atol):
        bad = np.where(~np.isclose(row_sums, 1.0, atol=atol))[0]
        raise StochasticValidationError(f"rows {bad.tolist()} do not sum to 1")
