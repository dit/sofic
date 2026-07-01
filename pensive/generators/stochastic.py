"""Shared stochastic row/column validation helpers."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from pensive.exceptions import StochasticValidationError


def shannon_entropy(values: Iterable[float], *, normalize: bool = False, atol: float = 0.0) -> float:
    """Shannon entropy (in bits) of ``values``.

    Values at or below ``atol`` are dropped. When ``normalize`` is true the
    retained values are rescaled to sum to one before the entropy is taken;
    otherwise they are assumed to already form a distribution.
    """
    probs = np.asarray(list(values), dtype=float)
    probs = probs[probs > atol]
    if probs.size == 0:
        return 0.0
    if normalize:
        total = float(probs.sum())
        if total <= 0.0:
            return 0.0
        probs = probs / total
    return float(-np.sum(probs * np.log2(probs)))


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
