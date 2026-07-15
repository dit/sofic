"""Tests for stochastic helpers."""

from __future__ import annotations

import numpy as np
import pytest

from sofic.exceptions import StochasticValidationError
from sofic.generators.stochastic import assert_stochastic_rows, normalize_row_weights


def test_normalize_row_weights():
    result = normalize_row_weights({("a", 0): 2.0, ("b", 1): 2.0})
    assert sum(result.values()) == pytest.approx(1.0)
    assert result[("a", 0)] == pytest.approx(0.5)


def test_assert_stochastic_rows():
    matrix = np.array([[0.5, 0.5], [0.25, 0.75]])
    assert_stochastic_rows(matrix)


def test_assert_stochastic_rows_raises():
    matrix = np.array([[0.5, 0.4], [0.25, 0.75]])
    with pytest.raises(StochasticValidationError):
        assert_stochastic_rows(matrix)
