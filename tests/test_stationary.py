"""Tests for numeric stationary distributions."""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from sofic.generators.stationary import stationary_distribution_from_transition


def test_reducible_chain_warns_that_stationary_distribution_is_not_unique():
    transition = np.array([[0.5, 0.5, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    with pytest.warns(RuntimeWarning, match="not unique"):
        pi = stationary_distribution_from_transition(transition)
    assert pi @ transition == pytest.approx(pi)
    assert pi.sum() == pytest.approx(1.0)


@pytest.mark.parametrize(
    "transition",
    [
        np.array([[0.5, 0.5], [0.25, 0.75]]),
        np.array([[0.0, 1.0], [1.0, 0.0]]),
        np.array([[0.5, 0.5, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]]),
    ],
)
def test_unique_stationary_distribution_does_not_warn(transition):
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        pi = stationary_distribution_from_transition(transition)
    assert pi @ transition == pytest.approx(pi)
