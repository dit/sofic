"""Tests for mixed-state presentation enumeration limits."""

from __future__ import annotations

import time

import pytest

from sofic.examples import golden_mean, sns
from sofic.exceptions import MixedStateExplosionError
from sofic.generators.epsilon_machine import EpsilonMachine
from sofic.generators.mixed_state_construction import build_mixed_state_presentation


def test_infinite_belief_set_hits_max_states_promptly():
    hmm = sns()
    start = time.perf_counter()
    with pytest.raises(MixedStateExplosionError, match="max_states=10000"):
        hmm.mixed_state_presentation()
    with pytest.raises(MixedStateExplosionError, match="max_states=10000"):
        EpsilonMachine.from_hmm(hmm)
    assert time.perf_counter() - start < 20.0


def test_max_states_counts_distinct_beliefs():
    with pytest.raises(MixedStateExplosionError, match="max_states=50"):
        build_mixed_state_presentation(sns(), max_states=50)
    assert len(list(build_mixed_state_presentation(golden_mean(), max_states=3).states())) == 3


def test_infinite_reverse_presentation_reaches_its_cap_quickly() -> None:
    """Regression: indexing beliefs by their first coordinate made every lookup scan all states when
    many beliefs share it, so this 4-state reverse inference ran for many minutes before its cap."""
    import numpy as np

    from sofic.examples.epsilon_machines import from_symbol_matrices
    from sofic.generators.bidirectional_construction import infer_reverse_epsilon_machine

    mats = {x: np.zeros((4, 4)) for x in (0, 1)}
    for x, i, j, p in (
        (0, 1, 2, 0.903832833130349),
        (0, 2, 3, 0.9529570928839382),
        (0, 3, 0, 0.0202046632250718),
        (1, 0, 1, 1.0),
        (1, 1, 0, 0.0961671668696509),
        (1, 2, 2, 0.0470429071160618),
        (1, 3, 1, 0.9797953367749282),
    ):
        mats[x][i, j] = p
    forward = from_symbol_matrices(tuple(range(4)), (0, 1), mats)
    start = time.monotonic()
    with pytest.raises(MixedStateExplosionError):
        infer_reverse_epsilon_machine(forward, max_states=5_000)
    assert time.monotonic() - start < 60
