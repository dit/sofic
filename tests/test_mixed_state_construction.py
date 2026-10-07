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
