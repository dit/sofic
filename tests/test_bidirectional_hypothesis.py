"""Property-based tests for bidirectional construction."""

from __future__ import annotations

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from pensive.examples import golden_mean_forward, golden_mean_reverse
from pensive.generators.bidirectional_epsilon_machine import BidirectionalEpsilonMachine


@given(p=st.floats(min_value=0.05, max_value=0.95))
@settings(max_examples=25, deadline=None)
@pytest.mark.measures
def test_golden_mean_bidirectional_always_three_states(p: float):
    pytest.importorskip("dit")
    forward = golden_mean_forward(p)
    reverse = golden_mean_reverse(p)
    bidir = BidirectionalEpsilonMachine.from_epsilon_machines(forward, reverse)
    assert len(list(bidir.states())) == 3
    joint = bidir.joint_distribution()
    assert len(joint) == 3
    total = sum(joint.values())
    assert total == pytest.approx(1.0, abs=1e-9)
