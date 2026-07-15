"""Tests for recurrent components of mixed-state presentations."""

from __future__ import annotations

import pytest

from sofic.examples.epsilon_machines import golden_mean
from sofic.generators.epsilon_machine import EpsilonMachine
from sofic.generators.mealy import MealyHMM
from sofic.generators.mixed_state import MixedState, MixedStatePresentation
from sofic.graph import ATTR_EMISSION, ATTR_PROB, TransitionGraph


def test_to_recurrent_returns_epsilon_machine_for_pure_recurrent_msp():
    msp = golden_mean(0.5).mixed_state_presentation()
    assert msp.transient_states

    recurrent = msp.to_recurrent()

    assert isinstance(recurrent, EpsilonMachine)
    recurrent.validate()
    assert set(recurrent.states()) == {"A", "B"}
    assert set(recurrent.initial_distribution) <= set(recurrent.states())
    assert sum(recurrent.initial_distribution.values()) == pytest.approx(1.0)


def test_to_recurrent_returns_unifilar_hmm_for_nonpure_recurrent_msp():
    msp, recurrent_state = _nonpure_mixed_state_presentation()

    recurrent = msp.to_recurrent()

    assert isinstance(recurrent, MealyHMM)
    assert not isinstance(recurrent, EpsilonMachine)
    assert set(recurrent.states()) == {recurrent_state}
    assert set(recurrent.initial_distribution) == {recurrent_state}
    assert sum(recurrent.initial_distribution.values()) == pytest.approx(1.0)
    recurrent.validate_stochastic()
    assert recurrent.is_unifilar()
    assert all(transition.source == recurrent_state for transition in recurrent.transitions())
    assert all(transition.target == recurrent_state for transition in recurrent.transitions())


def _nonpure_mixed_state_presentation() -> tuple[MixedStatePresentation, MixedState]:
    transient_state = MixedState.from_vector((1.0, 0.0))
    recurrent_state = MixedState.from_vector((0.5, 0.5))
    assert transient_state is not None
    assert recurrent_state is not None

    graph = TransitionGraph()
    graph.add_state(transient_state)
    graph.add_state(recurrent_state)
    graph.add_transition(transient_state, recurrent_state, **{ATTR_PROB: 1.0, ATTR_EMISSION: 0})
    graph.add_transition(recurrent_state, recurrent_state, **{ATTR_PROB: 0.4, ATTR_EMISSION: 0})
    graph.add_transition(recurrent_state, recurrent_state, **{ATTR_PROB: 0.6, ATTR_EMISSION: 1})

    msp = MixedStatePresentation(
        graph=graph,
        basis_states=("A", "B"),
        initial_mixed_state=transient_state,
        pure_states=frozenset({transient_state}),
        recurrent_states=frozenset({recurrent_state}),
        transient_states=frozenset({transient_state}),
        initial_distribution={transient_state: 1.0},
        observation_alphabet=frozenset({0, 1}),
    )
    return msp, recurrent_state
