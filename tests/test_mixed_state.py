"""Tests for mixed-state presentations."""

from __future__ import annotations

import pytest

from pensive.examples.epsilon_machines import bernoulli, even_process, golden_mean
from pensive.generators.mixed_state import MixedState, MixedStatePresentation, mixed_state_entropy
from pensive.generators.mixed_state_construction import build_mixed_state_presentation
from pensive.viz.graphviz import model_to_graphviz

graphviz = pytest.importorskip("graphviz")


def test_bernoulli_msp_is_single_pure_state():
    eps = bernoulli(0.5)
    msp = eps.mixed_state_presentation()
    assert isinstance(msp, MixedStatePresentation)
    assert len(list(msp.states())) == 1
    state = next(msp.states())
    assert msp.is_pure(state)
    assert msp.is_recurrent(state)
    assert not msp.is_transient(state)
    assert mixed_state_entropy(state) == pytest.approx(0.0)


def test_golden_mean_msp_has_transient_belief_states():
    eps = golden_mean(0.5)
    msp = build_mixed_state_presentation(eps)
    assert len(msp.transient_states) >= 1
    assert len(msp.pure_states) == 2
    assert msp.initial_mixed_state in msp.transient_states or msp.is_pure(msp.initial_mixed_state)
    assert msp.pure_states <= msp.recurrent_states


def test_msp_edges_are_valid_probabilities():
    eps = golden_mean(0.5)
    msp = eps.mixed_state_presentation()
    for state in msp.states():
        total = sum(transition.data["prob"] for transition in msp.graph.out_transitions(state))
        if list(msp.graph.out_transitions(state)):
            assert total == pytest.approx(1.0)


def test_even_process_msp_pure_states_name_causal_states():
    eps = even_process(0.5)
    msp = eps.mixed_state_presentation()
    named = {msp.causal_state(state) for state in msp.pure_states}
    assert named == {"A", "B"}


def test_epsilon_machine_viz_suppresses_start_node():
    source = model_to_graphviz(golden_mean()).source
    assert "__start__" not in source
    assert "π=" not in source


def test_msp_viz_highlights_initial_without_start_arrow():
    msp = golden_mean().mixed_state_presentation()
    source = model_to_graphviz(msp).source
    assert "__start__" not in source
    assert "penwidth=2.5" in source
    assert "μ=" in source


def test_mixed_state_canonicalization_merges_near_duplicates():
    left = MixedState.from_vector((1 / 3, 2 / 3))
    right = MixedState.from_vector((0.333333333333, 0.666666666667))
    assert left == right


def test_msp_deduplicates_canonical_beliefs():
    from pensive.generators.mealy import MealyHMM
    from pensive.graph import ATTR_EMISSION, ATTR_PROB

    hmm = MealyHMM(observation_alphabet=frozenset({0, 1}))
    for state in ("A", "B"):
        hmm.graph.add_state(state)
    hmm.graph.add_transition("A", "A", **{ATTR_PROB: 0.5, ATTR_EMISSION: 0})
    hmm.graph.add_transition("A", "B", **{ATTR_PROB: 0.5, ATTR_EMISSION: 0})
    hmm.graph.add_transition("B", "A", **{ATTR_PROB: 1.0, ATTR_EMISSION: 1})
    hmm.initial_distribution = {"A": 1.0}

    eta_a = MixedState.from_vector((1 / 3, 2 / 3))
    eta_b = MixedState.from_vector((0.333333333333, 0.666666666667))
    assert eta_a == eta_b
    msp_a = build_mixed_state_presentation(hmm, initial_mixed_state=eta_a)
    msp_b = build_mixed_state_presentation(hmm, initial_mixed_state=eta_b)
    assert len(list(msp_a.states())) == len(list(msp_b.states()))


def test_custom_initial_belief():
    eps = golden_mean(0.5)
    msp = build_mixed_state_presentation(eps, initial_mixed_state={"A": 1.0})
    assert msp.initial_mixed_state == MixedState.from_vector((1.0, 0.0))
    assert msp.initial_mixed_state in msp.pure_states
