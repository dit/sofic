"""Tests for edge-machine construction."""

from __future__ import annotations

import pytest

from pensive.examples import fair_coin, golden_mean
from pensive.generators.edge_machine import ATTR_EDGE_SOURCE, ATTR_EDGE_TARGET, hmm_to_edge_machine
from pensive.generators.mealy import MealyHMM
from pensive.graph import ATTR_EMISSION, ATTR_PROB


def _count_labeled_transitions(hmm) -> int:
    edges: set[tuple] = set()
    for transition in hmm.transitions():
        emission = transition.data.get(ATTR_EMISSION)
        if emission is None:
            continue
        prob = float(transition.data.get(ATTR_PROB, 0.0))
        if prob <= 0.0:
            continue
        edges.add((transition.source, emission, transition.target))
    return len(edges)


def _transitions_between(hmm, source, target):
    return [
        transition
        for transition in hmm.transitions()
        if transition.source == source and transition.target == target
    ]


def _duplicate_edge_hmm() -> MealyHMM:
    hmm = MealyHMM(
        initial_distribution={"A": 1.0},
        observation_alphabet=frozenset({"x", "y"}),
    )
    for state in ("A", "B", "C"):
        hmm.graph.add_state(state)
    hmm.add_transition("A", "B", "x", 0.2)
    hmm.add_transition("A", "B", "x", 0.3)
    hmm.add_transition("A", "C", "y", 0.5)
    hmm.add_transition("B", "A", "x", 1.0)
    hmm.add_transition("C", "A", "y", 1.0)
    return hmm


def _period_two_hmm() -> MealyHMM:
    hmm = MealyHMM(
        initial_distribution={"A": 1.0},
        observation_alphabet=frozenset({"a", "b"}),
    )
    hmm.graph.add_state("A")
    hmm.graph.add_state("B")
    hmm.add_transition("A", "B", "a", 1.0)
    hmm.add_transition("B", "A", "b", 1.0)
    return hmm


@pytest.mark.parametrize("builder", [fair_coin, golden_mean])
@pytest.mark.measures
def test_edge_machine_preserves_entropy_rate(builder):
    pytest.importorskip("dit")
    hmm = builder()
    edge = hmm.to_edge_machine()
    edge.validate()
    assert edge.entropy_rate() == pytest.approx(hmm.entropy_rate(), abs=1e-9)


@pytest.mark.parametrize("builder", [fair_coin, golden_mean])
@pytest.mark.measures
def test_edge_machine_preserves_block_distribution(builder):
    pytest.importorskip("dit")
    hmm = builder()
    edge = hmm.to_edge_machine()
    symbols = sorted(hmm.observation_alphabet, key=repr)
    dist_hmm0 = hmm.joint_block_distribution(history_length=0)
    dist_edge0 = edge.joint_block_distribution(history_length=0)
    for symbol in symbols:
        outcome = (symbol,)
        assert dist_hmm0[outcome] == pytest.approx(dist_edge0[outcome], abs=1e-9)

    dist_hmm1 = hmm.joint_block_distribution(history_length=1)
    dist_edge1 = edge.joint_block_distribution(history_length=1)
    for past in symbols:
        for present in symbols:
            outcome = (past, present)
            assert dist_hmm1[outcome] == pytest.approx(dist_edge1[outcome], abs=1e-9)


def test_edge_machine_state_count():
    hmm = golden_mean(0.5)
    edge = hmm_to_edge_machine(hmm)
    assert len(list(edge.states())) == _count_labeled_transitions(hmm)


def test_fair_coin_edge_machine_has_two_states():
    edge = fair_coin().to_edge_machine()
    assert len(list(edge.states())) == 2


def test_edge_machine_uses_cmpy_style_tuple_states():
    edge = golden_mean(0.5).to_edge_machine()

    state = ("A", 0, "A")
    assert state in set(edge.states())
    attrs = edge.graph.state_attrs(state)
    assert attrs[ATTR_EDGE_SOURCE] == "A"
    assert attrs[ATTR_EMISSION] == 0
    assert attrs[ATTR_EDGE_TARGET] == "A"


def test_edge_machine_sums_duplicate_labeled_edges():
    edge = _duplicate_edge_hmm().to_edge_machine()

    assert ("A", "x", "B") in set(edge.states())
    transitions = _transitions_between(edge, ("B", "x", "A"), ("A", "x", "B"))
    assert len(transitions) == 1
    assert transitions[0].data[ATTR_PROB] == pytest.approx(0.5)
    edge.validate()


def test_edge_machine_keeps_zero_probability_edge_states():
    hmm = MealyHMM(
        initial_distribution={"A": 1.0},
        observation_alphabet=frozenset({"0", "1"}),
    )
    hmm.graph.add_state("A")
    hmm.graph.add_state("B")
    hmm.add_transition("A", "A", "0", 1.0)
    hmm.add_transition("A", "B", "1", 0.0)
    hmm.add_transition("B", "A", "0", 1.0)

    edge = hmm.to_edge_machine()

    assert ("A", "1", "B") in set(edge.states())
    transitions = _transitions_between(edge, ("B", "0", "A"), ("A", "1", "B"))
    assert len(transitions) == 1
    assert transitions[0].data[ATTR_PROB] == pytest.approx(0.0)
    edge.validate()


def test_edge_machine_style_zero_pulls_and_style_one_pushes_symbols():
    hmm = _period_two_hmm()
    source = ("A", "a", "B")
    target = ("B", "b", "A")

    pull = _transitions_between(hmm.to_edge_machine(style=0), source, target)
    push = _transitions_between(hmm.to_edge_machine(style=1), source, target)

    assert pull[0].data[ATTR_EMISSION] == "b"
    assert push[0].data[ATTR_EMISSION] == "a"


def test_edge_machine_iterations_build_sliding_path_states():
    edge = _period_two_hmm().to_edge_machine(iterations=2)

    source = ("A", "a", "B", "b", "A")
    target = ("B", "b", "A", "a", "B")
    assert source in set(edge.states())
    assert target in set(edge.states())

    transitions = _transitions_between(edge, source, target)
    assert len(transitions) == 1
    assert transitions[0].data[ATTR_EMISSION] == "a"
    assert transitions[0].data[ATTR_PROB] == pytest.approx(1.0)
