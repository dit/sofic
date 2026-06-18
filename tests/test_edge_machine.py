"""Tests for edge-machine construction."""

from __future__ import annotations

import pytest

from pensive.dit_bridge import joint_block_distribution
from pensive.examples import fair_coin, golden_mean
from pensive.generators.edge_machine import edge_machine_from_hmm
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


@pytest.mark.parametrize("builder", [fair_coin, golden_mean])
@pytest.mark.measures
def test_edge_machine_preserves_entropy_rate(builder):
    pytest.importorskip("dit")
    hmm = builder()
    edge = edge_machine_from_hmm(hmm)
    edge.validate()
    assert edge.entropy_rate() == pytest.approx(hmm.entropy_rate(), abs=1e-9)


@pytest.mark.parametrize("builder", [fair_coin, golden_mean])
@pytest.mark.measures
def test_edge_machine_preserves_block_distribution(builder):
    pytest.importorskip("dit")
    hmm = builder()
    edge = edge_machine_from_hmm(hmm)
    symbols = sorted(hmm.observation_alphabet, key=repr)
    dist_hmm0 = joint_block_distribution(hmm, history_length=0)
    dist_edge0 = joint_block_distribution(edge, history_length=0)
    for symbol in symbols:
        outcome = (symbol,)
        assert dist_hmm0[outcome] == pytest.approx(dist_edge0[outcome], abs=1e-9)

    dist_hmm1 = joint_block_distribution(hmm, history_length=1)
    dist_edge1 = joint_block_distribution(edge, history_length=1)
    for past in symbols:
        for present in symbols:
            outcome = (past, present)
            assert dist_hmm1[outcome] == pytest.approx(dist_edge1[outcome], abs=1e-9)


def test_edge_machine_state_count():
    hmm = golden_mean(0.5)
    edge = edge_machine_from_hmm(hmm)
    assert len(list(edge.states())) == _count_labeled_transitions(hmm)


def test_fair_coin_edge_machine_has_two_states():
    edge = edge_machine_from_hmm(fair_coin())
    assert len(list(edge.states())) == 2
