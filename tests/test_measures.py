"""Tests for generator information measures."""

import pytest

from sofic.examples.epsilon_machines import (
    bernoulli,
    butterfly_process,
    golden_mean,
)
from sofic.generators.bidirectional_epsilon_machine import BidirectionalEpsilonMachine
from sofic.generators.epsilon_machine import EpsilonMachine
from sofic.generators.moore import MooreHMM
from sofic.generators.nmachine import NMachine
from sofic.graph import ATTR_EMISSION, ATTR_EMISSION_DIST, ATTR_PROB, ATTR_QUASIPROB

pytest.importorskip("dit")


def _epsilon() -> EpsilonMachine:
    eps = EpsilonMachine(
        initial_distribution={"q0": 1.0},
        observation_alphabet=frozenset({"0", "1"}),
    )
    eps.graph.add_state("q0")
    eps.graph.add_transition("q0", "q0", **{ATTR_PROB: 0.5, ATTR_EMISSION: "0"})
    eps.graph.add_transition("q0", "q0", **{ATTR_PROB: 0.5, ATTR_EMISSION: "1"})
    return eps


def _moore() -> MooreHMM:
    hmm = MooreHMM(
        initial_distribution={"q0": 1.0},
        observation_alphabet=frozenset({"0", "1"}),
    )
    hmm.graph.add_state("q0", **{ATTR_EMISSION_DIST: {"0": 0.5, "1": 0.5}})
    hmm.graph.add_transition("q0", "q0", **{ATTR_PROB: 1.0})
    return hmm


def test_state_distribution_matches_stationary_vector():
    eps = _epsilon()
    dist = eps.state_distribution()
    idx = eps.reindex()
    assert list(dist.outcomes) == [(state,) for state in idx.states]
    assert float(dist.pmf.sum()) == pytest.approx(1.0)
    for i, state in enumerate(idx.states):
        assert float(dist[(state,)]) == pytest.approx(eps.stationary_distribution()[i])


def test_state_distribution_edge_machine_tuple_states_roundtrip():
    """Edge-machine states are tuples; state_distribution must encode them losslessly."""
    from sofic.generators.edge_machine import parse_edge_state_label

    edge = bernoulli().to_edge_machine()
    dist = edge.state_distribution()
    idx = edge.reindex()
    assert float(dist.pmf.sum()) == pytest.approx(1.0)
    # Outcomes are single dit-safe labels that decode back to the tuple states.
    decoded = [parse_edge_state_label(outcome[0]) for outcome in dist.outcomes]
    assert set(decoded) == set(idx.states)


def test_state_entropy_edge_machine():
    """state_entropy must not crash on tuple-valued edge-machine states."""
    edge = bernoulli().to_edge_machine()
    assert edge.state_entropy() == pytest.approx(1.0, abs=1e-9)


def test_entropy_rate_epsilon():
    eps = _epsilon()
    assert eps.entropy_rate() == pytest.approx(1.0, abs=1e-6)


def test_entropy_rate_moore():
    hmm = _moore()
    assert hmm.entropy_rate() == pytest.approx(1.0, abs=1e-6)


def test_joint_block_distribution_respects_block_length_mealy():
    eps = _epsilon()
    dist = eps.joint_block_distribution(block_length=3)
    assert all(len(outcome) == 3 for outcome in dist.outcomes)
    assert dist[("0", "1", "0")] == pytest.approx(0.125, abs=1e-12)


def test_joint_block_distribution_supports_moore_hmms():
    hmm = _moore()
    dist = hmm.joint_block_distribution(block_length=3)
    assert all(len(outcome) == 3 for outcome in dist.outcomes)
    assert dist[("0", "1", "0")] == pytest.approx(0.125, abs=1e-12)


def test_state_entropy_and_statistical_complexity():
    eps = _epsilon()
    assert eps.state_entropy() == pytest.approx(0.0, abs=1e-6)
    assert eps.statistical_complexity() == pytest.approx(eps.state_entropy(), abs=1e-12)


def test_fair_coin_excess_entropy_and_crypticity():
    coin = bernoulli()
    assert coin.excess_entropy() == pytest.approx(0.0, abs=1e-9)
    assert coin.crypticity() == pytest.approx(0.0, abs=1e-9)


def test_golden_mean_excess_entropy():
    forward = golden_mean(0.5)
    reverse = golden_mean(0.5)
    bidir = BidirectionalEpsilonMachine.from_pair(forward, reverse)
    assert forward.excess_entropy() == pytest.approx(bidir.excess_entropy(), abs=1e-9)
    assert forward.excess_entropy() == pytest.approx(0.25162916738782304, abs=1e-9)


def test_golden_mean_crypticity():
    forward = golden_mean(0.5)
    assert forward.crypticity() == pytest.approx(forward.statistical_complexity() - forward.excess_entropy(), abs=1e-12)
    assert forward.crypticity() == pytest.approx(2.0 / 3.0, abs=1e-9)


def test_butterfly_statistical_complexity():
    butterfly = butterfly_process()
    pi = butterfly.stationary_distribution()
    dit = pytest.importorskip("dit")
    expected = float(dit.shannon.entropy(dit.Distribution([(i,) for i in range(len(pi))], pi)))
    assert butterfly.statistical_complexity() == pytest.approx(expected, abs=1e-9)


def test_collision_entropy_nmachine():
    nm = NMachine(
        initial_quasidistribution={"q0": 1.0},
        observation_alphabet=frozenset({"0", "1"}),
    )
    nm.graph.add_state("q0")
    nm.graph.add_transition("q0", "q0", **{ATTR_QUASIPROB: 0.6, ATTR_EMISSION: "0"})
    nm.graph.add_transition("q0", "q0", **{ATTR_QUASIPROB: 0.4, ATTR_EMISSION: "1"})
    assert nm.collision_entropy() >= 0.0


def test_bernoulli_state_distribution_single_outcome():
    eps = bernoulli(0.5)
    dist = eps.state_distribution()
    assert len(list(dist.outcomes)) == 1


def test_structural_information_is_excess_entropy_not_limiting_state_uncertainty():
    eps = golden_mean(0.5)
    excess = eps.excess_entropy()
    assert eps.structural_information() == pytest.approx(excess, abs=1e-12)
    diagram = eps.block_entropy_diagram(6)
    state_uncertainty = diagram.block_state_entropy - diagram.block_entropy
    assert state_uncertainty[-1] == pytest.approx(0.0, abs=1e-12)
    assert excess > 0.2
