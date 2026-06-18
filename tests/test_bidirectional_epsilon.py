"""Tests for bidirectional epsilon machines."""

from __future__ import annotations

import math

import pytest

from pensive.dit_bridge import (
    bidirectional_statistical_complexity,
    crypticity,
    excess_entropy_bidirectional,
)
from pensive.examples import (
    ellison_fig9_forward,
    ellison_fig9_reverse,
    ellison_fig15_bidirectional,
    even_process,
    fair_coin,
    golden_mean_forward,
    golden_mean_reverse,
    tent_map_misiurewicz_bidirectional,
)
from pensive.generators.bidirectional_construction import infer_reverse_epsilon_machine
from pensive.generators.bidirectional_epsilon_machine import BidirectionalEpsilonMachine
from pensive.generators.epsilon_machine import EpsilonMachine
from pensive.generators.reversal import time_reverse_stochastic
from pensive.graph import ATTR_EMISSION, ATTR_PROB


def test_bidirectional_reversible_coin():
    coin = fair_coin()
    bidir = BidirectionalEpsilonMachine.from_epsilon_machines(coin, coin)
    bidir.validate()
    assert len(list(bidir.states())) == 1


@pytest.mark.measures
def test_reversible_excess_entropy_near_zero():
    dit = pytest.importorskip("dit")
    del dit
    coin = fair_coin()
    bidir = BidirectionalEpsilonMachine.from_epsilon_machines(coin, coin)
    assert excess_entropy_bidirectional(bidir) == pytest.approx(0.0, abs=1e-9)


def test_from_epsilon_machine_matches_reverse_pipeline():
    import networkx as nx

    forward = ellison_fig9_forward()
    derived = BidirectionalEpsilonMachine.from_epsilon_machine(forward)
    assert len(list(derived.states())) == 4
    assert len(list(nx.weakly_connected_components(derived.to_networkx()))) == 1
    joint = derived.joint_distribution()
    assert len(joint) == 4
    for mass in joint.values():
        assert mass == pytest.approx(0.25, abs=1e-9)
    assert derived.excess_entropy() == pytest.approx(0.5, abs=1e-9)


def test_from_epsilon_machine_golden_mean_forward_three_states():
    """Causally reversible golden mean yields three recurrent joint states."""
    import networkx as nx

    forward = golden_mean_forward(0.5)
    bidir = BidirectionalEpsilonMachine.from_epsilon_machine(forward)
    assert len(list(bidir.states())) == 3
    graph = bidir.to_networkx()
    assert len(list(nx.strongly_connected_components(graph))) == 1
    joint = bidir.joint_distribution()
    for mass in joint.values():
        assert mass == pytest.approx(1.0 / 3.0, abs=1e-9)


def test_from_epsilon_machine_golden_mean_shift_three_states():
    from pensive.examples import golden_mean

    import networkx as nx

    forward = golden_mean(0.5)
    bidir = BidirectionalEpsilonMachine.from_epsilon_machine(forward)
    assert len(list(bidir.states())) == 3
    graph = bidir.to_networkx()
    assert len(list(nx.strongly_connected_components(graph))) == 1


def test_from_epsilon_machine_transition_endpoints_canonical():
    forward = golden_mean_forward(0.5)
    bidir = BidirectionalEpsilonMachine.from_epsilon_machine(forward)
    states = list(bidir.states())
    for transition in bidir.transitions():
        assert any(transition.source is state for state in states)
        assert any(transition.target is state for state in states)


def test_marginalize_forward_fig9():
    forward = ellison_fig9_forward()
    reverse = ellison_fig9_reverse()
    bidir = BidirectionalEpsilonMachine.from_epsilon_machines(forward, reverse)
    recovered = bidir.marginalize_forward()
    for state in forward.states():
        for transition in forward.graph.out_transitions(state):
            symbol = transition.data.get(ATTR_EMISSION)
            prob = float(transition.data.get(ATTR_PROB, 0.0))
            match = [
                t
                for t in recovered.graph.out_transitions(state)
                if t.data.get(ATTR_EMISSION) == symbol and t.target == transition.target
            ]
            assert match
            assert float(match[0].data.get(ATTR_PROB, 0.0)) == pytest.approx(prob, abs=1e-9)


def test_marginalize_reverse_fig9():
    forward = ellison_fig9_forward()
    reverse = ellison_fig9_reverse()
    bidir = BidirectionalEpsilonMachine.from_epsilon_machines(forward, reverse)
    recovered = bidir.marginalize_reverse()
    joint = bidir.joint_distribution()
    pi_minus: dict[str, float] = {}
    for (_alpha, gamma), mass in joint.items():
        pi_minus[gamma] = pi_minus.get(gamma, 0.0) + mass
    for state in bidir.reverse_machine.states():
        if pi_minus.get(state, 0.0) <= 1e-15:
            continue
        for transition in bidir.reverse_machine.graph.out_transitions(state):
            symbol = transition.data.get(ATTR_EMISSION)
            prob = float(transition.data.get(ATTR_PROB, 0.0))
            match = [
                t
                for t in recovered.graph.out_transitions(state)
                if t.data.get(ATTR_EMISSION) == symbol and t.target == transition.target
            ]
            assert match
            assert float(match[0].data.get(ATTR_PROB, 0.0)) == pytest.approx(prob, abs=1e-9)


@pytest.mark.measures
def test_fig9_information_identities():
    pytest.importorskip("dit")
    forward = ellison_fig9_forward()
    reverse = ellison_fig9_reverse()
    bidir = BidirectionalEpsilonMachine.from_epsilon_machines(forward, reverse)

    c_plus = forward.statistical_complexity()
    c_minus = reverse.statistical_complexity()
    excess = excess_entropy_bidirectional(bidir)
    c_bidir = bidirectional_statistical_complexity(bidir)

    assert c_plus == pytest.approx(1.0, abs=1e-9)
    assert c_minus == pytest.approx(1.5, abs=1e-9)
    assert excess == pytest.approx(0.5, abs=1e-9)
    assert c_bidir == pytest.approx(2.0, abs=1e-9)
    assert c_bidir == pytest.approx(c_plus + c_minus - excess, abs=1e-9)


def test_bidirectional_golden_mean_paper_topology():
    """Ellison et al., arXiv:0905.3587 Fig. 4(c): three joint states, one SCC."""
    forward = golden_mean_forward(0.5)
    reverse = golden_mean_reverse(0.5)
    bidir = BidirectionalEpsilonMachine.from_epsilon_machines(forward, reverse)
    import networkx as nx

    graph = bidir.to_networkx()
    expected = {("A", "C"), ("A", "D"), ("B", "C")}
    assert set(bidir.states()) == expected
    assert len(list(nx.strongly_connected_components(graph))) == 1
    joint = bidir.joint_distribution()
    for state, mass in joint.items():
        assert mass == pytest.approx(1.0 / 3.0, abs=1e-9)
        assert state in expected


@pytest.mark.measures
def test_bidirectional_golden_mean_paper_invariants():
    """Fig. 5 spot-check at p = 1/2: C±μ, E, χ from arXiv:0905.3587."""
    pytest.importorskip("dit")
    p = 0.5
    forward = golden_mean_forward(p)
    reverse = golden_mean_reverse(p)
    bidir = BidirectionalEpsilonMachine.from_epsilon_machines(forward, reverse)

    c_plus = forward.statistical_complexity()
    c_minus = reverse.statistical_complexity()
    excess = excess_entropy_bidirectional(bidir)
    c_bidir = bidirectional_statistical_complexity(bidir)
    chi = crypticity(bidir)

    assert c_plus == pytest.approx(c_minus, abs=1e-9)
    assert c_plus == pytest.approx(math.log2(3) - 2 / 3, abs=1e-9)
    assert c_bidir == pytest.approx(math.log2(3), abs=1e-9)
    assert excess == pytest.approx(0.25162916738782304, abs=1e-9)
    assert chi == pytest.approx(4.0 / 3.0, abs=1e-9)
    assert c_bidir == pytest.approx(c_plus + c_minus - excess, abs=1e-9)
    assert c_bidir == pytest.approx(excess + chi, abs=1e-9)


def test_bidirectional_fig9_d_sector_is_connected():
    # Fig. 15 D-sector (reverse future symbol 1): (A,D) and (B,D) form one weak component.
    forward = ellison_fig9_forward()
    reverse = ellison_fig9_reverse()
    bidir = BidirectionalEpsilonMachine.from_epsilon_machines(forward, reverse)
    import networkx as nx

    graph = bidir.to_networkx()
    d_sector = {("A", "D"), ("B", "D")}
    subgraph = graph.subgraph(d_sector).copy()
    assert subgraph.number_of_edges() >= 2
    assert len(list(nx.weakly_connected_components(subgraph))) == 1


def test_bidirectional_copy_clears_joint_pi_cache():
    forward = golden_mean_forward(0.5)
    reverse = golden_mean_reverse(0.5)
    bidir = BidirectionalEpsilonMachine.from_epsilon_machines(forward, reverse)
    assert bidir._joint_pi is not None
    cloned = bidir.copy()
    assert cloned._joint_pi is None
    assert cloned.joint_distribution() == bidir.joint_distribution()


@pytest.mark.measures
def test_bidirectional_joint_pi_matches_marginals():
    """Joint π from IPF matches forward/reverse stationary marginals."""
    pytest.importorskip("dit")
    forward = golden_mean_forward(0.5)
    reverse = golden_mean_reverse(0.5)
    bidir = BidirectionalEpsilonMachine.from_epsilon_machines(forward, reverse)
    joint = bidir.joint_distribution()
    pi_plus = dict.fromkeys(forward.states(), 0.0)
    pi_minus = dict.fromkeys(bidir.reverse_machine.states(), 0.0)
    for (alpha, gamma), mass in joint.items():
        pi_plus[alpha] += mass
        pi_minus[gamma] += mass
    idx_f = forward.reindex()
    pi_f = forward.stationary_distribution()
    idx_r = bidir.reverse_machine.reindex()
    pi_r = bidir.reverse_machine.stationary_distribution()
    for i, state in enumerate(idx_f.states):
        assert pi_plus[state] == pytest.approx(float(pi_f[i]), abs=1e-9)
    for i, state in enumerate(idx_r.states):
        assert pi_minus[state] == pytest.approx(float(pi_r[i]), abs=1e-9)


@pytest.mark.measures
def test_bidirectional_tent_map_fig8_topology():
    """James et al. (2013) supplement Fig. 8: eight recurrent joint states."""
    bidir = tent_map_misiurewicz_bidirectional()
    # Supplement labels (S⁺, S⁻); reverse states relabel to E,F,G,H after collision-free merge.
    expected = {
        ("B", "E"),
        ("A", "F"),
        ("C", "G"),
        ("B", "G"),
        ("D", "E"),
        ("D", "F"),
        ("D", "G"),
        ("C", "F"),
    }
    assert set(bidir.states()) == expected


def _bidirectional_edge_set(bidir):
    return {
        (
            transition.source,
            transition.data.get(ATTR_EMISSION),
            transition.target,
            float(transition.data.get(ATTR_PROB, 0.0)),
        )
        for transition in bidir.transitions()
    }


def _epsilon_edge_set(eps):
    return {
        (
            transition.source,
            transition.data.get(ATTR_EMISSION),
            transition.target,
            float(transition.data.get(ATTR_PROB, 0.0)),
        )
        for transition in eps.transitions()
    }


def _assert_isomorphic_epsilon_machines(left, right):
    """Match transition tables up to a state relabeling."""
    left_edges = _epsilon_edge_set(left)
    right_edges = _epsilon_edge_set(right)
    if left_edges == right_edges:
        return
    left_states = sorted(left.states(), key=str)
    right_states = sorted(right.states(), key=str)
    assert len(left_states) == len(right_states)
    mapping = dict(zip(left_states, right_states, strict=True))

    def remap(edges):
        return {
            (mapping.get(source, source), symbol, mapping.get(target, target), prob)
            for source, symbol, target, prob in edges
        }

    assert remap(left_edges) == right_edges


def test_infer_reverse_matches_msp_pipeline():
    for factory in (ellison_fig9_forward, lambda: golden_mean_forward(0.5), lambda: even_process(0.5)):
        forward = factory()
        inferred = infer_reverse_epsilon_machine(forward)
        from_rev = EpsilonMachine.from_generator(time_reverse_stochastic(forward))
        _assert_isomorphic_epsilon_machines(inferred, from_rev)


def _assert_detailed_balance(forward):
    """Check π(i) P_ij = π(j) P̃_ji for the stochastic time reversal."""
    rev = time_reverse_stochastic(forward)
    pi = forward.stationary_distribution()
    idx = forward.reindex()
    for source in idx.states:
        i = idx.index(source)
        for transition in forward.graph.out_transitions(source):
            target = transition.target
            j = idx.index(target)
            symbol = transition.data.get(ATTR_EMISSION)
            forward_prob = float(transition.data.get(ATTR_PROB, 0.0))
            reverse_prob = next(
                float(t.data.get(ATTR_PROB, 0.0))
                for t in rev.graph.out_transitions(target)
                if t.target == source and t.data.get(ATTR_EMISSION) == symbol
            )
            assert pi[i] * forward_prob == pytest.approx(pi[j] * reverse_prob, abs=1e-9)


def _assert_marginalize_forward(forward, bidir):
    recovered = bidir.marginalize_forward()
    for state in forward.states():
        for transition in forward.graph.out_transitions(state):
            symbol = transition.data.get(ATTR_EMISSION)
            prob = float(transition.data.get(ATTR_PROB, 0.0))
            match = [
                t
                for t in recovered.graph.out_transitions(state)
                if t.data.get(ATTR_EMISSION) == symbol and t.target == transition.target
            ]
            assert match
            assert float(match[0].data.get(ATTR_PROB, 0.0)) == pytest.approx(prob, abs=1e-9)


def _assert_marginalize_reverse(bidir):
    recovered = bidir.marginalize_reverse()
    joint = bidir.joint_distribution()
    pi_minus: dict[str, float] = {}
    for (_alpha, gamma), mass in joint.items():
        pi_minus[gamma] = pi_minus.get(gamma, 0.0) + mass
    for state in bidir.reverse_machine.states():
        if pi_minus.get(state, 0.0) <= 1e-15:
            continue
        for transition in bidir.reverse_machine.graph.out_transitions(state):
            symbol = transition.data.get(ATTR_EMISSION)
            prob = float(transition.data.get(ATTR_PROB, 0.0))
            match = [
                t
                for t in recovered.graph.out_transitions(state)
                if t.data.get(ATTR_EMISSION) == symbol and t.target == transition.target
            ]
            assert match
            assert float(match[0].data.get(ATTR_PROB, 0.0)) == pytest.approx(prob, abs=1e-9)


def test_ellison_fig9_fig15_process():
    """Ellison et al., arXiv:1107.2168 Fig. 9 (M⁺, M⁻) and Fig. 15 (M±).

    Fig. 9 gives separate forward and reverse ε-machines for the same process.
    Time reversal links them at the process level; Eq. (15) builds the Fig. 15
    bidirectional presentation from that pair.
    """
    import networkx as nx

    forward = ellison_fig9_forward()
    reverse = ellison_fig9_reverse()
    bidir = ellison_fig15_bidirectional()

    # Time reversal: forward M⁺ and its stochastic reversal satisfy detailed balance.
    _assert_detailed_balance(forward)

    # Same stationary process (equal entropy rates on the two Fig. 9 presentations).
    assert forward.entropy_rate() == pytest.approx(reverse.entropy_rate(), abs=1e-9)

    # Eq. (15) marginalizes back to the Fig. 9 machines.
    _assert_marginalize_forward(forward, bidir)
    _assert_marginalize_reverse(bidir)

    # Fig. 15: four recurrent joint states with uniform stationary joint π.
    expected_states = {("A", "D"), ("A", "E"), ("B", "C"), ("B", "D")}
    assert set(bidir.states()) == expected_states
    joint = bidir.joint_distribution()
    assert set(joint) == expected_states
    for mass in joint.values():
        assert mass == pytest.approx(0.25, abs=1e-9)

    # Fig. 15 transition topology (reverse states relabelled C,D,E after collision-free merge).
    expected_edges = {
        (("A", "D"), 1, ("B", "C"), 0.5),
        (("A", "D"), 1, ("B", "D"), 0.5),
        (("A", "E"), 0, ("A", "D"), 0.5),
        (("A", "E"), 0, ("A", "E"), 0.5),
        (("B", "C"), 2, ("A", "D"), 0.5),
        (("B", "C"), 2, ("A", "E"), 0.5),
        (("B", "D"), 1, ("B", "C"), 0.5),
        (("B", "D"), 1, ("B", "D"), 0.5),
    }
    assert _bidirectional_edge_set(bidir) == expected_edges

    derived = BidirectionalEpsilonMachine.from_epsilon_machine(forward)
    assert set(derived.states()) == expected_states
    derived_joint = derived.joint_distribution()
    for state, mass in joint.items():
        assert derived_joint[state] == pytest.approx(mass, abs=1e-9)
    assert len(list(nx.weakly_connected_components(derived.to_networkx()))) == 1

    bidir.validate()


def test_ellison_fig15_single_weak_component():
    """Fig. 15 bidirectional machine is one weakly connected graph (arXiv:1107.2168)."""
    import networkx as nx

    bidir = ellison_fig15_bidirectional()
    graph = bidir.to_networkx()
    assert len(list(nx.weakly_connected_components(graph))) == 1
