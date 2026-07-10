"""Tests for bidirectional epsilon machines."""

from __future__ import annotations

import math

import pytest

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
    bidir = BidirectionalEpsilonMachine.from_pair(coin, coin)
    bidir.validate()
    assert len(list(bidir.states())) == 1


def test_reversible_excess_entropy_near_zero():
    dit = pytest.importorskip("dit")
    del dit
    coin = fair_coin()
    bidir = BidirectionalEpsilonMachine.from_pair(coin, coin)
    assert bidir.excess_entropy() == pytest.approx(0.0, abs=1e-9)


def test_from_forward_matches_reverse_pipeline():
    import networkx as nx

    forward = ellison_fig9_forward()
    derived = BidirectionalEpsilonMachine.from_forward(forward)
    assert len(list(derived.states())) == 4
    assert len(list(nx.weakly_connected_components(derived.to_networkx()))) == 1
    joint = derived.joint_distribution()
    assert len(joint) == 4
    for mass in joint.values():
        assert mass == pytest.approx(0.25, abs=1e-9)
    assert derived.excess_entropy() == pytest.approx(0.5, abs=1e-9)


def test_from_forward_golden_mean_forward_three_states():
    """Causally reversible golden mean yields three recurrent joint states."""
    import networkx as nx

    forward = golden_mean_forward(0.5)
    bidir = BidirectionalEpsilonMachine.from_forward(forward)
    assert len(list(bidir.states())) == 3
    graph = bidir.to_networkx()
    assert len(list(nx.strongly_connected_components(graph))) == 1
    joint = bidir.joint_distribution()
    for mass in joint.values():
        assert mass == pytest.approx(1.0 / 3.0, abs=1e-9)


def test_from_forward_golden_mean_shift_three_states():
    import networkx as nx

    from pensive.examples import golden_mean

    forward = golden_mean(0.5)
    bidir = BidirectionalEpsilonMachine.from_forward(forward)
    assert len(list(bidir.states())) == 3
    graph = bidir.to_networkx()
    assert len(list(nx.strongly_connected_components(graph))) == 1


def test_from_forward_transition_endpoints_canonical():
    forward = golden_mean_forward(0.5)
    bidir = BidirectionalEpsilonMachine.from_forward(forward)
    states = list(bidir.states())
    for transition in bidir.transitions():
        assert any(transition.source is state for state in states)
        assert any(transition.target is state for state in states)


def test_forward_epsilon_machine_fig9():
    forward = ellison_fig9_forward()
    reverse = ellison_fig9_reverse()
    bidir = BidirectionalEpsilonMachine.from_pair(forward, reverse)
    recovered = bidir.forward_epsilon_machine()
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


def test_reverse_epsilon_machine_fig9():
    forward = ellison_fig9_forward()
    reverse = ellison_fig9_reverse()
    bidir = BidirectionalEpsilonMachine.from_pair(forward, reverse)
    recovered = bidir.reverse_epsilon_machine()
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


def test_fig9_information_identities():
    pytest.importorskip("dit")
    forward = ellison_fig9_forward()
    reverse = ellison_fig9_reverse()
    bidir = BidirectionalEpsilonMachine.from_pair(forward, reverse)

    c_plus = forward.statistical_complexity()
    c_minus = reverse.statistical_complexity()
    excess = bidir.excess_entropy()
    c_bidir = bidir.statistical_complexity()

    assert c_plus == pytest.approx(1.0, abs=1e-9)
    assert c_minus == pytest.approx(1.5, abs=1e-9)
    assert excess == pytest.approx(0.5, abs=1e-9)
    assert c_bidir == pytest.approx(2.0, abs=1e-9)
    assert c_bidir == pytest.approx(c_plus + c_minus - excess, abs=1e-9)


def test_bidirectional_even_process_single_weak_component_and_anatomy():
    """Even process bidirectional machine is connected with b_μ = h_μ, r_μ = 0."""
    pytest.importorskip("dit")
    import networkx as nx

    bidir = even_process(0.5).to_bidirectional()
    graph = bidir.to_networkx()
    assert len(list(nx.weakly_connected_components(graph))) == 1
    h_mu = bidir.entropy_rate()
    assert bidir.ephemeral_information() == pytest.approx(0.0, abs=1e-9)
    assert bidir.bound_information() == pytest.approx(h_mu, abs=1e-9)
    assert bidir.bound_information() + bidir.ephemeral_information() == pytest.approx(h_mu, abs=1e-9)


def test_bidirectional_golden_mean_paper_topology():
    """Ellison et al., arXiv:0905.3587 Fig. 4(c): three joint states, one SCC."""
    forward = golden_mean_forward(0.5)
    reverse = golden_mean_reverse(0.5)
    bidir = BidirectionalEpsilonMachine.from_pair(forward, reverse)
    import networkx as nx

    graph = bidir.to_networkx()
    expected = {("A", "C"), ("A", "D"), ("B", "C")}
    assert set(bidir.states()) == expected
    assert len(list(nx.strongly_connected_components(graph))) == 1
    joint = bidir.joint_distribution()
    for state, mass in joint.items():
        assert mass == pytest.approx(1.0 / 3.0, abs=1e-9)
        assert state in expected


def test_bidirectional_golden_mean_paper_invariants():
    """Fig. 5 spot-check at p = 1/2: C±μ, E, χ from arXiv:0905.3587."""
    pytest.importorskip("dit")
    p = 0.5
    forward = golden_mean_forward(p)
    reverse = golden_mean_reverse(p)
    bidir = BidirectionalEpsilonMachine.from_pair(forward, reverse)

    c_plus = forward.statistical_complexity()
    c_minus = reverse.statistical_complexity()
    excess = bidir.excess_entropy()
    c_bidir = bidir.statistical_complexity()
    chi = bidir.crypticity()

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
    bidir = BidirectionalEpsilonMachine.from_pair(forward, reverse)
    import networkx as nx

    graph = bidir.to_networkx()
    d_sector = {("A", "D"), ("B", "D")}
    subgraph = graph.subgraph(d_sector).copy()
    assert subgraph.number_of_edges() >= 2
    assert len(list(nx.weakly_connected_components(subgraph))) == 1


def test_bidirectional_copy_clears_joint_pi_cache():
    forward = golden_mean_forward(0.5)
    reverse = golden_mean_reverse(0.5)
    bidir = BidirectionalEpsilonMachine.from_pair(forward, reverse)
    assert bidir._joint_pi is not None
    cloned = bidir.copy()
    assert cloned._joint_pi is None
    assert cloned.joint_distribution() == bidir.joint_distribution()


def test_bidirectional_joint_pi_matches_marginals():
    """Joint π from IPF matches forward/reverse stationary marginals."""
    pytest.importorskip("dit")
    forward = golden_mean_forward(0.5)
    reverse = golden_mean_reverse(0.5)
    bidir = BidirectionalEpsilonMachine.from_pair(forward, reverse)
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


def test_nemo_bidirectional_selects_true_recurrent_component():
    """Nemo admits a spurious all-0 3-cycle; the genuine 6-state class must be selected.

    Mahoney et al., arXiv:0906.5099. The Eq. (15) graph has two closed recurrent
    components; the degenerate one gives E = log2 3 > C_mu, which is impossible.
    """
    pytest.importorskip("dit")
    import networkx as nx

    from pensive.examples import nemo_process

    forward = nemo_process(0.5, 0.5)
    bidir = forward.to_bidirectional()

    assert len(list(bidir.states())) == 6
    graph = bidir.to_networkx()
    assert len(list(nx.weakly_connected_components(graph))) == 1

    c_mu = forward.statistical_complexity()
    assert bidir.excess_entropy() <= c_mu + 1e-9
    assert bidir.entropy_rate() == pytest.approx(0.75, abs=1e-9)

    # Joint marginals equal the forward/reverse causal-state stationary marginals.
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


def test_bidirectional_tent_map_fig8_edges():
    """Supplement Fig.~8 topology with symbolic ``1/2`` and ``a/(a+1)`` weights."""
    from pensive.examples.epsilon_machines import _tent_map_misiurewicz_fig8_edges, tent_map_misiurewicz_a

    a = tent_map_misiurewicz_a()
    half = 0.5
    inv_a1 = 1.0 / (a + 1.0)
    frac_a1 = a / (a + 1.0)
    expected = [
        (("B", "E"), 0, ("C", "G"), 1.0),
        (("C", "G"), 0, ("A", "F"), half),
        (("C", "G"), 1, ("D", "F"), half),
        (("A", "F"), 1, ("B", "E"), inv_a1),
        (("A", "F"), 1, ("B", "G"), frac_a1),
        (("B", "G"), 1, ("A", "F"), half),
        (("B", "G"), 0, ("C", "F"), half),
        (("D", "E"), 0, ("C", "G"), 1.0),
        (("D", "F"), 1, ("D", "E"), inv_a1),
        (("D", "F"), 1, ("D", "G"), frac_a1),
        (("D", "G"), 1, ("D", "F"), half),
        (("D", "G"), 0, ("C", "F"), half),
        (("C", "F"), 1, ("D", "G"), frac_a1),
        (("C", "F"), 1, ("D", "E"), inv_a1),
    ]
    actual = sorted(
        (source, symbol, target, prob) for source, target, symbol, prob in _tent_map_misiurewicz_fig8_edges(a)
    )
    assert len(actual) == len(expected)
    for got, want in zip(actual, sorted(expected), strict=True):
        assert got[:3] == want[:3]
        assert got[3] == pytest.approx(want[3], rel=0.0, abs=1e-12)


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
        from_rev = EpsilonMachine.from_hmm(time_reverse_stochastic(forward))
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


def _assert_forward_epsilon_machine(forward, bidir):
    recovered = bidir.forward_epsilon_machine()
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


def _assert_reverse_epsilon_machine(bidir):
    recovered = bidir.reverse_epsilon_machine()
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
    _assert_forward_epsilon_machine(forward, bidir)
    _assert_reverse_epsilon_machine(bidir)

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

    derived = BidirectionalEpsilonMachine.from_forward(forward)
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
