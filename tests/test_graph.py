"""Tests for TransitionGraph and Transition."""

import networkx as nx

from pensive.graph import (
    ATTR_PROB,
    ATTR_SYMBOL,
    EPSILON,
    Transition,
    TransitionGraph,
)


def test_parallel_edges():
    g = TransitionGraph()
    g.add_state("q0")
    g.add_state("q1")
    k1 = g.add_transition("q0", "q1", **{ATTR_SYMBOL: "a"})
    k2 = g.add_transition("q0", "q1", **{ATTR_SYMBOL: "b"})
    assert k1 != k2
    transitions = list(g.transitions())
    assert len(transitions) == 2
    symbols = {t.data[ATTR_SYMBOL] for t in transitions}
    assert symbols == {"a", "b"}


def test_attr_round_trip():
    g = TransitionGraph()
    g.add_state("s", emission="x")
    g.add_transition("s", "s", **{ATTR_SYMBOL: EPSILON, ATTR_PROB: 0.5})
    nx_g = g.nx
    assert nx_g.nodes["s"]["emission"] == "x"
    edge_data = nx_g.get_edge_data("s", "s")
    assert len(edge_data) == 1
    _, attrs = next(iter(edge_data.items()))
    assert attrs[ATTR_SYMBOL] is EPSILON
    assert attrs[ATTR_PROB] == 0.5

    g2 = TransitionGraph(nx_g.copy())
    t = next(g2.transitions())
    assert isinstance(t, Transition)
    assert t.source == "s"
    assert t.target == "s"
    assert t.data[ATTR_PROB] == 0.5
