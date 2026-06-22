"""Tests for MarkovDyckShift."""

import networkx as nx
import numpy as np
import pytest

from pensive.graph import KIND_CALL, KIND_RETURN, TransitionGraph
from pensive.shifts.markov_dyck import MarkovDyckShift


def _call(label):
    return KIND_CALL, label


def _return(label):
    return KIND_RETURN, label


def test_full_two_matrix_behaves_like_dyck_shift_order_two():
    shift = MarkovDyckShift.from_adjacency(np.ones((2, 2), dtype=int))

    shift.validate()
    assert shift.is_admissible_word((_call(0), _return(0)))
    assert shift.is_admissible_word((_call(0), _call(1), _return(1), _return(0)))
    assert not shift.is_admissible_word((_call(0), _return(1)))


def test_fibonacci_matrix_restricts_call_adjacencies():
    shift = MarkovDyckShift.from_adjacency(np.array([[1, 1], [1, 0]]), labels=("0", "1"))

    shift.validate()
    assert shift.is_admissible_word((_call("1"), _call("0")))
    assert not shift.is_admissible_word((_call("1"), _call("1")))


def test_return_after_call_must_have_same_label():
    shift = MarkovDyckShift.from_adjacency(np.ones((2, 2), dtype=int), labels=("x", "y"))

    assert shift.is_admissible_word((_call("x"), _return("x")))
    assert not shift.is_admissible_word((_call("x"), _return("y")))


def test_from_adjacency_rejects_bad_matrix():
    with pytest.raises(ValueError, match="square"):
        MarkovDyckShift.from_adjacency(np.ones((2, 3), dtype=int))


def test_from_adjacency_rejects_bad_labels():
    with pytest.raises(ValueError, match="labels length"):
        MarkovDyckShift.from_adjacency(np.eye(2), labels=("a",))
    with pytest.raises(ValueError, match="distinct"):
        MarkovDyckShift.from_adjacency(np.eye(2), labels=("a", "a"))


def test_vertex_type_graph_constructor():
    graph = nx.DiGraph()
    graph.add_edges_from([("A", "B"), ("B", "C"), ("C", "A")])

    shift = MarkovDyckShift.from_graph(graph, kind="vertex")

    shift.validate()
    assert shift.is_admissible_word((_call("A"), _call("B")))
    assert not shift.is_admissible_word((_call("B"), _call("A")))


def test_edge_type_graph_constructor():
    graph = nx.DiGraph()
    graph.add_edges_from([("A", "B"), ("B", "C"), ("C", "A")])
    edge_ab = ("A", "B")
    edge_bc = ("B", "C")

    shift = MarkovDyckShift.from_graph(graph, kind="edge")

    shift.validate()
    assert shift.is_admissible_word((_call(edge_ab), _call(edge_bc)))
    assert not shift.is_admissible_word((_call(edge_bc), _call(edge_ab)))


def test_transition_graph_vertex_constructor():
    graph = TransitionGraph()
    graph.add_transition("A", "B")
    graph.add_transition("B", "A")

    shift = MarkovDyckShift.from_graph(graph, kind="vertex")

    assert shift.is_admissible_word((_call("A"), _call("B")))


def test_from_graph_rejects_unknown_kind():
    graph = nx.DiGraph()

    with pytest.raises(ValueError, match="kind"):
        MarkovDyckShift.from_graph(graph, kind="unknown")
