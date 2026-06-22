"""Markov-Dyck shift constructions."""

from __future__ import annotations

from collections.abc import Hashable, Sequence
from typing import Any, Literal

import numpy as np

from pensive.graph import ATTR_KIND, ATTR_SYMBOL, KIND_CALL, KIND_RETURN, TransitionGraph
from pensive.shifts.sofic_dyck import MatchedEdge, SoficDyckShift, TransitionRef, transition_ref

GraphKind = Literal["edge", "vertex"]


class MarkovDyckShift(SoficDyckShift):
    """Markov-Dyck shift associated with a finite directed graph or matrix."""

    @classmethod
    def from_adjacency(
        cls,
        matrix: np.ndarray,
        labels: Sequence[Hashable] | None = None,
        **kwargs: Any,
    ) -> MarkovDyckShift:
        """Build the Markov-Dyck shift from a square adjacency matrix."""
        arr = _as_square_binary_matrix(matrix)
        resolved_labels = _resolve_labels(arr.shape[0], labels)
        return _from_binary_matrix(cls, arr, resolved_labels, **kwargs)

    @classmethod
    def from_graph(
        cls,
        graph: Any,
        *,
        kind: GraphKind = "edge",
        **kwargs: Any,
    ) -> MarkovDyckShift:
        """Build an edge- or vertex-type Markov-Dyck shift from ``graph``."""
        if kind == "edge":
            matrix, labels = _edge_adjacency(graph)
        elif kind == "vertex":
            matrix, labels = _vertex_adjacency(graph)
        else:
            raise ValueError("kind must be 'edge' or 'vertex'")
        return _from_binary_matrix(cls, matrix, labels, **kwargs)


def _from_binary_matrix(
    cls: type[MarkovDyckShift],
    matrix: np.ndarray,
    labels: tuple[Hashable, ...],
    **kwargs: Any,
) -> MarkovDyckShift:
    graph = TransitionGraph()
    call_symbols = tuple((KIND_CALL, label) for label in labels)
    return_symbols = tuple((KIND_RETURN, label) for label in labels)

    for symbol in (*call_symbols, *return_symbols):
        graph.add_state(symbol)

    row_intersections = _row_intersections(matrix)
    n = matrix.shape[0]
    for previous_index in range(n):
        for next_index in range(n):
            previous_call = call_symbols[previous_index]
            previous_return = return_symbols[previous_index]
            next_call = call_symbols[next_index]
            next_return = return_symbols[next_index]

            if matrix[next_index, previous_index]:
                graph.add_transition(previous_call, next_call, **{ATTR_KIND: KIND_CALL, ATTR_SYMBOL: next_call})
            if previous_index == next_index:
                graph.add_transition(previous_call, next_return, **{ATTR_KIND: KIND_RETURN, ATTR_SYMBOL: next_return})
            if row_intersections[next_index, previous_index]:
                graph.add_transition(previous_return, next_call, **{ATTR_KIND: KIND_CALL, ATTR_SYMBOL: next_call})
            if matrix[previous_index, next_index]:
                graph.add_transition(previous_return, next_return, **{ATTR_KIND: KIND_RETURN, ATTR_SYMBOL: next_return})

    matched_edges = _matched_edges_by_label(graph, labels)
    return cls(
        graph=graph,
        call_alphabet=frozenset(call_symbols),
        return_alphabet=frozenset(return_symbols),
        matched_edges=matched_edges,
        **kwargs,
    )


def _as_square_binary_matrix(matrix: np.ndarray) -> np.ndarray:
    arr = np.asarray(matrix)
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        raise ValueError("adjacency matrix must be square")
    return (arr != 0).astype(bool)


def _resolve_labels(n: int, labels: Sequence[Hashable] | None) -> tuple[Hashable, ...]:
    if labels is None:
        return tuple(range(n))
    resolved = tuple(labels)
    if len(resolved) != n:
        raise ValueError("labels length must match adjacency matrix size")
    if len(frozenset(resolved)) != n:
        raise ValueError("labels must be distinct")
    return resolved


def _row_intersections(matrix: np.ndarray) -> np.ndarray:
    n = matrix.shape[0]
    intersections = np.zeros((n, n), dtype=bool)
    for i in range(n):
        for j in range(n):
            intersections[i, j] = bool(np.any(matrix[i] & matrix[j]))
    return intersections


def _matched_edges_by_label(graph: TransitionGraph, labels: tuple[Hashable, ...]) -> frozenset[MatchedEdge]:
    call_edges: dict[Hashable, set[TransitionRef]] = {label: set() for label in labels}
    return_edges: dict[Hashable, set[TransitionRef]] = {label: set() for label in labels}
    for transition in graph.transitions():
        symbol = transition.data.get(ATTR_SYMBOL)
        if not isinstance(symbol, tuple) or len(symbol) != 2:
            continue
        role, label = symbol
        if role == KIND_CALL:
            call_edges[label].add(transition_ref(transition))
        elif role == KIND_RETURN:
            return_edges[label].add(transition_ref(transition))

    matched_edges: set[MatchedEdge] = set()
    for label in labels:
        for call_ref in call_edges[label]:
            for return_ref in return_edges[label]:
                matched_edges.add((call_ref, return_ref))
    return frozenset(matched_edges)


def _vertex_adjacency(graph: Any) -> tuple[np.ndarray, tuple[Hashable, ...]]:
    if isinstance(graph, TransitionGraph):
        labels = tuple(graph.states())
        index = {label: i for i, label in enumerate(labels)}
        matrix = np.zeros((len(labels), len(labels)), dtype=bool)
        for transition in graph.transitions():
            matrix[index[transition.target], index[transition.source]] = True
        return matrix, labels

    labels = tuple(graph.nodes())
    index = {label: i for i, label in enumerate(labels)}
    matrix = np.zeros((len(labels), len(labels)), dtype=bool)
    for source, target in graph.edges():
        matrix[index[target], index[source]] = True
    return matrix, labels


def _edge_adjacency(graph: Any) -> tuple[np.ndarray, tuple[Hashable, ...]]:
    endpoints: tuple[tuple[Hashable, Hashable], ...]
    if isinstance(graph, TransitionGraph):
        transitions = tuple(graph.transitions())
        labels = tuple(transition_ref(transition) for transition in transitions)
        endpoints = tuple((transition.source, transition.target) for transition in transitions)
    elif graph.is_multigraph():
        labels = tuple((source, target, key) for source, target, key in graph.edges(keys=True))
        endpoints = tuple((source, target) for source, target, _key in labels)
    else:
        labels = tuple((source, target) for source, target in graph.edges())
        endpoints = labels

    matrix = np.zeros((len(labels), len(labels)), dtype=bool)
    for previous, (_source_previous, target_previous) in enumerate(endpoints):
        for next_, (source_next, _target_next) in enumerate(endpoints):
            matrix[next_, previous] = target_previous == source_next
    return matrix, labels
