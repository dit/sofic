"""Shift presentations: factor languages, trimming, entropy."""

from __future__ import annotations

from collections import deque
from collections.abc import Hashable, Iterator
from typing import Any

import numpy as np

from sofic.graph import ATTR_MULTIPLICITY, ATTR_SYMBOL
from sofic.shifts.base import SymbolicModel


def trim_transient(model: SymbolicModel) -> SymbolicModel:
    """Remove states not on bi-infinite paths.

    Repeatedly deletes states with no incoming or no outgoing transition; what
    survives is exactly the set of states that lie on some bi-infinite path.
    """
    result = model.copy()
    graph = result.graph.nx
    pending = deque(state for state in graph.nodes if graph.in_degree(state) == 0 or graph.out_degree(state) == 0)
    while pending:
        state = pending.popleft()
        if state not in graph:
            continue
        neighbours = set(graph.predecessors(state)) | set(graph.successors(state))
        graph.remove_node(state)
        for neighbour in neighbours - {state}:
            if neighbour in graph and (graph.in_degree(neighbour) == 0 or graph.out_degree(neighbour) == 0):
                pending.append(neighbour)
    return result


def factor_language(model: SymbolicModel, length: int) -> Iterator[tuple[Any, ...]]:
    """Yield distinct factor words of the given length.

    Walks only the trimmed presentation (:func:`trim_transient`), so every word
    yielded extends to a bi-infinite point of the shift.
    """
    if length <= 0:
        yield ()
        return
    trimmed = trim_transient(model)
    seen: set[tuple[Any, ...]] = set()
    for start in trimmed.states():
        queue: deque[tuple[Hashable, tuple[Any, ...]]] = deque([(start, ())])
        while queue:
            state, prefix = queue.popleft()
            if len(prefix) == length:
                if prefix not in seen:
                    seen.add(prefix)
                    yield prefix
                continue
            for transition in trimmed.graph.out_transitions(state):
                symbol = transition.data.get(ATTR_SYMBOL)
                if symbol is None:
                    continue
                queue.append((transition.target, prefix + (symbol,)))


def adjacency_matrix(model: SymbolicModel, *, multiplicity: bool = True) -> tuple[np.ndarray, tuple[Hashable, ...]]:
    """Return the adjacency matrix and its state order.

    Each transition counts its ``multiplicity`` attribute when ``multiplicity``
    is true (edge-shift semantics), and once otherwise.
    """
    states = tuple(model.states())
    index = {state: i for i, state in enumerate(states)}
    n = len(states)
    matrix = np.zeros((n, n), dtype=float)
    for transition in model.transitions():
        i = index[transition.source]
        j = index[transition.target]
        matrix[i, j] += float(transition.data.get(ATTR_MULTIPLICITY, 1)) if multiplicity else 1.0
    return matrix, states


def topological_entropy_from_matrix(matrix: np.ndarray) -> float:
    """Return ``log2`` of the spectral radius of ``matrix`` (bits per symbol)."""
    if matrix.size == 0:
        return 0.0
    eigenvalues = np.linalg.eigvals(matrix)
    spectral_radius = float(np.max(np.abs(eigenvalues)))
    if spectral_radius <= 0.0:
        return 0.0
    return float(np.log2(spectral_radius))


def sofic_topological_entropy(model: SymbolicModel) -> float:
    """Topological entropy (bits) of the shift a labeled presentation generates.

    The ``log2`` spectral radius of a right-resolving presentation
    (:func:`~sofic.shifts.wheeler.right_resolve`), which counts words rather than
    paths :cite:`LindMarcus1995` (Theorem 4.3.3). Edge multiplicities are ignored: two
    parallel edges with one label present the same words as a single edge.
    """
    from sofic.shifts.wheeler import right_resolve

    matrix, _ = adjacency_matrix(right_resolve(model), multiplicity=False)
    return topological_entropy_from_matrix(matrix)
