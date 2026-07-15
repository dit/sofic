"""Shift presentations: factor languages, trimming, entropy."""

from __future__ import annotations

from collections import deque
from collections.abc import Hashable, Iterator
from typing import Any

import numpy as np

from sofic.graph import ATTR_SYMBOL
from sofic.shifts.base import SymbolicModel


def trim_transient(model: SymbolicModel) -> SymbolicModel:
    """Remove states not on bi-infinite paths."""
    forward = _forward_reachable(model)
    backward = _backward_reachable(model)
    keep = forward & backward
    result = model.copy()
    for state in list(result.states()):
        if state not in keep:
            result.graph.nx.remove_node(state)
    return result


def factor_language(model: SymbolicModel, length: int) -> Iterator[tuple[Any, ...]]:
    """Yield distinct factor words of the given length."""
    if length <= 0:
        yield ()
        return
    seen: set[tuple[Any, ...]] = set()
    for start in model.states():
        queue: deque[tuple[Hashable, tuple[Any, ...]]] = deque([(start, ())])
        while queue:
            state, prefix = queue.popleft()
            if len(prefix) == length:
                if prefix not in seen:
                    seen.add(prefix)
                    yield prefix
                continue
            for transition in model.graph.out_transitions(state):
                symbol = transition.data.get(ATTR_SYMBOL)
                if symbol is None:
                    continue
                queue.append((transition.target, prefix + (symbol,)))


def adjacency_matrix(model: SymbolicModel) -> tuple[np.ndarray, tuple[Hashable, ...]]:
    states = tuple(model.states())
    index = {state: i for i, state in enumerate(states)}
    n = len(states)
    matrix = np.zeros((n, n), dtype=float)
    for transition in model.transitions():
        i = index[transition.source]
        j = index[transition.target]
        matrix[i, j] += 1.0
    return matrix, states


def topological_entropy_from_matrix(matrix: np.ndarray) -> float:
    if matrix.size == 0:
        return 0.0
    eigenvalues = np.linalg.eigvals(matrix)
    spectral_radius = float(np.max(np.abs(eigenvalues)))
    return float(np.log(max(spectral_radius, 0.0)))


def _forward_reachable(model: SymbolicModel) -> set[Hashable]:
    return set(model.graph.forward_reachable(set(model.states())))


def _backward_reachable(model: SymbolicModel) -> set[Hashable]:
    reverse = model.graph.reverse()
    reachable: set[Hashable] = set()
    queue = deque(reverse.states())
    while queue:
        state = queue.popleft()
        if state in reachable:
            continue
        reachable.add(state)
        for transition in reverse.out_transitions(state):
            queue.append(transition.target)
    return reachable
