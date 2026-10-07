"""TMC construction and Sofic conversion."""

from __future__ import annotations

from typing import Any

import numpy as np

from sofic.graph import ATTR_MULTIPLICITY, ATTR_SYMBOL, TransitionGraph
from sofic.shifts.algorithms import adjacency_matrix, topological_entropy_from_matrix
from sofic.shifts.sofic import SoficShift
from sofic.shifts.tmc import TopologicalMarkovChain
from sofic.states import sequential_labels


def _sorted_symbols(symbols: frozenset[Any]) -> tuple[Any, ...]:
    try:
        return tuple(sorted(symbols))
    except TypeError:
        return tuple(sorted(symbols, key=repr))


def from_adjacency(
    matrix: np.ndarray,
    symbol_alphabet: frozenset[Any] | None = None,
    *,
    cls: type[TopologicalMarkovChain] = TopologicalMarkovChain,
    **kwargs: Any,
) -> TopologicalMarkovChain:
    """Build a TMC whose edge ``i -> j`` carries multiplicity ``matrix[i, j]``.

    Edges into state ``j`` are labeled ``j``, or with the ``j``-th symbol
    (cyclically) of ``symbol_alphabet`` in sorted order, so labels do not
    depend on frozenset iteration order.
    """
    arr = np.asarray(matrix, dtype=float)
    n = arr.shape[0]
    states = sequential_labels(n)
    graph = TransitionGraph()
    for state in states:
        graph.add_state(state)
    symbols = _sorted_symbols(symbol_alphabet) if symbol_alphabet else tuple(range(n))
    for i in range(n):
        for j in range(n):
            count = int(arr[i, j])
            if count <= 0:
                continue
            symbol = symbols[j % len(symbols)] if symbol_alphabet else j
            graph.add_transition(states[i], states[j], **{ATTR_SYMBOL: symbol, ATTR_MULTIPLICITY: count})
    return cls(
        graph=graph,
        symbol_alphabet=symbol_alphabet if symbol_alphabet is not None else frozenset(symbols),
        **kwargs,
    )


def to_sofic_shift(tmc: TopologicalMarkovChain) -> SoficShift:
    graph = TransitionGraph()
    for state in tmc.states():
        graph.add_state(state)
    for transition in tmc.transitions():
        multiplicity = int(transition.data.get(ATTR_MULTIPLICITY, 1))
        symbol = transition.data.get(ATTR_SYMBOL)
        for _ in range(max(multiplicity, 1)):
            graph.add_transition(
                transition.source,
                transition.target,
                **{ATTR_SYMBOL: symbol},
            )
    return SoficShift(graph=graph, symbol_alphabet=tmc.symbol_alphabet)


def topological_entropy(model: TopologicalMarkovChain) -> float:
    """Edge-shift entropy: ``log2`` spectral radius of the adjacency matrix with multiplicities."""
    matrix, _ = adjacency_matrix(model)
    return topological_entropy_from_matrix(matrix)
