"""TMC construction and Sofic conversion."""

from __future__ import annotations

from typing import Any

import numpy as np

from sofic.graph import ATTR_MULTIPLICITY, ATTR_SYMBOL, TransitionGraph
from sofic.shifts.algorithms import adjacency_matrix, topological_entropy_from_matrix
from sofic.shifts.sofic import SoficShift
from sofic.shifts.tmc import TopologicalMarkovChain
from sofic.states import sequential_labels


def from_adjacency(
    matrix: np.ndarray,
    symbol_alphabet: frozenset[Any] | None = None,
) -> TopologicalMarkovChain:
    arr = np.asarray(matrix, dtype=float)
    n = arr.shape[0]
    states = sequential_labels(n)
    graph = TransitionGraph()
    for state in states:
        graph.add_state(state)
    symbols = tuple(symbol_alphabet) if symbol_alphabet else tuple(range(n))
    for i in range(n):
        for j in range(n):
            count = int(arr[i, j])
            if count <= 0:
                continue
            symbol = symbols[j % len(symbols)] if symbol_alphabet else j
            graph.add_transition(states[i], states[j], **{ATTR_SYMBOL: symbol, ATTR_MULTIPLICITY: count})
    return TopologicalMarkovChain(
        graph=graph,
        symbol_alphabet=symbol_alphabet if symbol_alphabet is not None else frozenset(symbols),
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


def topological_entropy(model: TopologicalMarkovChain | SoficShift) -> float:
    matrix, _ = adjacency_matrix(model)
    return topological_entropy_from_matrix(matrix)
