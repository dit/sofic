"""Q-graphs: finite quantizers of the channel-output history.

A Q-graph :cite:`Sabag2017` is a directed graph whose nodes carry one outgoing
edge per channel output symbol, so a walk from a fixed node maps every output
sequence to a node (a *context*). Capacity bounds then condition on the
context ``Q`` instead of the whole output past.
"""

from __future__ import annotations

from collections.abc import Hashable, Sequence
from dataclasses import dataclass
from itertools import product
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from sofic.automata.dfa import DFA
    from sofic.generators.capacity._channel import ChannelTensors


@dataclass(frozen=True, eq=False)
class QGraph:
    """Complete deterministic graph over an output alphabet.

    ``table[q, y]`` is the successor of node ``q`` on the ``y``-th symbol of
    ``outputs``; ``labels`` names the nodes.
    """

    outputs: tuple[Any, ...]
    table: np.ndarray
    labels: tuple[Hashable, ...]

    def __post_init__(self) -> None:
        table = np.asarray(self.table, dtype=int)
        if table.ndim != 2 or table.shape[1] != len(self.outputs):
            raise ValueError("table must have shape (n_nodes, n_outputs)")
        if table.size and (table.min() < 0 or table.max() >= table.shape[0]):
            raise ValueError("table entries must index nodes")
        if len(self.labels) != table.shape[0]:
            raise ValueError("one label per node is required")
        object.__setattr__(self, "table", table)

    @property
    def n_nodes(self) -> int:
        return self.table.shape[0]

    @classmethod
    def markov(cls, outputs: Sequence[Any], order: int) -> QGraph:
        """Return the order-``k`` Markov Q-graph whose nodes are the last ``k`` outputs."""
        if order < 0:
            raise ValueError("order must be nonnegative")
        outputs = tuple(outputs)
        words = list(product(range(len(outputs)), repeat=order))
        index = {word: i for i, word in enumerate(words)}
        table = np.zeros((len(words), len(outputs)), dtype=int)
        for word, q in index.items():
            for y in range(len(outputs)):
                table[q, y] = index[(*word, y)[1:]] if order else 0
        labels = tuple("".join(str(outputs[y]) for y in word) for word in words)
        return cls(outputs, table, labels)

    @classmethod
    def from_dfa(cls, dfa: DFA, outputs: Sequence[Any] | None = None) -> QGraph:
        """Build a Q-graph from a complete DFA over the output alphabet."""
        alphabet = tuple(outputs) if outputs is not None else tuple(sorted(dfa.input_alphabet, key=repr))
        states = sorted(dfa.states(), key=repr)
        index = {state: i for i, state in enumerate(states)}
        table = np.zeros((len(states), len(alphabet)), dtype=int)
        for state in states:
            for y, symbol in enumerate(alphabet):
                successors = dfa.delta(state, symbol)
                if len(successors) != 1:
                    raise ValueError(f"DFA is not complete at ({state!r}, {symbol!r})")
                table[index[state], y] = index[next(iter(successors))]
        return cls(alphabet, table, tuple(states))

    def aligned(self, outputs: Sequence[Any]) -> np.ndarray:
        """Return ``table`` with columns ordered as ``outputs``."""
        if set(outputs) != set(self.outputs):
            raise ValueError(f"Q-graph alphabet {self.outputs!r} differs from channel outputs {tuple(outputs)!r}")
        return self.table[:, [self.outputs.index(y) for y in outputs]]


def coupled_kernel(tensors: ChannelTensors, table: np.ndarray) -> np.ndarray:
    """Return ``K[s, q, x, s', q']``: the ``(S, Q)`` transition law given input ``x``."""
    n_s, n_x, n_y = tensors.P.shape
    n_q = table.shape[0]
    kernel = np.zeros((n_s, n_q, n_x, n_s, n_q))
    for s, x, y in zip(*np.nonzero(tensors.P > 0), strict=True):
        kernel[s, np.arange(n_q), x, tensors.F[s, x, y], table[:, y]] += tensors.P[s, x, y]
    return kernel


def coupled_chain(tensors: ChannelTensors, qgraph: QGraph, policy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return the ``(S, Q)`` transition matrix under ``policy[s, q, x]`` and a stationary law.

    States are flattened as ``s * |Q| + q``.
    """
    from sofic.generators.stationary import stationary_distribution_from_transition

    table = qgraph.aligned(tensors.outputs)
    kernel = coupled_kernel(tensors, table)
    n = tensors.n_states * qgraph.n_nodes
    matrix = np.einsum("sqx,sqxtr->sqtr", policy, kernel).reshape(n, n)
    return matrix, stationary_distribution_from_transition(matrix)
