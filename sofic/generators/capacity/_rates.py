"""Information rates of finite-state inputs driving a unifilar channel."""

from __future__ import annotations

import networkx as nx
import numpy as np


def unique_closed_class(matrix: np.ndarray) -> np.ndarray | None:
    """Return the states of the single closed class of ``matrix``, or ``None`` if there are several."""
    support = nx.DiGraph()
    support.add_nodes_from(range(matrix.shape[0]))
    support.add_edges_from(zip(*np.nonzero(matrix > 1e-12), strict=True))
    condensation = nx.condensation(support)
    closed = [c for c in condensation.nodes if condensation.out_degree(c) == 0]
    if len(closed) != 1:
        return None
    return np.array(sorted(condensation.nodes[closed[0]]["members"]))


def driven_rate_bounds(symbol_tensors: np.ndarray, noise: np.ndarray, block: int) -> tuple[float, float] | None:
    r"""Bracket :math:`\bar h(Y) - \mathbb{E}_\pi[\text{noise}]` for an ergodic edge-emitting HMM.

    ``symbol_tensors[y, z, z']`` is the hidden chain of a finite-state input
    driving a unifilar channel, and ``noise[z]`` is
    :math:`\mathbb{E}[H(Y \mid X, S) \mid Z = z]`. Because ``(s, x, y)``
    determines the next channel state, :math:`H(Y^n \| X^n) / n \to
    \mathbb{E}_\pi H(Y \mid X, S)`, so the result brackets the (directed)
    information rate; the entropy-rate bracket is
    :math:`H[Y_n \mid Y_{0:n}, Z_0] \le \bar h \le H[Y_n \mid Y_{0:n}]`
    :cite:`Cover2006`. Returns ``None`` unless the chain has one closed class.
    """
    from sofic.generators.measures import _iter_entropy_rate_bounds
    from sofic.generators.stationary import stationary_distribution_from_transition

    matrix = symbol_tensors.sum(axis=0)
    if unique_closed_class(matrix) is None:
        return None
    pi = stationary_distribution_from_transition(matrix)
    for index, (lower, upper, _words) in enumerate(_iter_entropy_rate_bounds(pi, symbol_tensors)):
        if index == block:
            conditional = float(pi @ noise)
            return lower - conditional, upper - conditional
    raise AssertionError("unreachable")


def default_block(n_outputs: int, bits: int) -> int:
    """Return the largest block length ``n >= 1`` with ``n_outputs ** n <= 2 ** bits``."""
    if n_outputs <= 1:
        return 1
    return max(1, int(bits / np.log2(n_outputs) + 1e-9))
