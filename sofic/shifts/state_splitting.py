"""State splitting and amalgamation of edge shifts (Lind & Marcus §2.4).

Edges of a :class:`~sofic.shifts.tmc.TopologicalMarkovChain` are counted with
multiplicity, so an individual edge is named ``(source, target, key, copy)``:
the transition ``key`` between ``source`` and ``target`` and the copy index
``0 <= copy < multiplicity``. :func:`out_edges` and :func:`in_edges` list them.

Splitting a state ``I`` into ``m`` parts names the new states ``(I, 0), ...,
(I, m - 1)``; states left out of the partition keep their names.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Collection, Hashable, Mapping, Sequence
from typing import Any

import numpy as np

from sofic.graph import ATTR_MULTIPLICITY, ATTR_SYMBOL
from sofic.shifts.algorithms import edge_multiplicity, integer_adjacency_matrix
from sofic.shifts.tmc import TopologicalMarkovChain

Edge = tuple[Hashable, Hashable, int, int]
Partition = Mapping[Hashable, Sequence[Collection[Edge]]]


def out_edges(tmc: TopologicalMarkovChain, state: Hashable) -> list[Edge]:
    """Return the edges leaving ``state``, one per unit of multiplicity."""
    return [
        (t.source, t.target, t.key, copy)
        for t in tmc.transitions()
        if t.source == state
        for copy in range(edge_multiplicity(t.data))
    ]


def in_edges(tmc: TopologicalMarkovChain, state: Hashable) -> list[Edge]:
    """Return the edges entering ``state``, one per unit of multiplicity."""
    return [
        (t.source, t.target, t.key, copy)
        for t in tmc.transitions()
        if t.target == state
        for copy in range(edge_multiplicity(t.data))
    ]


def out_split(tmc: TopologicalMarkovChain, partitions: Partition) -> TopologicalMarkovChain:
    """Return the out-split graph ``G[P]`` :cite:`LindMarcus1995` (Definition 2.4.3).

    Each state ``I`` in ``partitions`` has its out-edges partitioned into
    nonempty parts ``E_I^0, ..., E_I^(m-1)``. An edge ``e`` from ``I`` to
    ``J`` in part ``i`` becomes one edge from ``(I, i)`` to every copy of
    ``J``: out-edges are split, in-edges are copied. Every copy keeps the
    symbol of ``e``, so erasing the split is a 1-block conjugacy of edge shifts
    :cite:`LindMarcus1995` (Theorem 2.4.10).

    Parameters
    ----------
    tmc : TopologicalMarkovChain
        The graph ``G``.
    partitions : Mapping
        ``{state: [part, ...]}`` where each part is a collection of
        :func:`out_edges` of ``state``; the parts must partition them.

    Returns
    -------
    TopologicalMarkovChain
        The split graph.
    """
    names, part_of = _split_names(tmc, partitions, out_edges)
    return _build_split(tmc, names, part_of, outgoing=True)


def in_split(tmc: TopologicalMarkovChain, partitions: Partition) -> TopologicalMarkovChain:
    """Return the in-split graph ``G[P]`` :cite:`LindMarcus1995` (Definition 2.4.7).

    Each state ``J`` in ``partitions`` has its in-edges partitioned into
    nonempty parts. An edge ``e`` from ``I`` to ``J`` in part ``j`` becomes
    one edge from every copy of ``I`` to ``(J, j)``: in-edges are split,
    out-edges are copied. The edge shift is conjugate to that of ``tmc``.

    Parameters
    ----------
    tmc : TopologicalMarkovChain
        The graph ``G``.
    partitions : Mapping
        ``{state: [part, ...]}`` where each part is a collection of
        :func:`in_edges` of ``state``; the parts must partition them.

    Returns
    -------
    TopologicalMarkovChain
        The split graph.
    """
    names, part_of = _split_names(tmc, partitions, in_edges)
    return _build_split(tmc, names, part_of, outgoing=False)


def out_split_matrices(tmc: TopologicalMarkovChain, partitions: Partition) -> tuple[np.ndarray, np.ndarray]:
    """Return the division and edge matrices ``(D, E)`` of an out-splitting.

    ``D(I, J^j) = [I = J]`` and ``E(I^i, J) = |E_I^i ∩ E^J|``, so that
    ``A_G = DE`` and ``A_{G[P]} = ED`` :cite:`LindMarcus1995` (Definition
    2.4.11, Theorem 2.4.12). Rows and columns follow ``tmc.states()`` and
    ``out_split(tmc, partitions).states()``.
    """
    names, part_of = _split_names(tmc, partitions, out_edges)
    return _split_matrices(tmc, names, part_of, outgoing=True)


def in_split_matrices(tmc: TopologicalMarkovChain, partitions: Partition) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(D, E)`` with ``A_G = DE`` and ``A_{G[P]} = ED`` for an in-splitting.

    In-splitting ``G`` is out-splitting its transpose
    :cite:`LindMarcus1995` (Exercise 2.4.9), so here ``E`` is the transposed
    division matrix, ``E(J^j, J) = 1``, and ``D(I, J^j) = |E^j_J ∩ E_I|`` counts
    the edges from ``I`` in part ``j`` of ``J``. Rows and columns follow
    ``tmc.states()`` and ``in_split(tmc, partitions).states()``.
    """
    names, part_of = _split_names(tmc, partitions, in_edges)
    return _split_matrices(tmc, names, part_of, outgoing=False)


def out_amalgamate(
    tmc: TopologicalMarkovChain, classes: Mapping[Hashable, Sequence[Hashable]]
) -> TopologicalMarkovChain:
    """Merge each class of states into one state, inverting an out-splitting.

    A class can be out-amalgamated when its states have identical columns in
    the adjacency matrix (every state sends the same number of edges to each
    of them); then ``A' = ED`` and ``A = DE`` with ``D`` the class division
    matrix :cite:`LindMarcus1995` (Definition 2.4.9, Theorem 2.4.14). The
    merged state keeps the out-edges of all members and the in-edges of the
    first listed member.

    Parameters
    ----------
    tmc : TopologicalMarkovChain
        The split graph.
    classes : Mapping
        ``{new_state: [state, ...]}``; unlisted states are kept as is.

    Returns
    -------
    TopologicalMarkovChain
        The amalgamated graph.
    """
    return _amalgamate(tmc, classes, outgoing=True)


def in_amalgamate(
    tmc: TopologicalMarkovChain, classes: Mapping[Hashable, Sequence[Hashable]]
) -> TopologicalMarkovChain:
    """Merge each class of states into one state, inverting an in-splitting.

    Dual of :func:`out_amalgamate`: members must have identical rows, and the
    merged state keeps the in-edges of all members and the out-edges of the
    first listed member :cite:`LindMarcus1995` (Definition 2.4.9, Exercise 2.4.9).
    """
    return _amalgamate(tmc, classes, outgoing=False)


def _split_names(
    tmc: TopologicalMarkovChain, partitions: Partition, edges_of: Any
) -> tuple[dict[Hashable, list[Hashable]], dict[Edge, int]]:
    states = list(tmc.states())
    names: dict[Hashable, list[Hashable]] = {state: [state] for state in states}
    part_of: dict[Edge, int] = {}
    for state, parts in partitions.items():
        if state not in names:
            raise ValueError(f"unknown state {state!r}")
        expected = set(edges_of(tmc, state))
        seen: set[Edge] = set()
        for index, part in enumerate(parts):
            part = set(part)
            if not part:
                raise ValueError(f"partition of {state!r} has an empty part")
            if part & seen or not part <= expected:
                raise ValueError(f"parts for {state!r} must be disjoint edges of that state")
            seen |= part
            part_of.update(dict.fromkeys(part, index))
        if seen != expected:
            raise ValueError(f"parts for {state!r} do not cover all of its edges")
        names[state] = [(state, index) for index in range(len(parts))]
    new_names = [name for state in states for name in names[state]]
    if len(set(new_names)) != len(new_names):
        raise ValueError("split state names collide with existing states")
    return names, part_of


def _part_counts(tmc: TopologicalMarkovChain, part_of: dict[Edge, int], transition: Any) -> Counter[int]:
    return Counter(
        part_of.get((transition.source, transition.target, transition.key, copy), 0)
        for copy in range(edge_multiplicity(transition.data))
    )


def _build_split(
    tmc: TopologicalMarkovChain,
    names: dict[Hashable, list[Hashable]],
    part_of: dict[Edge, int],
    *,
    outgoing: bool,
) -> TopologicalMarkovChain:
    result = type(tmc)(symbol_alphabet=tmc.symbol_alphabet)
    for state in tmc.states():
        for name in names[state]:
            result.graph.add_state(name)
    for transition in tmc.transitions():
        attrs = {k: v for k, v in transition.data.items() if k not in (ATTR_SYMBOL, ATTR_MULTIPLICITY)}
        symbol = transition.data.get(ATTR_SYMBOL)
        for part, count in _part_counts(tmc, part_of, transition).items():
            if outgoing:
                pairs = [(names[transition.source][part], target) for target in names[transition.target]]
            else:
                pairs = [(source, names[transition.target][part]) for source in names[transition.source]]
            for source, target in pairs:
                result.add_transition(source, target, symbol, **{ATTR_MULTIPLICITY: count, **attrs})
    return result


def _split_matrices(
    tmc: TopologicalMarkovChain,
    names: dict[Hashable, list[Hashable]],
    part_of: dict[Edge, int],
    *,
    outgoing: bool,
) -> tuple[np.ndarray, np.ndarray]:
    states = list(tmc.states())
    row = {state: i for i, state in enumerate(states)}
    column = {name: j for j, name in enumerate(name for state in states for name in names[state])}
    division = np.zeros((len(states), len(column)), dtype=int)
    for state in states:
        for name in names[state]:
            division[row[state], column[name]] = 1
    edge = np.zeros((len(column), len(states)), dtype=int)
    for transition in tmc.transitions():
        owner = transition.source if outgoing else transition.target
        other = transition.target if outgoing else transition.source
        for part, count in _part_counts(tmc, part_of, transition).items():
            edge[column[names[owner][part]], row[other]] += count
    if outgoing:
        return division, edge
    return edge.T, division.T


def _amalgamate(
    tmc: TopologicalMarkovChain, classes: Mapping[Hashable, Sequence[Hashable]], *, outgoing: bool
) -> TopologicalMarkovChain:
    matrix, states = integer_adjacency_matrix(tmc)
    index = {state: i for i, state in enumerate(states)}
    merged = {state: state for state in states}
    representative: dict[Hashable, Hashable] = {}
    assigned: set[Hashable] = set()
    for new_state, members in classes.items():
        members = list(members)
        if not members:
            raise ValueError(f"class {new_state!r} is empty")
        for member in members:
            if member not in index:
                raise ValueError(f"unknown state {member!r}")
            if member in assigned:
                raise ValueError(f"state {member!r} is in more than one class")
            assigned.add(member)
            merged[member] = new_state
        representative[new_state] = members[0]
        lines = [matrix[:, index[m]] if outgoing else matrix[index[m], :] for m in members]
        if any(list(line) != list(lines[0]) for line in lines):
            kind = "columns" if outgoing else "rows"
            raise ValueError(f"class {new_state!r} cannot be amalgamated: its {kind} differ")
    if (set(states) - assigned) & set(classes):
        raise ValueError("amalgamated state names collide with existing states")

    result = type(tmc)(symbol_alphabet=tmc.symbol_alphabet)
    for state in states:
        if not result.graph.has_state(merged[state]):
            result.graph.add_state(merged[state])
    for transition in tmc.transitions():
        kept = transition.target if outgoing else transition.source
        if merged[kept] in representative and representative[merged[kept]] != kept:
            continue
        attrs = {k: v for k, v in transition.data.items() if k != ATTR_SYMBOL}
        result.add_transition(
            merged[transition.source], merged[transition.target], transition.data.get(ATTR_SYMBOL), **attrs
        )
    return result
