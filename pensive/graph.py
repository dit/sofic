"""NetworkX-backed transition graph wrapper."""

from __future__ import annotations

from collections.abc import Hashable, Iterator
from dataclasses import dataclass
from typing import Any

import networkx as nx

# Sentinel for NFA epsilon transitions.
EPSILON = object()

# Node attribute keys.
ATTR_EMISSION = "emission"
ATTR_EMISSION_DIST = "emission_dist"
ATTR_FUTURE_SYMBOL = "future_symbol"
ATTR_OUTPUT = "output"

# Edge attribute keys.
ATTR_SYMBOL = "symbol"
ATTR_PROB = "prob"
ATTR_QUASIPROB = "quasiprob"
ATTR_EMISSION_EDGE = "emission"
ATTR_KIND = "kind"
ATTR_STACK_SYMBOL = "stack_symbol"
ATTR_MULTIPLICITY = "multiplicity"

# VPA edge kinds.
KIND_CALL = "call"
KIND_RETURN = "return"
KIND_INTERNAL = "internal"


@dataclass(frozen=True, slots=True)
class Transition:
    """One directed transition in a :class:`TransitionGraph`."""

    source: Hashable
    target: Hashable
    key: int
    data: dict[str, Any]


class TransitionGraph:
    """NetworkX-backed directed multigraph for pensive models.

    States are nodes; transitions are edges with attribute dictionaries keyed by
    module constants such as :data:`ATTR_PROB` and :data:`ATTR_EMISSION`.

    Examples
    --------
    >>> from pensive.graph import TransitionGraph, ATTR_PROB
    >>> g = TransitionGraph()
    >>> g.add_state("A")
    >>> g.add_transition("A", "A", **{ATTR_PROB: 1.0})
    >>> len(list(g.transitions()))
    1
    """

    __slots__ = ("_g",)

    def __init__(self, graph: nx.MultiDiGraph | None = None) -> None:
        self._g = graph if graph is not None else nx.MultiDiGraph()

    @property
    def nx(self) -> nx.MultiDiGraph:
        return self._g

    def add_state(self, state: Hashable, **attrs: Any) -> None:
        self._g.add_node(state, **attrs)

    def add_transition(self, source: Hashable, target: Hashable, **attrs: Any) -> int:
        return self._g.add_edge(source, target, **attrs)

    def states(self) -> Iterator[Hashable]:
        yield from self._g.nodes

    def transitions(self) -> Iterator[Transition]:
        for source, target, key, data in self._g.edges(keys=True, data=True):
            yield Transition(source=source, target=target, key=key, data=dict(data))

    def copy(self) -> TransitionGraph:
        return TransitionGraph(self._g.copy())

    def has_state(self, state: Hashable) -> bool:
        return self._g.has_node(state)

    def state_attrs(self, state: Hashable) -> dict[str, Any]:
        return dict(self._g.nodes[state])

    def out_transitions(self, source: Hashable) -> Iterator[Transition]:
        for _target in self._g.successors(source):
            for key, data in self._g.get_edge_data(source, _target).items():
                yield Transition(source=source, target=_target, key=key, data=dict(data))

    def reverse(self) -> TransitionGraph:
        """Return a graph with the same nodes/attrs and all edges transposed."""
        reversed_graph = nx.MultiDiGraph()
        for state, attrs in self._g.nodes(data=True):
            reversed_graph.add_node(state, **dict(attrs))
        for source, target, _key, data in self._g.edges(keys=True, data=True):
            reversed_graph.add_edge(target, source, **dict(data))
        return TransitionGraph(reversed_graph)

    def forward_reachable(self, sources: set[Hashable] | frozenset[Hashable]) -> frozenset[Hashable]:
        """States reachable along directed edges from ``sources``."""
        from collections import deque

        if not sources:
            return frozenset()
        seen = set(sources)
        queue = deque(sources)
        while queue:
            state = queue.popleft()
            for transition in self.out_transitions(state):
                target = transition.target
                if target not in seen:
                    seen.add(target)
                    queue.append(target)
        return frozenset(seen)

    def terminal_recurrent_states(self) -> frozenset[Hashable]:
        """States in terminal strongly connected components (no exit to outside)."""
        recurrent: set[Hashable] = set()
        for component in nx.strongly_connected_components(self._g):
            if not component:
                continue
            exits_component = any(
                target not in component for source in component for _, target in self._g.out_edges(source)
            )
            if not exits_component:
                recurrent.update(component)
        return frozenset(recurrent)
