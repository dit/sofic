"""Topological synchronization orders for unifilar generators.

Markov order and cryptic order are computed via the power-automaton algorithms
of James, Mahoney, Ellison & Crutchfield (arXiv:1010.5545). Both depend only
on unifilar graph topology, not transition probabilities.
"""

from __future__ import annotations

import math
from collections.abc import Hashable, Mapping
from dataclasses import dataclass, field
from typing import Any

from pensive.exceptions import UnifilarityError
from pensive.graph import ATTR_EMISSION, ATTR_SYMBOL


@dataclass(frozen=True, slots=True)
class TopologicalUnifilarGraph:
    """Deterministic edge-labeled transition system (right-resolving)."""

    states: frozenset[Hashable]
    alphabet: frozenset[Any]
    transitions: Mapping[tuple[Hashable, Any], Hashable]

    def delta(self, state: Hashable, symbol: Any) -> Hashable | None:
        return self.transitions.get((state, symbol))

    def delta_set(self, states: frozenset[Hashable], symbol: Any) -> frozenset[Hashable]:
        successors: set[Hashable] = set()
        for state in states:
            target = self.delta(state, symbol)
            if target is not None:
                successors.add(target)
        return frozenset(successors)

    def is_recurrent_pa_state(self, pa_state: frozenset[Hashable]) -> bool:
        return len(pa_state) == 1 and next(iter(pa_state)) in self.states


@dataclass
class PowerAutomaton:
    """Subset automaton over a :class:`TopologicalUnifilarGraph`."""

    graph: TopologicalUnifilarGraph
    start: frozenset[Hashable]
    transitions: dict[frozenset[Hashable], dict[Any, frozenset[Hashable]]] = field(default_factory=dict)

    def is_transient(self, pa_state: frozenset[Hashable]) -> bool:
        return not self.graph.is_recurrent_pa_state(pa_state)


def build_topological_graph_from_transitions(
    states: frozenset[Hashable],
    alphabet: frozenset[Any],
    transitions: Mapping[tuple[Hashable, Any], Hashable],
) -> TopologicalUnifilarGraph:
    return TopologicalUnifilarGraph(states=states, alphabet=alphabet, transitions=dict(transitions))


def power_automaton(graph: TopologicalUnifilarGraph) -> PowerAutomaton:
    """Build the power automaton via subset construction from the full state set."""
    start = frozenset(graph.states)
    pa = PowerAutomaton(graph=graph, start=start)
    queue = [start]
    seen = {start}
    while queue:
        current = queue.pop(0)
        pa.transitions.setdefault(current, {})
        for symbol in graph.alphabet:
            successor = graph.delta_set(current, symbol)
            if not successor:
                continue
            pa.transitions[current][symbol] = successor
            if successor not in seen:
                seen.add(successor)
                queue.append(successor)
    return pa


def _bellman_ford_longest_transient_path(pa: PowerAutomaton) -> float:
    """Return longest prefix-free synchronizing word length, or ``math.inf``."""
    nodes = set(pa.transitions)
    for targets in pa.transitions.values():
        nodes.update(targets.values())
    nodes.add(pa.start)

    dist: dict[frozenset[Hashable], float] = dict.fromkeys(nodes, math.inf)
    dist[pa.start] = 0.0

    edges: list[tuple[frozenset[Hashable], frozenset[Hashable], int]] = []
    reverse_edges: dict[frozenset[Hashable], set[frozenset[Hashable]]] = {node: set() for node in nodes}
    for source, out_map in pa.transitions.items():
        for target in out_map.values():
            weight = -1 if pa.is_transient(source) else 0
            edges.append((source, target, weight))
            reverse_edges.setdefault(target, set()).add(source)

    sync_reachable: set[frozenset[Hashable]] = {
        node for node in nodes if pa.graph.is_recurrent_pa_state(node)
    }
    queue = list(sync_reachable)
    while queue:
        current = queue.pop(0)
        for predecessor in reverse_edges.get(current, set()):
            if predecessor in sync_reachable:
                continue
            sync_reachable.add(predecessor)
            queue.append(predecessor)

    edges = [(source, target, weight) for source, target, weight in edges if source in sync_reachable and target in sync_reachable]

    node_list = list(nodes)
    n = len(node_list)
    for _ in range(max(n - 1, 0)):
        updated = False
        for source, target, weight in edges:
            if dist[source] == math.inf:
                continue
            candidate = dist[source] + weight
            if candidate < dist[target]:
                dist[target] = candidate
                updated = True
        if not updated:
            break

    for source, target, weight in edges:
        if dist[source] != math.inf and dist[source] + weight < dist[target]:
            return math.inf

    best = math.inf
    for node in nodes:
        if not pa.graph.is_recurrent_pa_state(node):
            continue
        if dist[node] < best:
            best = dist[node]
    if best == math.inf:
        return math.inf
    if best > 0:
        return math.inf
    return -best


def markov_order_from_graph(graph: TopologicalUnifilarGraph) -> int | float:
    """Longest prefix-free synchronizing word length (Markov order ``R``)."""
    if not graph.states:
        return 0
    pa = power_automaton(graph)
    order = _bellman_ford_longest_transient_path(pa)
    if order == math.inf:
        return math.inf
    return int(order)


def _predecessors_on_symbol(graph: TopologicalUnifilarGraph, target: Hashable, symbol: Any) -> frozenset[Hashable]:
    return frozenset(state for state in graph.states if graph.delta(state, symbol) == target)


def _ensure_pa_state(
    transitions: dict[frozenset[Hashable], dict[Any, frozenset[Hashable]]],
    states: set[frozenset[Hashable]],
    graph: TopologicalUnifilarGraph,
    pa_state: frozenset[Hashable],
) -> None:
    """Ensure ``pa_state`` and its subset successors are present in the PA."""
    queue = [pa_state]
    while queue:
        current = queue.pop(0)
        if current in transitions:
            continue
        out_map: dict[Any, frozenset[Hashable]] = {}
        for symbol in graph.alphabet:
            successor = graph.delta_set(current, symbol)
            if successor:
                out_map[symbol] = successor
                if successor not in transitions:
                    queue.append(successor)
        transitions[current] = out_map
        states.add(current)


def _refine_cryptic_pa(pa: PowerAutomaton) -> PowerAutomaton:
    """Apply veracity refinement (James et al., Sec. VI.1) until quiescent."""
    graph = pa.graph
    transitions = {state: dict(out_map) for state, out_map in pa.transitions.items()}
    states = set(transitions)

    changed = True
    while changed:
        changed = False
        for source in list(states):
            out_map = transitions.get(source, {})
            for symbol, target in list(out_map.items()):
                if not graph.is_recurrent_pa_state(target):
                    continue
                target_state = next(iter(target))
                true_sources = _predecessors_on_symbol(graph, target_state, symbol) & source
                if not true_sources:
                    del out_map[symbol]
                    changed = True
                    continue
                if true_sources == source:
                    continue
                _ensure_pa_state(transitions, states, graph, true_sources)
                refined_out = transitions.setdefault(true_sources, {})
                if refined_out.get(symbol) != target:
                    refined_out[symbol] = target
                    changed = True
                if source != pa.start and symbol in out_map:
                    del out_map[symbol]
                    changed = True
            if not out_map:
                transitions.pop(source, None)
                states.discard(source)

    return PowerAutomaton(graph=graph, start=pa.start, transitions=transitions)


def _cryptic_order_from_refined_pa(pa: PowerAutomaton) -> int | float:
    refined = _refine_cryptic_pa(pa)
    order = _bellman_ford_longest_transient_path(refined)
    if order == math.inf:
        return math.inf
    return int(order)


def cryptic_order_from_graph(graph: TopologicalUnifilarGraph) -> int | float:
    """Cryptic order ``k_chi`` via refined power automaton (James et al., Sec. VI)."""
    if not graph.states:
        return 0
    pa = power_automaton(graph)
    markov_order = _bellman_ford_longest_transient_path(pa)
    order = _cryptic_order_from_refined_pa(pa)
    if markov_order != math.inf and (order == math.inf or order > markov_order):
        return int(markov_order)
    if order == math.inf:
        return math.inf
    return int(order)


def is_exactly_synchronizable(graph: TopologicalUnifilarGraph) -> bool:
    """Return whether the unifilar presentation has finite Markov order."""
    return markov_order_from_graph(graph) != math.inf


def graph_from_epsilon_machine(eps: Any) -> TopologicalUnifilarGraph:
    """Strip probabilities from an ε-machine into a topological graph."""
    from pensive.properties import is_unifilar_emissions

    if not is_unifilar_emissions(eps):
        for transition in eps.transitions():
            emission = transition.data.get(ATTR_EMISSION)
            if emission is None:
                continue
            key = (transition.source, emission)
            for other in eps.graph.out_transitions(transition.source):
                other_emission = other.data.get(ATTR_EMISSION)
                if other_emission == emission and other.target != transition.target:
                    raise UnifilarityError(f"non-unifilar duplicate emission {emission!r} from {transition.source!r}")

    transitions: dict[tuple[Hashable, Any], Hashable] = {}
    alphabet: set[Any] = set()
    for transition in eps.transitions():
        emission = transition.data.get(ATTR_EMISSION)
        if emission is None:
            continue
        key = (transition.source, emission)
        if key in transitions and transitions[key] != transition.target:
            raise UnifilarityError(f"non-unifilar duplicate emission {emission!r} from {transition.source!r}")
        transitions[key] = transition.target
        alphabet.add(emission)
    states = frozenset(eps.states())
    if eps.observation_alphabet:
        alphabet.update(eps.observation_alphabet)
    return TopologicalUnifilarGraph(
        states=states,
        alphabet=frozenset(alphabet),
        transitions=transitions,
    )


def graph_from_unifilar_automaton(aut: Any) -> TopologicalUnifilarGraph:
    """Build a topological graph from a :class:`~pensive.automata.unifilar.UnifilarAutomaton`."""
    from pensive.graph import EPSILON

    transitions: dict[tuple[Hashable, Any], Hashable] = {}
    alphabet: set[Any] = set()
    for transition in aut.transitions():
        symbol = transition.data.get(ATTR_SYMBOL)
        if symbol is None or symbol is EPSILON:
            continue
        key = (transition.source, symbol)
        if key in transitions and transitions[key] != transition.target:
            raise UnifilarityError(f"non-unifilar duplicate symbol {symbol!r} from {transition.source!r}")
        transitions[key] = transition.target
        alphabet.add(symbol)
    states = frozenset(aut.states())
    if aut.input_alphabet:
        alphabet.update(aut.input_alphabet)
    return TopologicalUnifilarGraph(
        states=states,
        alphabet=frozenset(alphabet),
        transitions=transitions,
    )


def graph_from_sofic_shift(shift: Any) -> TopologicalUnifilarGraph:
    """Build a topological graph from a right-resolving sofic presentation."""
    graph = graph_from_unifilar_automaton(shift)
    if shift.symbol_alphabet:
        missing = graph.alphabet - shift.symbol_alphabet
        if missing:
            raise UnifilarityError(f"shift alphabet does not cover transition symbols: {missing}")
    return graph
