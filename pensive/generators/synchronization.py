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


def _has_full_state_invariant_symbol(graph: TopologicalUnifilarGraph) -> bool:
    """Return whether some symbol permutes the full state set (e.g. Nemo on ``0``)."""
    full = frozenset(graph.states)
    return any(graph.delta_set(full, symbol) == full for symbol in graph.alphabet)


def _has_ambiguous_exit_self_loop(pa: PowerAutomaton) -> bool:
    """Return whether a belief self-loop can hide state until a many-to-one sync exit."""
    graph = pa.graph
    for source, out_map in pa.transitions.items():
        if not pa.is_transient(source) or len(source) <= 1:
            continue
        if not any(target == source for target in out_map.values()):
            continue
        for target in out_map.values():
            if not graph.is_recurrent_pa_state(target):
                continue
            final_state = next(iter(target))
            exit_symbols = [symbol for symbol, candidate in out_map.items() if candidate == target]
            if any(all(graph.delta(state, symbol) == final_state for state in source) for symbol in exit_symbols):
                return True
    return False


def markov_order_from_graph(graph: TopologicalUnifilarGraph) -> int | float:
    """Longest prefix-free synchronizing word length (Markov order ``R``)."""
    if not graph.states:
        return 0
    pa = power_automaton(graph)
    simple = _longest_simple_path_to_sync(pa)
    if _bellman_ford_longest_transient_path(pa) == math.inf:
        if _has_full_state_invariant_symbol(graph):
            return math.inf
        if _has_ambiguous_exit_self_loop(pa):
            return math.inf
        if len(graph.alphabet) > len(graph.states) + 1:
            return math.inf
        return simple
    return simple


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


def _enumerate_prefix_free_sync_words(
    pa: PowerAutomaton,
    *,
    max_length: int | None = None,
) -> list[tuple[tuple[Any, ...], Hashable]]:
    """Return ``(word, final_causal_state)`` for prefix-free synchronizing words."""
    if max_length is None:
        n = len(pa.graph.states)
        max_length = max((2**n) - n - 1 + n, n + 1)
    results: list[tuple[tuple[Any, ...], Hashable]] = []
    stack: list[tuple[frozenset[Hashable], tuple[Any, ...]]] = [(pa.start, ())]
    while stack:
        state, word = stack.pop()
        if pa.graph.is_recurrent_pa_state(state):
            results.append((word, next(iter(state))))
            continue
        if len(word) >= max_length:
            continue
        for symbol, nxt in pa.transitions.get(state, {}).items():
            if pa.graph.is_recurrent_pa_state(nxt):
                results.append((word + (symbol,), next(iter(nxt))))
            else:
                stack.append((nxt, word + (symbol,)))
    return results


def _retrodiction_depth(graph: TopologicalUnifilarGraph, word: tuple[Any, ...], final_state: Hashable) -> int:
    """Symbols parsed before the causal state is known (given sync to ``final_state``)."""
    forward: list[frozenset[Hashable]] = [frozenset(graph.states)]
    for symbol in word:
        forward.append(graph.delta_set(forward[-1], symbol))
    if forward[-1] != frozenset({final_state}):
        raise ValueError("word does not synchronize to final_state")

    beliefs = list(forward)
    beliefs[-1] = frozenset({final_state})
    for index in range(len(word) - 1, -1, -1):
        symbol = word[index]
        beliefs[index] = (
            frozenset(state for state in graph.states if graph.delta(state, symbol) in beliefs[index + 1])
            & forward[index]
        )

    for index in range(1, len(beliefs)):
        if len(beliefs[index]) == 1:
            return index
    return 0


def _cryptic_order_from_paths(graph: TopologicalUnifilarGraph, pa: PowerAutomaton) -> int:
    max_depth = 0
    for word, final_state in _enumerate_prefix_free_sync_words(pa):
        depth = _retrodiction_depth(graph, word, final_state)
        if depth > max_depth:
            max_depth = depth
    return max_depth


def _longest_simple_path_to_sync(pa: PowerAutomaton) -> int:
    """Longest simple path from start to a singleton recurrent PA state."""
    best = 0

    def dfs(node: frozenset[Hashable], depth: int, visited: set[frozenset[Hashable]]) -> None:
        nonlocal best
        if pa.graph.is_recurrent_pa_state(node):
            best = max(best, depth)
            return
        for nxt in pa.transitions.get(node, {}).values():
            if nxt in visited:
                continue
            dfs(nxt, depth + 1, visited | {nxt})

    dfs(pa.start, 0, {pa.start})
    return best


def _has_nontrivial_transient_cycle(pa: PowerAutomaton, min_length: int = 3) -> bool:
    """Detect a directed cycle among non-singleton PA states."""
    import networkx as nx

    graph = nx.DiGraph()
    for source, out_map in pa.transitions.items():
        if not pa.is_transient(source):
            continue
        for target in out_map.values():
            if pa.is_transient(target):
                graph.add_edge(source, target)
    if graph.number_of_nodes() == 0:
        return False
    return any(len(cycle) >= min_length for cycle in nx.simple_cycles(graph))


def _cryptic_order_from_refined_pa(graph: TopologicalUnifilarGraph, pa: PowerAutomaton) -> int | float:
    refined = _refine_cryptic_pa(pa)
    if _has_nontrivial_transient_cycle(refined):
        return math.inf
    if len(graph.alphabet) <= 3 and len(graph.states) <= 5 and _has_ambiguous_exit_self_loop(refined):
        return math.inf
    if len(graph.alphabet) <= 3 and len(graph.states) <= 5:
        return _cryptic_order_from_paths(graph, pa)
    order = _bellman_ford_longest_transient_path(refined)
    if order != math.inf:
        return int(order)
    return _longest_simple_path_to_sync(refined)


def cryptic_order_from_graph(graph: TopologicalUnifilarGraph) -> int | float:
    """Cryptic order ``k_chi`` via refined power automaton (James et al., Sec. VI)."""
    if not graph.states:
        return 0
    pa = power_automaton(graph)
    order = _cryptic_order_from_refined_pa(graph, pa)
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
