"""Topological synchronization orders for unifilar generators.

Markov order and cryptic order are computed via the power-automaton algorithms
of James, Mahoney, Ellison & Crutchfield (arXiv:1010.5545). Both depend only
on unifilar graph topology, not transition probabilities.
"""

from __future__ import annotations

import bisect
import math
from collections.abc import Hashable, Mapping
from dataclasses import dataclass, field
from typing import Any

import networkx as nx

from sofic.automata.wheeler import LabeledGraph, WheelerOrder, wheeler_order_of_graph
from sofic.exceptions import UnifilarityError
from sofic.graph import ATTR_EMISSION, ATTR_SYMBOL


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


def labeled_graph_of(graph: TopologicalUnifilarGraph) -> LabeledGraph:
    """View a topological graph as a :class:`~sofic.automata.wheeler.LabeledGraph`."""
    return LabeledGraph(
        states=tuple(sorted(graph.states, key=repr)),
        alphabet=tuple(sorted(graph.alphabet, key=repr)),
        edges=tuple((source, symbol, target) for (source, symbol), target in graph.transitions.items()),
    )


def power_automaton(graph: TopologicalUnifilarGraph) -> PowerAutomaton:
    """Build the power automaton by subset construction from the full state set.

    When ``graph`` is Wheeler the construction runs over co-lex *intervals*
    instead of arbitrary subsets. Path coherence guarantees the two agree
    :cite:`Gagie2017`, but the interval form visits at most ``n(n+1)/2`` states
    and steps in ``O(log n)`` rather than ``O(n)``, so the synchronization
    orders built on top of it stay polynomial.
    """
    order = wheeler_order_of_graph(labeled_graph_of(graph))
    if order is not None:
        return _interval_power_automaton(graph, order)
    return _subset_power_automaton(graph)


def _subset_power_automaton(graph: TopologicalUnifilarGraph) -> PowerAutomaton:
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


def _interval_power_automaton(graph: TopologicalUnifilarGraph, order: WheelerOrder) -> PowerAutomaton:
    """Subset construction restricted to intervals of a Wheeler order.

    Axiom two makes the target rank non-decreasing in the source rank for each
    symbol, so the image of a rank interval is bracketed by the first and last
    edges whose sources fall inside it -- two binary searches per step.
    """
    edges_by_symbol: dict[Any, tuple[list[int], list[int]]] = {}
    for symbol in graph.alphabet:
        pairs = sorted(
            (order.rank[source], order.rank[target])
            for (source, edge_symbol), target in graph.transitions.items()
            if edge_symbol == symbol
        )
        if pairs:
            edges_by_symbol[symbol] = ([source for source, _ in pairs], [target for _, target in pairs])

    def image(interval: tuple[int, int], symbol: Any) -> tuple[int, int] | None:
        found = edges_by_symbol.get(symbol)
        if found is None:
            return None
        sources, targets = found
        low = bisect.bisect_left(sources, interval[0])
        high = bisect.bisect_right(sources, interval[1]) - 1
        if low > high:
            return None
        return (targets[low], targets[high])

    def as_subset(interval: tuple[int, int]) -> frozenset[Hashable]:
        return frozenset(order.states[rank] for rank in range(interval[0], interval[1] + 1))

    start = (0, len(order.states) - 1)
    pa = PowerAutomaton(graph=graph, start=as_subset(start))
    queue = [start]
    seen = {start}
    while queue:
        current = queue.pop(0)
        out_map = pa.transitions.setdefault(as_subset(current), {})
        for symbol in graph.alphabet:
            successor = image(current, symbol)
            if successor is None:
                continue
            out_map[symbol] = as_subset(successor)
            if successor not in seen:
                seen.add(successor)
                queue.append(successor)
    return pa


def _power_automaton_digraph(pa: PowerAutomaton) -> nx.MultiDiGraph:
    """Materialize the power automaton as a networkx graph.

    Nodes are power-automaton states (frozensets of graph states). Each edge
    carries the emitted ``symbol`` that drives the subset transition and a
    ``weight`` of ``-1`` when the source is still unsynchronized (a non-
    singleton subset) and ``0`` once synchronized (a singleton). The negative
    transient weight turns the *longest* synchronizing path (Markov order) into
    a shortest-path problem solvable by Bellman-Ford. A
    :class:`networkx.MultiDiGraph` is used because two subsets can be joined by
    parallel edges labeled with different symbols.
    """
    digraph = nx.MultiDiGraph()
    digraph.add_node(pa.start)
    for source, out_map in pa.transitions.items():
        weight = -1 if pa.is_transient(source) else 0
        for symbol, target in out_map.items():
            digraph.add_edge(source, target, symbol=symbol, weight=weight)
    return digraph


def _states_reaching_synchronization(
    digraph: nx.MultiDiGraph, singletons: set[frozenset[Hashable]]
) -> set[frozenset[Hashable]]:
    """Return the singletons together with every node that can reach one.

    Computed as the ancestors of a virtual sink wired to every singleton, so a
    single reverse traversal covers all synchronized targets at once.
    """
    if not singletons:
        return set()
    reverse = digraph.reverse(copy=True)
    sink = object()
    for node in singletons:
        reverse.add_edge(sink, node)
    return nx.descendants(reverse, sink)


def _bellman_ford_longest_transient_path(pa: PowerAutomaton) -> float:
    """Return longest prefix-free synchronizing word length, or ``math.inf``.

    Restricts the power automaton to the nodes that can still reach a singleton
    (so a transient loop that never synchronizes is not mistaken for one that
    does), then solves the negative-weight shortest-path problem with networkx's
    Bellman-Ford. A negative cycle reachable from the start within that
    subgraph is an unbounded transient loop on a synchronizing path, i.e. an
    infinite Markov order, which networkx signals via ``NetworkXUnbounded``.
    """
    digraph = _power_automaton_digraph(pa)
    singletons = {node for node in digraph if pa.graph.is_recurrent_pa_state(node)}
    sync_reachable = _states_reaching_synchronization(digraph, singletons)
    if pa.start not in sync_reachable:
        return math.inf

    subgraph = digraph.subgraph(sync_reachable)
    try:
        dist = nx.single_source_bellman_ford_path_length(subgraph, pa.start, weight="weight")
    except nx.NetworkXUnbounded:
        return math.inf

    best = min((dist[node] for node in singletons if node in dist), default=math.inf)
    if best == math.inf:
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


def reset_threshold_from_graph(graph: TopologicalUnifilarGraph) -> int | float:
    """Reset threshold: length of the *shortest* synchronizing word.

    This is the shortest path from the full-support start of the power
    automaton to any singleton (synchronized) state -- the complement of the
    Markov order, which is the *longest* such prefix-free path. Returns ``0``
    for a single-state machine and ``math.inf`` when no finite synchronizing
    word exists (i.e. the presentation is not exactly synchronizable).

    The shortest-reset-word length is the quantity bounded by the Cerny
    conjecture for complete deterministic automata :cite:`Cerny1964`
    :cite:`Volkov2008`; note that right-resolving epsilon-machine presentations
    are generally *partial*, so that quadratic bound does not apply here. Like
    the Markov order, it is a topological property computed from the power
    automaton :cite:`James2010`.
    """
    if not graph.states:
        return 0
    pa = power_automaton(graph)
    lengths = nx.single_source_shortest_path_length(_power_automaton_digraph(pa), pa.start)
    singleton_lengths = [length for node, length in lengths.items() if pa.graph.is_recurrent_pa_state(node)]
    return min(singleton_lengths, default=math.inf)


def shortest_synchronizing_word_from_graph(
    graph: TopologicalUnifilarGraph,
) -> list[Any] | None:
    """Return a shortest synchronizing word, or ``None`` if none exists.

    A synchronizing (reset) word drives the observer's belief to a single
    state regardless of the start state :cite:`Travers2010`. The returned list
    of emitted symbols has length :func:`reset_threshold_from_graph`; it is
    empty for an already-synchronized single-state machine and ``None`` when
    the presentation is not exactly synchronizable.
    """
    if not graph.states:
        return []
    pa = power_automaton(graph)
    digraph = _power_automaton_digraph(pa)
    node_paths = nx.single_source_shortest_path(digraph, pa.start)
    best_path: list[frozenset[Hashable]] | None = None
    for node, node_path in node_paths.items():
        if pa.graph.is_recurrent_pa_state(node) and (best_path is None or len(node_path) < len(best_path)):
            best_path = node_path
    if best_path is None:
        return None
    ordered_symbols = sorted(graph.alphabet, key=repr)
    word: list[Any] = []
    for source, target in zip(best_path, best_path[1:], strict=False):
        symbol = next(sym for sym in ordered_symbols if graph.delta_set(source, sym) == target)
        word.append(symbol)
    return word


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
    """Return whether the presentation is exactly synchronizable.

    An epsilon-machine is *exact* iff it has some finite synchronizing word,
    equivalently iff ``Pr(SYN(M)) = 1`` -- the observer synchronizes to the
    hidden state in finite time for almost every generated sequence
    :cite:`Travers2010`. This holds iff the reset threshold is finite. Note
    this is strictly weaker than finite Markov order (see
    :func:`is_definite_from_graph`): the butterfly process is exact yet has
    infinite Markov order.
    """
    return reset_threshold_from_graph(graph) != math.inf


def is_definite_from_graph(graph: TopologicalUnifilarGraph) -> bool:
    """Return whether the presentation is a definite automaton (finite Markov order).

    A deterministic automaton is *definite* of degree ``R`` when the current
    state is fixed by the last ``R`` symbols regardless of the start state
    :cite:`Cerny1964`; this coincides with finite Markov order ``R`` for a
    right-resolving epsilon-machine :cite:`James2010`. Definiteness implies
    exact synchronizability, but not conversely.
    """
    return markov_order_from_graph(graph) != math.inf


def is_asymptotically_synchronizable_from_graph(graph: TopologicalUnifilarGraph) -> bool:
    """Return whether the presentation is asymptotically synchronizable.

    An epsilon-machine is asymptotically synchronizable iff the observer's
    state uncertainty vanishes as ``L -> infinity`` for almost every generated
    sequence (``Pr(WSYN(M)) = 1``) :cite:`Travers2010`. Every finite-state
    epsilon-machine has this property, so this returns ``True`` for any
    non-empty presentation -- it is theorem-backed rather than computed.
    """
    return bool(graph.states)


def graph_from_epsilon_machine(eps: Any) -> TopologicalUnifilarGraph:
    """Strip probabilities from an ε-machine into a topological graph."""
    from sofic.properties import is_unifilar_emissions

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
    """Build a topological graph from a :class:`~sofic.automata.unifilar.UnifilarAutomaton`."""
    from sofic.graph import EPSILON

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
