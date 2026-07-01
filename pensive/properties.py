"""Structural predicates for pensive state-machine models."""

from __future__ import annotations

from collections.abc import Hashable, Iterable
from typing import Any

import networkx as nx
import numpy as np

from pensive.base import StateMachine
from pensive.graph import ATTR_EMISSION, ATTR_PROB, ATTR_SYMBOL, EPSILON


def is_unifilar_labeled(
    model: StateMachine,
    *,
    label_attr: str,
    exclude_labels: Iterable[Any] = (),
) -> bool:
    """Return whether each state has at most one outgoing edge per label value."""
    excluded = set(exclude_labels)
    seen: set[tuple[Hashable, Any]] = set()
    for transition in model.transitions():
        label = transition.data.get(label_attr)
        if label is None or label in excluded:
            continue
        key = (transition.source, label)
        if key in seen:
            return False
        seen.add(key)
    return True


def is_unifilar_symbols(model: StateMachine) -> bool:
    """Right-resolving on input symbols (ε excluded)."""
    return is_unifilar_labeled(model, label_attr=ATTR_SYMBOL, exclude_labels=(EPSILON,))


def is_unifilar_emissions(model: StateMachine) -> bool:
    """Row-unifilar on edge emissions (CM generator sense)."""
    return is_unifilar_labeled(model, label_attr=ATTR_EMISSION)


def is_counifilar_emissions(model: StateMachine) -> bool:
    """Column-unifilar on edge emissions: ``(target, emission)`` identifies the source."""
    seen: dict[tuple[Hashable, Any], Hashable] = {}
    for transition in model.transitions():
        emission = transition.data.get(ATTR_EMISSION)
        if emission is None:
            continue
        key = (transition.target, emission)
        source = seen.get(key)
        if source is not None and source != transition.source:
            return False
        seen[key] = transition.source
    return True


def is_deterministic_automaton(aut: Any) -> bool:
    """DFA-style determinism: one initial, no ε, unifilar on symbols."""
    if len(aut.initial_states) != 1:
        return False
    for transition in aut.transitions():
        if transition.data.get(ATTR_SYMBOL) is EPSILON:
            return False
    return is_unifilar_symbols(aut)


def is_deterministic_markov(chain: StateMachine) -> bool:
    """Each state has exactly one successor with probability 1."""
    for state in chain.states():
        outgoing = list(chain.graph.out_transitions(state))
        if len(outgoing) != 1:
            return False
        prob = float(outgoing[0].data.get(ATTR_PROB, 0.0))
        if not np.isclose(prob, 1.0):
            return False
    return True


def is_deterministic_transducer(tr: StateMachine) -> bool:
    """At most one transition per (state, input symbol)."""
    return is_unifilar_symbols(tr)


def is_irreducible(model: StateMachine) -> bool:
    """Return whether the directed state graph is strongly connected."""
    graph = _simple_digraph(model)
    if graph.number_of_nodes() <= 1:
        return True
    return bool(nx.is_strongly_connected(graph))


def recurrent_components(model: StateMachine) -> list[frozenset[Hashable]]:
    """Terminal strongly connected components of the directed state graph."""
    graph = _simple_digraph(model)
    components: list[frozenset[Hashable]] = []
    for component in nx.strongly_connected_components(graph):
        exits = any(target not in component for source in component for target in graph.successors(source))
        if not exits:
            components.append(frozenset(component))
    return components


def recurrent_states(model: StateMachine) -> frozenset[Hashable]:
    """States in terminal strongly connected components."""
    states: set[Hashable] = set()
    for component in recurrent_components(model):
        states.update(component)
    return frozenset(states)


def is_ergodic(model: StateMachine, *, weak: bool = True) -> bool:
    """Return weak/strong ergodicity of the internal finite-state dynamics.

    Weak ergodicity means all positive-mass initial states can reach exactly
    one terminal SCC. Strong ergodicity additionally requires that terminal
    component to be aperiodic.
    """
    graph = _simple_digraph(model)
    components = recurrent_components(model)
    if not components:
        return False

    initial = getattr(model, "initial_distribution", {})
    starts = {state for state, mass in initial.items() if mass > 0.0} if initial else set(model.states())

    reachable_terminal: list[frozenset[Hashable]] = []
    for component in components:
        if any(
            source in graph and target in nx.descendants(graph, source) | {source}
            for source in starts
            for target in component
        ):
            reachable_terminal.append(component)

    if len(reachable_terminal) != 1:
        return False
    if weak:
        return True

    subgraph = graph.subgraph(reachable_terminal[0]).copy()
    return bool(nx.is_aperiodic(subgraph))


def is_stationary(model: StateMachine, *, rtol: float = 1e-8, atol: float = 1e-10) -> bool:
    """Return whether ``initial_distribution`` is invariant under internal dynamics."""
    initial_distribution = getattr(model, "initial_distribution", None)
    if initial_distribution is None:
        return False
    vector, transition = _initial_vector_and_transition(model)
    return bool(np.allclose(vector @ transition, vector, rtol=rtol, atol=atol))


def is_detailed_balance(model: StateMachine, *, rtol: float = 1e-8, atol: float = 1e-10) -> bool:
    """Return whether stationary labeled flows satisfy detailed balance."""
    try:
        pi = np.asarray(model.stationary_distribution(), dtype=float)
    except Exception:
        pi, _transition = _initial_vector_and_transition(model)

    matrices = _labeled_or_internal_matrices(model)
    for matrix in matrices:
        flow = pi[:, None] * matrix
        if not np.allclose(flow, flow.T, rtol=rtol, atol=atol):
            return False
    return True


def is_periodic(model: StateMachine) -> bool:
    """Return whether every terminal component has graph period greater than one."""
    graph = _simple_digraph(model)
    components = recurrent_components(model)
    if not components:
        return False
    periodic_components = 0
    for component in components:
        subgraph = graph.subgraph(component).copy()
        if subgraph.number_of_nodes() == 0:
            continue
        if nx.is_aperiodic(subgraph):
            return False
        periodic_components += 1
    return periodic_components > 0


def is_strictly_sofic(model: Any) -> bool:
    """Return whether the HMM support appears strictly sofic.

    For finite right-resolving support presentations, finite Markov order is
    the finite-type case; infinite Markov order is strictly sofic.
    """
    from pensive.generators.conversions import hmm_to_support_dfa
    from pensive.generators.synchronization import (
        build_topological_graph_from_transitions,
        graph_from_unifilar_automaton,
        markov_order_from_graph,
    )

    support = model.to_sofic_shift()
    if support.is_unifilar():
        transitions: dict[tuple[Hashable, Any], Hashable] = {}
        alphabet = set(support.symbol_alphabet)
        for transition in support.transitions():
            symbol = transition.data.get(ATTR_SYMBOL)
            if symbol is None:
                continue
            transitions[(transition.source, symbol)] = transition.target
            alphabet.add(symbol)
        graph = build_topological_graph_from_transitions(
            states=frozenset(support.states()),
            alphabet=frozenset(alphabet),
            transitions=transitions,
        )
    else:
        graph = graph_from_unifilar_automaton(hmm_to_support_dfa(model))
    return markov_order_from_graph(graph) == float("inf")


def _simple_digraph(model: StateMachine) -> nx.DiGraph:
    graph = nx.DiGraph()
    graph.add_nodes_from(model.states())
    graph.add_edges_from((transition.source, transition.target) for transition in model.transitions())
    return graph


def transition_matrix(
    model: StateMachine,
    *,
    attr: str = ATTR_PROB,
    states: Iterable[Hashable] | None = None,
) -> tuple[np.ndarray, list[Hashable]]:
    """Return the dense state-to-state matrix accumulating edge ``attr`` weights.

    Rows/columns follow ``states`` when given (edges to states outside the set
    are ignored), otherwise all model states in iteration order. Returns the
    matrix together with the ordered state list defining its axes.
    """
    ordered = list(states) if states is not None else list(model.states())
    index = {state: i for i, state in enumerate(ordered)}
    n = len(ordered)
    matrix = np.zeros((n, n), dtype=float)
    for state in ordered:
        i = index[state]
        for transition in model.graph.out_transitions(state):
            j = index.get(transition.target)
            if j is None:
                continue
            matrix[i, j] += float(transition.data.get(attr, 0.0))
    return matrix, ordered


def _initial_vector_and_transition(model: StateMachine) -> tuple[np.ndarray, np.ndarray]:
    idx = model.reindex()
    n = len(idx)
    vector = np.zeros(n, dtype=float)
    for state, mass in getattr(model, "initial_distribution", {}).items():
        vector[idx.index(state)] = float(mass)
    transition, _states = transition_matrix(model, attr=ATTR_PROB, states=idx.states)
    return vector, transition


def _labeled_or_internal_matrices(model: StateMachine) -> list[np.ndarray]:
    observation_alphabet = getattr(model, "observation_alphabet", None)
    to_mealy = getattr(model, "to_mealy", None)
    if observation_alphabet is not None and to_mealy is not None:
        from pensive.generators.hmm_inference import _emission_transition_tensors_from_mealy

        _pi, matrices = _emission_transition_tensors_from_mealy(to_mealy())
        return list(matrices.values())

    _initial, transition = _initial_vector_and_transition(model)
    return [transition]
