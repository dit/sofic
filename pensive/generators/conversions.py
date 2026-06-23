"""Generator presentation conversions."""

from __future__ import annotations

from collections.abc import Hashable
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from pensive.generators.synchronization import TopologicalUnifilarGraph

from pensive.automata.dfa import DFA
from pensive.automata.nfa import NFA
from pensive.generators.base import HiddenMarkovModel
from pensive.generators.mealy import MealyHMM
from pensive.generators.moore import MooreHMM
from pensive.generators.nmachine import NMachine
from pensive.generators.pfa import ProbabilisticFiniteAutomaton
from pensive.generators.quasi_realization import QuasiRealization
from pensive.graph import (
    ATTR_EMISSION,
    ATTR_EMISSION_DIST,
    ATTR_PROB,
    ATTR_QUASIPROB,
    ATTR_SYMBOL,
    EPSILON,
    TransitionGraph,
)
from pensive.shifts.sofic import SoficShift
from pensive.states import sequential_labels


def moore_to_mealy(moore: MooreHMM) -> MealyHMM:
    """Convert Moore HMM to Mealy with joint P(q', o | q) on departure state."""
    graph = TransitionGraph()
    for state in moore.states():
        graph.add_state(state)
    for transition in moore.transitions():
        source = transition.source
        target = transition.target
        trans_prob = float(transition.data.get(ATTR_PROB, 0.0))
        emission_dist = moore.graph.state_attrs(source).get(ATTR_EMISSION_DIST, {})
        for emission, emit_prob in emission_dist.items():
            joint = trans_prob * float(emit_prob)
            if joint > 0.0:
                graph.add_transition(
                    source,
                    target,
                    **{ATTR_PROB: joint, ATTR_EMISSION: emission},
                )
    return MealyHMM(
        graph=graph,
        initial_distribution=moore.initial_distribution,
        observation_alphabet=moore.observation_alphabet,
    )


def pfa_to_mealy(pfa: ProbabilisticFiniteAutomaton) -> MealyHMM:
    graph = pfa.graph.copy()
    return MealyHMM(
        graph=graph,
        initial_distribution=pfa.initial_distribution,
        observation_alphabet=frozenset(pfa.output_alphabet),
    )


def hmm_to_sofic_shift(hmm: HiddenMarkovModel) -> SoficShift:
    """Strip probabilities from an HMM and keep its labeled support."""
    support = _mealy_support(hmm)
    graph = _support_graph(support, edge_attr=ATTR_SYMBOL)
    return SoficShift(graph=graph, symbol_alphabet=support.observation_alphabet)


def hmm_to_support_nfa(hmm: HiddenMarkovModel) -> NFA:
    """Build an NFA whose language is the finite-word support of an HMM."""
    support = _mealy_support(hmm)
    graph = _support_graph(support, edge_attr=ATTR_SYMBOL)
    states = frozenset(support.states())
    start = _fresh_start_state(states)
    graph.add_state(start)
    for state in states:
        graph.add_transition(start, state, **{ATTR_SYMBOL: EPSILON})
    return NFA(
        graph=graph,
        input_alphabet=support.observation_alphabet,
        initial_states=frozenset({start}),
        accepting_states=states,
    )


def hmm_to_support_dfa(hmm: HiddenMarkovModel) -> DFA:
    """Determinize the HMM support NFA from the all-states subset."""
    support = _mealy_support(hmm)
    states = frozenset(support.states())
    nfa = hmm_to_support_nfa(support)
    nfa.initial_states = states

    dfa = nfa.determinize(alphabet=support.observation_alphabet)
    empty_subset = frozenset()
    if dfa.graph.has_state(empty_subset):
        dfa.graph.nx.remove_node(empty_subset)
    dfa.accepting_states = dfa.graph.terminal_recurrent_states()
    return dfa


def _mealy_support(hmm: HiddenMarkovModel) -> MealyHMM:
    return hmm.to_mealy()


def _support_graph(hmm: MealyHMM, *, edge_attr: str) -> TransitionGraph:
    graph = TransitionGraph()
    for state in hmm.states():
        graph.add_state(state)
    for transition in hmm.transitions():
        prob = float(transition.data.get(ATTR_PROB, 0.0))
        emission = transition.data.get(ATTR_EMISSION)
        if prob <= 0.0 or emission is None:
            continue
        graph.add_transition(transition.source, transition.target, **{edge_attr: emission})
    return graph


def _fresh_start_state(states: frozenset[Hashable]) -> Hashable:
    start: Hashable = ("__pensive_hmm_start__",)
    suffix = 0
    while start in states:
        suffix += 1
        start = ("__pensive_hmm_start__", suffix)
    return start


def quasi_realization_from_nmachine(nm: NMachine) -> QuasiRealization:
    idx = nm.reindex()
    pi = np.array([nm.initial_quasidistribution.get(s, 0.0) for s in idx.states], dtype=float)
    tau = np.ones(len(idx), dtype=float)
    symbol_maps: dict[Any, np.ndarray] = {}
    for transition in nm.transitions():
        emission = transition.data.get(ATTR_EMISSION)
        if emission is None:
            continue
        matrix = symbol_maps.setdefault(emission, np.zeros((len(idx), len(idx)), dtype=float))
        i = idx.index(transition.source)
        j = idx.index(transition.target)
        matrix[i, j] += float(transition.data.get(ATTR_QUASIPROB, 0.0))
    return QuasiRealization(pi=pi, tau=tau, symbol_maps=symbol_maps)


def nmachine_from_quasi_realization(
    qr: QuasiRealization, observation_alphabet: frozenset[Any] | None = None
) -> NMachine:
    states = sequential_labels(len(qr.pi))
    graph = TransitionGraph()
    for state in states:
        graph.add_state(state)
    alphabet = observation_alphabet if observation_alphabet is not None else frozenset(qr.symbol_maps)
    for symbol, matrix in qr.symbol_maps.items():
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                weight = float(matrix[i, j])
                if weight != 0.0:
                    graph.add_transition(states[i], states[j], **{ATTR_QUASIPROB: weight, ATTR_EMISSION: symbol})
    initial = {states[i]: float(qr.pi[i]) for i in range(len(qr.pi))}
    return NMachine(
        graph=graph,
        initial_quasidistribution=initial,
        observation_alphabet=alphabet,
    )


def hmm_to_edge_machine(hmm: MealyHMM | MooreHMM, iterations: int = 1, style: int = 0) -> MealyHMM:
    """Convert an HMM to its edge (generator) presentation."""
    from pensive.generators.edge_machine import hmm_to_edge_machine as _build

    return _build(hmm, iterations=iterations, style=style)


def epsilon_machine_to_unifilar_graph(eps: MealyHMM) -> TopologicalUnifilarGraph:
    """Strip emission-labeled transitions to a topological unifilar graph."""
    from pensive.generators.synchronization import graph_from_epsilon_machine

    return graph_from_epsilon_machine(eps)
