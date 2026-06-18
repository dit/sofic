"""Generator presentation conversions."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from pensive.generators.synchronization import TopologicalUnifilarGraph

from pensive.generators.base import QuasiStochasticModel
from pensive.generators.mealy import MealyHMM
from pensive.generators.moore import MooreHMM
from pensive.generators.nmachine import NMachine
from pensive.generators.pfa import ProbabilisticFiniteAutomaton
from pensive.generators.quasi_realization import QuasiRealization
from pensive.graph import ATTR_EMISSION, ATTR_EMISSION_DIST, ATTR_PROB, ATTR_QUASIPROB, TransitionGraph
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


def pfa_to_mealy_hmm(pfa: ProbabilisticFiniteAutomaton) -> MealyHMM:
    graph = pfa.graph.copy()
    return MealyHMM(
        graph=graph,
        initial_distribution=pfa.initial_distribution,
        observation_alphabet=frozenset(pfa.output_alphabet),
    )


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


def nmachine_from_quasi_realization(qr: QuasiRealization, observation_alphabet: frozenset[Any] | None = None) -> NMachine:
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


def edge_machine_from_hmm(hmm: MealyHMM | MooreHMM) -> MealyHMM:
    """Convert an HMM to its edge (generator) presentation."""
    from pensive.generators.edge_machine import edge_machine_from_hmm as _build

    return _build(hmm)


def epsilon_machine_to_unifilar_graph(eps: MealyHMM) -> "TopologicalUnifilarGraph":
    """Strip emission-labeled transitions to a topological unifilar graph."""
    from pensive.generators.synchronization import graph_from_epsilon_machine

    return graph_from_epsilon_machine(eps)
