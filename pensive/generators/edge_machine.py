"""Edge-machine (generator) presentation of an HMM.

States index labeled transitions of the source HMM; the emitted symbol on a
step is carried by the destination edge state. See Travers & Crutchfield,
*Equivalence of History and Generator ε-Machines*.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Hashable
from typing import Any

from pensive.generators.base import HiddenMarkovModel
from pensive.generators.mealy import MealyHMM
from pensive.graph import ATTR_EMISSION, ATTR_PROB, TransitionGraph

type EdgeState = tuple[Hashable, Any, Hashable]

_ATTR_EDGE_SOURCE = "edge_source"
_ATTR_EDGE_TARGET = "edge_target"


def edge_state_label(edge: EdgeState) -> str:
    """Stable string label for an edge triple (dit-friendly state name)."""
    source, emission, target = edge
    return f"{source!r}\x1e{emission!r}\x1e{target!r}"


def parse_edge_state_label(label: str) -> EdgeState:
    """Recover ``(source, emission, target)`` from :func:`edge_state_label`."""
    parts = label.split("\x1e", 2)
    if len(parts) != 3:
        raise ValueError(f"not an edge state label: {label!r}")
    return (eval(parts[0]), eval(parts[1]), eval(parts[2]))


def hmm_to_edge_machine(hmm: HiddenMarkovModel) -> MealyHMM:
    """Build the edge machine whose states are the source HMM's transitions.

    Each edge state encodes ``(source, emission, target)``.  Leaving edge ``e₀``
    for edge ``e₁`` emits symbol ``e₁[1]`` with probability ``P(e₁ | e₀.target)``.
    """
    hmm = hmm.to_mealy()

    edge_probs: dict[EdgeState, float] = defaultdict(float)
    outgoing_by_source: dict[Hashable, list[EdgeState]] = defaultdict(list)

    for transition in hmm.transitions():
        emission = transition.data.get(ATTR_EMISSION)
        if emission is None:
            continue
        prob = float(transition.data.get(ATTR_PROB, 0.0))
        if prob <= 0.0:
            continue
        edge: EdgeState = (transition.source, emission, transition.target)
        edge_probs[edge] += prob
        outgoing_by_source[transition.source].append(edge)

    if not edge_probs:
        raise ValueError("HMM has no positive-probability labeled transitions")

    graph = TransitionGraph()
    label_for_edge: dict[EdgeState, str] = {}
    for edge in edge_probs:
        label = edge_state_label(edge)
        label_for_edge[edge] = label
        graph.add_state(
            label,
            **{_ATTR_EDGE_SOURCE: edge[0], ATTR_EMISSION: edge[1], _ATTR_EDGE_TARGET: edge[2]},
        )

    for _source_state, edges in outgoing_by_source.items():
        total = sum(edge_probs[edge] for edge in edges)
        if total <= 0.0:
            continue
        for edge_from in edges:
            target_state = edge_from[2]
            next_edges = outgoing_by_source.get(target_state, [])
            next_total = sum(edge_probs[edge] for edge in next_edges)
            if next_total <= 0.0:
                continue
            for edge_to in next_edges:
                graph.add_transition(
                    label_for_edge[edge_from],
                    label_for_edge[edge_to],
                    **{
                        ATTR_PROB: edge_probs[edge_to] / next_total,
                        ATTR_EMISSION: edge_to[1],
                    },
                )

    idx = hmm.reindex()
    pi = hmm.stationary_distribution()
    initial: dict[str, float] = defaultdict(float)
    for edge, prob in edge_probs.items():
        source_index = idx.index(edge[0])
        initial[label_for_edge[edge]] += float(pi[source_index] * prob)
    total = sum(initial.values())
    if total <= 0.0:
        raise ValueError("edge machine initial distribution is empty")
    initial = {label: mass / total for label, mass in initial.items() if mass > 0.0}

    return MealyHMM(
        graph=graph,
        initial_distribution=dict(initial),
        observation_alphabet=hmm.observation_alphabet,
    )
