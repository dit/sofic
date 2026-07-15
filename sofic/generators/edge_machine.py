"""Edge-machine (generator) presentation of an HMM.

States index labeled transition paths of the source HMM. By default, the
emitted symbol on a step is pulled from the destination edge state, matching
cmpy's edge-machine construction.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from sofic.generators.base import HiddenMarkovModel
from sofic.generators.mealy import MealyHMM
from sofic.graph import ATTR_EMISSION, ATTR_PROB, TransitionGraph

EdgeState = tuple[Any, ...]

ATTR_EDGE_SOURCE = "edge_source"
ATTR_EDGE_TARGET = "edge_target"
_ATTR_EDGE_SOURCE = ATTR_EDGE_SOURCE
_ATTR_EDGE_TARGET = ATTR_EDGE_TARGET


def edge_state_label(edge: EdgeState) -> str:
    """Stable string label for an edge-state tuple (dit-friendly state name)."""
    return "\x1e".join(repr(part) for part in edge)


def parse_edge_state_label(label: str) -> EdgeState:
    """Recover an edge-state tuple from :func:`edge_state_label`."""
    parts = label.split("\x1e")
    if not label or not parts:
        raise ValueError(f"not an edge state label: {label!r}")
    return tuple(eval(part) for part in parts)


def hmm_to_edge_machine(hmm: HiddenMarkovModel, iterations: int = 1, style: int = 0) -> MealyHMM:
    """Build the edge machine whose states are source-HMM transition paths.

    With ``iterations=1``, each state is ``(source, emission, target)``. More
    iterations extend states by sliding-path suffixes, as in cmpy. ``style=0``
    uses the default PULL labels from destination states; ``style=1`` rewrites
    final transition emissions to PUSH labels from source states.
    """
    hmm = hmm.to_mealy()
    iterations = int(iterations)
    if iterations == 0:
        return hmm
    if iterations < 0:
        raise ValueError("iterations must be nonnegative")

    edge_machine = _edge_machine_iteration(hmm, edge_machine=False)
    for _iteration in range(iterations - 1):
        edge_machine = _edge_machine_iteration(edge_machine, edge_machine=True)

    if style == 1:
        _push_transition_labels(edge_machine)
    return edge_machine


def _edge_machine_iteration(machine: MealyHMM, *, edge_machine: bool) -> MealyHMM:
    edge_states, edge_probs, states_by_source = _edge_states(machine, edge_machine=edge_machine)
    if not edge_states:
        raise ValueError("HMM has no labeled transitions")

    graph = _edge_transition_graph(edge_states, edge_probs, states_by_source, edge_machine=edge_machine)
    initial = _edge_initial_distribution(machine, edge_states, edge_probs, edge_machine=edge_machine)
    return MealyHMM(
        graph=graph,
        initial_distribution=initial,
        observation_alphabet=machine.observation_alphabet,
    )


def _edge_states(
    machine: MealyHMM,
    *,
    edge_machine: bool,
) -> tuple[list[EdgeState], dict[EdgeState, float], dict[Any, list[EdgeState]]]:
    edge_states: list[EdgeState] = []
    edge_probs: dict[EdgeState, float] = defaultdict(float)
    states_by_source: dict[Any, list[EdgeState]] = defaultdict(list)
    seen: set[EdgeState] = set()

    for transition in machine.transitions():
        emission = transition.data.get(ATTR_EMISSION)
        if emission is None:
            continue
        state = _state_for_transition(
            transition.source,
            transition.target,
            emission,
            edge_machine=edge_machine,
        )
        if state not in seen:
            seen.add(state)
            edge_states.append(state)
            states_by_source[_state_source(state, edge_machine=edge_machine)].append(state)
        edge_probs[state] += float(transition.data.get(ATTR_PROB, 0.0))

    return edge_states, dict(edge_probs), dict(states_by_source)


def _state_for_transition(source: Any, target: Any, emission: Any, *, edge_machine: bool) -> EdgeState:
    if edge_machine:
        if not isinstance(source, tuple) or not isinstance(target, tuple):
            raise TypeError("iterated edge-machine states must be tuples")
        return source + target[-2:]
    return (source, emission, target)


def _state_source(state: EdgeState, *, edge_machine: bool) -> Any:
    return state[:-2] if edge_machine else state[0]


def _state_target(state: EdgeState, *, edge_machine: bool) -> Any:
    return state[2:] if edge_machine else state[2]


def _edge_transition_graph(
    edge_states: list[EdgeState],
    edge_probs: dict[EdgeState, float],
    states_by_source: dict[Any, list[EdgeState]],
    *,
    edge_machine: bool,
) -> TransitionGraph:
    graph = TransitionGraph()
    for edge in edge_states:
        graph.add_state(
            edge,
            **_edge_state_attrs(edge),
        )

    for edge_from in edge_states:
        target_state = _state_target(edge_from, edge_machine=edge_machine)
        for edge_to in states_by_source.get(target_state, []):
            graph.add_transition(
                edge_from,
                edge_to,
                **{
                    ATTR_PROB: edge_probs[edge_to],
                    ATTR_EMISSION: edge_to[-2],
                },
            )
    return graph


def _edge_state_attrs(edge: EdgeState) -> dict[str, Any]:
    if len(edge) != 3:
        return {}
    return {_ATTR_EDGE_SOURCE: edge[0], ATTR_EMISSION: edge[1], _ATTR_EDGE_TARGET: edge[2]}


def _edge_initial_distribution(
    machine: MealyHMM,
    edge_states: list[EdgeState],
    edge_probs: dict[EdgeState, float],
    *,
    edge_machine: bool,
) -> dict[EdgeState, float]:
    idx = machine.reindex()
    pi = machine.stationary_distribution()
    initial: dict[EdgeState, float] = defaultdict(float)
    for edge in edge_states:
        source = _state_source(edge, edge_machine=edge_machine)
        if source not in idx:
            continue
        initial[edge] += float(pi[idx.index(source)] * edge_probs[edge])
    total = sum(initial.values())
    if total <= 0.0:
        raise ValueError("edge machine initial distribution is empty")
    return {edge: mass / total for edge, mass in initial.items() if mass > 0.0}


def _push_transition_labels(machine: MealyHMM) -> None:
    for source, _target, _key, data in machine.graph.nx.edges(keys=True, data=True):
        data[ATTR_EMISSION] = source[-2]
