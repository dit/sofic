"""N-machine state-splitting construction."""

from __future__ import annotations

from collections.abc import Hashable, Mapping

from pensive.generators.epsilon_machine import EpsilonMachine
from pensive.generators.nmachine import NMachine
from pensive.graph import ATTR_EMISSION, ATTR_PROB, ATTR_QUASIPROB, TransitionGraph


def build_nmachine_from_epsilon(
    eps: EpsilonMachine,
    splits: Mapping[Hashable, int] | None = None,
) -> NMachine:
    split_counts = dict(splits or {})
    for state in eps.states():
        split_counts.setdefault(state, 1)

    graph = TransitionGraph()
    coarse_map: dict[tuple[Hashable, int], Hashable] = {}
    for state in eps.states():
        count = split_counts[state]
        for branch in range(count):
            substate = (state, branch)
            graph.add_state(substate)
            coarse_map[substate] = state

    pi = eps.stationary_distribution()
    idx = eps.reindex()
    initial: dict[tuple[Hashable, int], float] = {}
    for state in eps.states():
        count = split_counts[state]
        mass = float(pi[idx.index(state)]) / count
        for branch in range(count):
            initial[(state, branch)] = mass

    for state in eps.states():
        count = split_counts[state]
        outgoing = list(eps.graph.out_transitions(state))
        for branch in range(count):
            source = (state, branch)
            for transition in outgoing:
                emission = transition.data.get(ATTR_EMISSION)
                target_state = transition.target
                target_branch = branch % split_counts[target_state]
                prob = float(transition.data.get(ATTR_PROB, 0.0))
                graph.add_transition(
                    source,
                    (target_state, target_branch),
                    **{ATTR_QUASIPROB: prob, ATTR_EMISSION: emission},
                )

    return NMachine(
        graph=graph,
        initial_quasidistribution=initial,
        observation_alphabet=eps.observation_alphabet,
    )


def coarse_grained_distribution(nm: NMachine, eps_states: tuple[Hashable, ...]) -> dict[Hashable, float]:
    pi = nm.stationary_quasidistribution()
    idx = nm.reindex()
    coarse: dict[Hashable, float] = dict.fromkeys(eps_states, 0.0)
    for substate, mass in zip(idx.states, pi, strict=False):
        if isinstance(substate, tuple) and len(substate) == 2:
            coarse[substate[0]] = coarse.get(substate[0], 0.0) + float(mass)
        else:
            coarse[substate] = coarse.get(substate, 0.0) + float(mass)
    return coarse
