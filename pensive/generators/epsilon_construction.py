"""Epsilon machine construction via causal-state merging.

For a unifilar finite predictive model, merge probabilistically equivalent
states by Hopcroft-style partition refinement on ``(symbol, probability,
successor_block)`` labels (Loomis & Crutchfield, arXiv:1808.08639, Cor. 1).
Non-unifilar inputs are first converted to a unifilar mixed-state presentation
(Ellison, Mahoney & Crutchfield, J. Stat. Phys. 2009).
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np

from pensive.exceptions import StochasticValidationError
from pensive.generators.base import HiddenMarkovModel
from pensive.generators.epsilon_machine import EpsilonMachine
from pensive.generators.mealy import MealyHMM
from pensive.generators.mixed_state import MixedStatePresentation
from pensive.graph import ATTR_EMISSION, ATTR_PROB, TransitionGraph
from pensive.states import sequential_labels

TransitionSignature = tuple[tuple[Any, int, float], ...]


def build_epsilon_machine(hmm: HiddenMarkovModel) -> EpsilonMachine:
    hmm = hmm.to_mealy()
    presentation = _unifilar_presentation(hmm)
    stationary = presentation.stationary_distribution()
    if stationary.sum() <= 0.0:
        raise StochasticValidationError("generator must have a stationary distribution")

    partitions = _refine_probabilistic_partitions(presentation)
    return _quotient_machine(presentation, partitions, stationary)


def _unifilar_presentation(hmm: MealyHMM) -> MealyHMM:
    """Return a row-unifilar presentation generating the same process."""
    if not hmm.is_unifilar():
        hmm = hmm.mixed_state_presentation()
    if isinstance(hmm, MixedStatePresentation) and hmm.recurrent_states:
        return _restrict_presentation(hmm, set(hmm.recurrent_states))
    return hmm


def _restrict_presentation(hmm: MealyHMM, keep: set[Any]) -> MealyHMM:
    """Return the subpresentation induced by ``keep`` with stationary initial weights."""
    graph = TransitionGraph()
    for state in keep:
        graph.add_state(state)
    for state in keep:
        for transition in hmm.graph.out_transitions(state):
            if transition.target in keep:
                graph.add_transition(state, transition.target, **transition.data)

    idx = hmm.reindex()
    pi = hmm.stationary_distribution()
    initial: dict[Any, float] = {}
    for state in keep:
        mass = float(pi[idx.index(state)])
        if mass > 0.0:
            initial[state] = mass
    if not initial:
        raise StochasticValidationError("restricted presentation has no positive stationary mass")
    total = sum(initial.values())
    initial = {state: mass / total for state, mass in initial.items()}

    restricted = MealyHMM(
        graph=graph,
        initial_distribution=initial,
        observation_alphabet=hmm.observation_alphabet,
    )
    restricted.validate_stochastic()
    return restricted


def _refine_probabilistic_partitions(hmm: MealyHMM) -> list[set[Any]]:
    """Hopcroft-style refinement on probabilistic transition signatures."""
    partitions: list[set[Any]] = [set(hmm.states())]
    changed = True
    while changed:
        changed = False
        state_to_block = _state_to_block_index(partitions)
        new_partitions: list[set[Any]] = []
        for block in partitions:
            subblocks = _split_block(hmm, block, state_to_block)
            if len(subblocks) > 1:
                changed = True
            new_partitions.extend(subblocks)
        partitions = new_partitions
    return partitions


def _state_to_block_index(partitions: list[set[Any]]) -> dict[Any, int]:
    mapping: dict[Any, int] = {}
    for index, block in enumerate(partitions):
        for state in block:
            mapping[state] = index
    return mapping


def _split_block(
    hmm: MealyHMM,
    block: set[Any],
    state_to_block: dict[Any, int],
) -> list[set[Any]]:
    """Split a block when states disagree on labeled successor blocks."""
    futures: dict[TransitionSignature, set[Any]] = defaultdict(set)
    for state in block:
        futures[_transition_signature(hmm, state, state_to_block)].add(state)
    return list(futures.values())


def _transition_signature(
    hmm: MealyHMM,
    state: Any,
    state_to_block: dict[Any, int],
) -> TransitionSignature:
    triples: list[tuple[Any, int, float]] = []
    for transition in hmm.graph.out_transitions(state):
        emission = transition.data.get(ATTR_EMISSION)
        if emission is None:
            continue
        triples.append(
            (
                emission,
                state_to_block[transition.target],
                float(transition.data.get(ATTR_PROB, 0.0)),
            )
        )
    return tuple(sorted(triples, key=repr))


def _quotient_machine(
    hmm: MealyHMM,
    partitions: list[set[Any]],
    stationary: np.ndarray,
) -> EpsilonMachine:
    state_map: dict[Any, int] = {}
    for index, block in enumerate(partitions):
        for state in block:
            state_map[state] = index

    graph = TransitionGraph()
    labels = sequential_labels(len(partitions))
    for label in labels:
        graph.add_state(label)
    label_for_index = {index: labels[index] for index in range(len(partitions))}

    for block_index, block in enumerate(partitions):
        representative = next(iter(block))
        source = label_for_index[block_index]
        for transition in hmm.graph.out_transitions(representative):
            emission = transition.data.get(ATTR_EMISSION)
            if emission is None:
                continue
            target = label_for_index[state_map[transition.target]]
            prob = float(transition.data.get(ATTR_PROB, 0.0))
            graph.add_transition(
                source,
                target,
                **{ATTR_PROB: prob, ATTR_EMISSION: emission},
            )

    initial = np.zeros(len(partitions), dtype=float)
    for state, mass in hmm.initial_distribution.items():
        initial[state_map[state]] += float(mass)
    if initial.sum() <= 0.0:
        idx = hmm.reindex()
        for state, mass in zip(idx.states, stationary, strict=False):
            initial[state_map[state]] += float(mass)
    initial /= initial.sum()

    eps = EpsilonMachine(
        graph=graph,
        initial_distribution={
            label_for_index[i]: float(initial[i]) for i in range(len(partitions)) if initial[i] > 0.0
        },
        observation_alphabet=hmm.observation_alphabet,
    )
    eps.validate()
    return eps
