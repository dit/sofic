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

from sofic.exceptions import StochasticValidationError
from sofic.generators.base import HiddenMarkovModel
from sofic.generators.epsilon_machine import EpsilonMachine
from sofic.generators.mealy import MealyHMM
from sofic.generators.mixed_state import MixedStatePresentation
from sofic.generators.prob import (
    array_sum,
    as_prob,
    canonical_prob_key,
    has_symbolic,
    is_positive_mass,
    is_zero,
    simplify_prob,
    sum_probs,
)
from sofic.graph import ATTR_EMISSION, ATTR_PROB, TransitionGraph
from sofic.states import sequential_labels

TransitionSignature = tuple[tuple[Any, int, Any], ...]


def build_epsilon_machine(hmm: HiddenMarkovModel) -> EpsilonMachine:
    hmm = hmm.to_mealy()
    presentation = _unifilar_presentation(hmm)
    stationary = presentation.stationary_distribution()
    if is_zero(array_sum(stationary)):
        raise StochasticValidationError("generator must have a stationary distribution")

    constraints = getattr(presentation, "symbol_constraints", None) or getattr(
        hmm, "symbol_constraints", None
    )
    partitions = _refine_probabilistic_partitions(presentation, constraints=constraints)
    return _quotient_machine(presentation, partitions, stationary, constraints=constraints)


def _unifilar_presentation(hmm: MealyHMM) -> MealyHMM:
    """Return a row-unifilar presentation generating the same process."""
    if not hmm.is_unifilar():
        hmm = hmm.mixed_state_presentation()
    if isinstance(hmm, MixedStatePresentation) and hmm.recurrent_states:
        return hmm.to_recurrent()
    return hmm


def _refine_probabilistic_partitions(
    hmm: MealyHMM,
    *,
    constraints: Any = None,
) -> list[set[Any]]:
    """Hopcroft-style refinement on probabilistic transition signatures."""
    partitions: list[set[Any]] = [set(hmm.states())]
    changed = True
    while changed:
        changed = False
        state_to_block = _state_to_block_index(partitions)
        new_partitions: list[set[Any]] = []
        for block in partitions:
            subblocks = _split_block(hmm, block, state_to_block, constraints=constraints)
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
    *,
    constraints: Any = None,
) -> list[set[Any]]:
    """Split a block when states disagree on labeled successor blocks."""
    futures: dict[TransitionSignature, set[Any]] = defaultdict(set)
    for state in block:
        futures[_transition_signature(hmm, state, state_to_block, constraints=constraints)].add(state)
    return list(futures.values())


def _transition_signature(
    hmm: MealyHMM,
    state: Any,
    state_to_block: dict[Any, int],
    *,
    constraints: Any = None,
) -> TransitionSignature:
    triples: list[tuple[Any, int, Any]] = []
    for transition in hmm.graph.out_transitions(state):
        emission = transition.data.get(ATTR_EMISSION)
        if emission is None:
            continue
        prob = as_prob(transition.data.get(ATTR_PROB, 0.0))
        triples.append(
            (
                emission,
                state_to_block[transition.target],
                canonical_prob_key(prob, constraints),
            )
        )
    return tuple(sorted(triples, key=repr))


def _quotient_machine(
    hmm: MealyHMM,
    partitions: list[set[Any]],
    stationary: np.ndarray,
    *,
    constraints: Any = None,
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
            prob = as_prob(transition.data.get(ATTR_PROB, 0.0))
            graph.add_transition(
                source,
                target,
                **{ATTR_PROB: prob, ATTR_EMISSION: emission},
            )

    symbolic = stationary.dtype == object or has_symbolic(stationary.ravel())
    if symbolic:
        import sympy as sp

        initial = [sp.Integer(0)] * len(partitions)
        for state, mass in hmm.initial_distribution.items():
            initial[state_map[state]] = simplify_prob(as_prob(initial[state_map[state]]) + as_prob(mass))
        if is_zero(sum_probs(initial)):
            idx = hmm.reindex()
            for state, mass in zip(idx.states, stationary, strict=False):
                initial[state_map[state]] = simplify_prob(
                    as_prob(initial[state_map[state]]) + as_prob(mass)
                )
        total = sum_probs(initial)
        initial_dist = {
            label_for_index[i]: simplify_prob(as_prob(initial[i]) / total)
            for i in range(len(partitions))
            if is_positive_mass(initial[i])
        }
    else:
        initial = np.zeros(len(partitions), dtype=float)
        for state, mass in hmm.initial_distribution.items():
            initial[state_map[state]] += float(mass)
        if initial.sum() <= 0.0:
            idx = hmm.reindex()
            for state, mass in zip(idx.states, stationary, strict=False):
                initial[state_map[state]] += float(mass)
        initial /= initial.sum()
        initial_dist = {
            label_for_index[i]: float(initial[i]) for i in range(len(partitions)) if initial[i] > 0.0
        }

    eps = EpsilonMachine(
        graph=graph,
        initial_distribution=initial_dist,
        observation_alphabet=hmm.observation_alphabet,
        symbol_constraints=constraints,
    )
    eps.validate()
    return eps
