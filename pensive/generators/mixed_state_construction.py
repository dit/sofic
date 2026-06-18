"""Build mixed-state presentations via belief-state enumeration."""

from __future__ import annotations

from collections import deque
from collections.abc import Hashable, Mapping, Sequence
from typing import Any

import numpy as np

from pensive.generators.base import HiddenMarkovModel
from pensive.generators.mealy import MealyHMM
from pensive.generators.mixed_state import (
    MixedState,
    MixedStatePresentation,
    is_pure_mixed_state,
)
from pensive.graph import ATTR_EMISSION, ATTR_PROB, TransitionGraph


def _terminal_recurrent_states(graph: TransitionGraph) -> frozenset[Any]:
    return graph.terminal_recurrent_states()


def _resolve_initial_belief(
    hmm: HiddenMarkovModel,
    basis: tuple[Hashable, ...],
    initial_mixed_state: MixedState | Mapping[Hashable, float] | Sequence[float] | None,
) -> MixedState:
    if initial_mixed_state is None:
        pi = hmm.stationary_distribution()
        vector = pi
    elif isinstance(initial_mixed_state, MixedState):
        vector = initial_mixed_state.as_array()
    elif isinstance(initial_mixed_state, Mapping):
        index = {state: i for i, state in enumerate(basis)}
        vector = np.zeros(len(basis), dtype=float)
        for state, mass in initial_mixed_state.items():
            vector[index[state]] = float(mass)
    else:
        vector = np.asarray(initial_mixed_state, dtype=float)

    if vector.shape != (len(basis),):
        raise ValueError(f"initial belief has length {vector.shape[0]}, expected {len(basis)}")
    mixed = MixedState.from_vector(vector)
    if mixed is None:
        raise ValueError("initial mixed state has zero total mass")
    return mixed


def build_mixed_state_presentation(
    hmm: HiddenMarkovModel,
    *,
    initial_mixed_state: MixedState | Mapping[Hashable, float] | Sequence[float] | None = None,
) -> MixedStatePresentation:
    """Enumerate the mixed-state presentation of ``hmm``.

    States are reachable belief distributions over the source presentation's hidden
    states, updated by Bayes' rule on emitted symbols.  The construction follows
    Ellison, Mahoney & Crutchfield (J. Stat. Phys. 2009), Sec. VIII.2.

    Any Mealy-style HMM (joint edge emissions) is accepted; unifilarity is not required.
    """
    if isinstance(hmm, MixedStatePresentation):
        raise TypeError("cannot build a mixed-state presentation from another mixed-state presentation")
    if not isinstance(hmm, MealyHMM):
        raise TypeError(f"mixed-state presentation requires a MealyHMM, not {type(hmm)!r}")

    from pensive.generators.hmm_inference import _emission_transition_tensors

    idx = hmm.reindex()
    basis = idx.states
    _, joint = _emission_transition_tensors(hmm)
    symbols = tuple(sorted(joint, key=str))
    eta0 = _resolve_initial_belief(hmm, basis, initial_mixed_state)

    graph = TransitionGraph()
    discovered: dict[MixedState, MixedState] = {}
    queue: deque[MixedState] = deque()

    def register(state: MixedState) -> MixedState:
        existing = discovered.get(state)
        if existing is not None:
            return existing
        discovered[state] = state
        graph.add_state(state)
        queue.append(state)
        return state

    register(eta0)

    while queue:
        eta = queue.popleft()
        row = eta.as_array()
        for symbol in symbols:
            matrix = joint[symbol]
            mass = row @ matrix
            probability = float(mass.sum())
            if probability <= 0.0:
                continue
            successor = MixedState.from_vector(mass)
            if successor is None:
                continue
            successor = register(successor)
            graph.add_transition(
                eta,
                successor,
                **{ATTR_PROB: probability, ATTR_EMISSION: symbol},
            )

    pure_states = frozenset(state for state in discovered if is_pure_mixed_state(state))
    recurrent_states = _terminal_recurrent_states(graph)
    reachable = frozenset(discovered)
    transient_states = reachable - recurrent_states

    return MixedStatePresentation(
        graph=graph,
        basis_states=basis,
        initial_mixed_state=eta0,
        pure_states=pure_states,
        recurrent_states=recurrent_states,
        transient_states=transient_states,
        initial_distribution={eta0: 1.0},
        observation_alphabet=hmm.observation_alphabet,
    )
