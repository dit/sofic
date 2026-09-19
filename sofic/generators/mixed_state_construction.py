"""Build mixed-state presentations via belief-state enumeration."""

from __future__ import annotations

from collections import deque
from collections.abc import Hashable, Mapping, Sequence
from typing import Any

import numpy as np

from sofic.exceptions import MixedStateExplosionError
from sofic.generators.base import HiddenMarkovModel
from sofic.generators.mealy import MealyHMM
from sofic.generators.mixed_state import (
    MixedState,
    MixedStatePresentation,
    is_pure_mixed_state,
)
from sofic.generators.prob import (
    as_prob,
    has_symbolic,
    is_positive_mass,
    matvec,
    simplify_prob,
    sum_probs,
    zeros,
)
from sofic.graph import ATTR_EMISSION, ATTR_PROB, TransitionGraph


def _terminal_recurrent_states(graph: TransitionGraph) -> frozenset[Any]:
    return graph.terminal_recurrent_states()


def _resolve_initial_belief(
    hmm: HiddenMarkovModel,
    basis: tuple[Hashable, ...],
    initial_mixed_state: MixedState | Mapping[Hashable, Any] | Sequence[Any] | None,
) -> MixedState:
    if initial_mixed_state is None:
        vector = hmm.stationary_distribution()
    elif isinstance(initial_mixed_state, MixedState):
        vector = initial_mixed_state.as_array()
    elif isinstance(initial_mixed_state, Mapping):
        index = {state: i for i, state in enumerate(basis)}
        symbolic = has_symbolic(initial_mixed_state.values())
        vector = zeros((len(basis),), symbolic=symbolic)
        for state, mass in initial_mixed_state.items():
            vector[index[state]] = as_prob(mass)
    else:
        vector = np.asarray(initial_mixed_state, dtype=object if has_symbolic(initial_mixed_state) else float)

    if len(np.asarray(vector).ravel()) != len(basis):
        raise ValueError(f"initial belief has length {len(np.asarray(vector).ravel())}, expected {len(basis)}")
    mixed = MixedState.from_vector(vector)
    if mixed is None:
        raise ValueError("initial mixed state has zero total mass")
    return mixed


def build_mixed_state_presentation(
    hmm: HiddenMarkovModel,
    *,
    initial_mixed_state: MixedState | Mapping[Hashable, Any] | Sequence[Any] | None = None,
    max_states: int = 10_000,
) -> MixedStatePresentation:
    """Enumerate the mixed-state presentation of ``hmm``.

    States are reachable belief distributions over the source presentation's hidden
    states, updated by Bayes' rule on emitted symbols.  The construction follows
    Ellison, Mahoney & Crutchfield (J. Stat. Phys. 2009), Sec. VIII.2.

    Any Mealy-style HMM (joint edge emissions) is accepted; unifilarity is not required.
    Probabilities may be floats or exact sympy expressions.

    Parameters
    ----------
    max_states
        Safety cap on enumerated beliefs (symbolic machines can otherwise grow
        without bound when successor beliefs fail to identify).
    """
    if isinstance(hmm, MixedStatePresentation):
        raise TypeError("cannot build a mixed-state presentation from another mixed-state presentation")
    if not isinstance(hmm, MealyHMM):
        raise TypeError(f"mixed-state presentation requires a MealyHMM, not {type(hmm)!r}")

    from sofic.generators.hmm_inference import _emission_transition_tensors

    constraints = getattr(hmm, "symbol_constraints", None)

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
        for known in discovered:
            if _beliefs_equal(known, state, constraints=constraints):
                discovered[state] = known
                return known
        if len(discovered) >= max_states:
            raise MixedStateExplosionError(
                f"mixed-state presentation exceeded max_states={max_states}; the reachable belief set may be infinite"
            )
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
            mass = matvec(row, matrix)
            probability = simplify_prob(sum_probs(mass.tolist()))
            if not is_positive_mass(probability):
                continue
            successor = MixedState.from_vector(mass)
            if successor is None:
                continue
            successor = register(successor)
            graph.add_transition(
                eta,
                successor,
                **{ATTR_PROB: as_prob(probability), ATTR_EMISSION: symbol},
            )

    pure_states = frozenset(state for state in discovered.values() if is_pure_mixed_state(state))
    unique_states = frozenset(discovered.values())
    recurrent_states = _terminal_recurrent_states(graph) & unique_states
    if not recurrent_states:
        recurrent_states = _terminal_recurrent_states(graph)
    reachable = frozenset(discovered.values())
    transient_states = reachable - recurrent_states

    return MixedStatePresentation(
        graph=graph,
        basis_states=basis,
        initial_mixed_state=eta0,
        pure_states=pure_states,
        recurrent_states=frozenset(recurrent_states),
        transient_states=frozenset(transient_states),
        initial_distribution={eta0: as_prob(1)},
        observation_alphabet=hmm.observation_alphabet,
        symbol_constraints=constraints,
    )


def _beliefs_equal(left: MixedState, right: MixedState, *, constraints: Any = None) -> bool:
    from sofic.generators.prob import probs_equal

    if len(left.belief) != len(right.belief):
        return False
    return all(probs_equal(a, b, constraints=constraints) for a, b in zip(left.belief, right.belief, strict=True))
