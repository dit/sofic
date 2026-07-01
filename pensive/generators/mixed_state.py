"""Mixed-state presentations of hidden Markov models."""

from __future__ import annotations

from collections.abc import Hashable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from pensive.exceptions import StochasticValidationError
from pensive.generators.mealy import MealyHMM
from pensive.graph import TransitionGraph


@dataclass(frozen=True, slots=True)
class MixedState:
    """A normalized belief distribution over presentation basis states."""

    belief: tuple[float, ...]

    @classmethod
    def from_vector(
        cls,
        vector: Sequence[float] | np.ndarray,
        *,
        decimals: int = 12,
        atol: float = 1e-12,
    ) -> MixedState | None:
        """Return a canonical mixed state, or ``None`` if the vector has no mass."""
        array = np.asarray(vector, dtype=float)
        total = float(array.sum())
        if total <= atol:
            return None
        normalized = array / total
        rounded = tuple(round(float(value), decimals) for value in normalized)
        total_rounded = sum(rounded)
        if total_rounded <= atol:
            return None
        if not np.isclose(total_rounded, 1.0, atol=10 ** (-decimals + 2)):
            rounded = tuple(round(value / total_rounded, decimals) for value in rounded)
        return cls(rounded)

    def as_array(self) -> np.ndarray:
        return np.asarray(self.belief, dtype=float)


def is_pure_mixed_state(
    state: MixedState | Sequence[float],
    *,
    atol: float = 1e-9,
) -> bool:
    """Return whether ``state`` is a vertex of the belief simplex."""
    belief = state.belief if isinstance(state, MixedState) else tuple(state)
    positives = [value for value in belief if value > atol]
    return len(positives) == 1 and np.isclose(sum(belief), 1.0, atol=atol)


def pure_state_index(state: MixedState, *, atol: float = 1e-9) -> int | None:
    """Return the basis index for a pure mixed state, else ``None``."""
    if not is_pure_mixed_state(state, atol=atol):
        return None
    for index, value in enumerate(state.belief):
        if value > atol:
            return index
    return None


def mixed_state_entropy(state: MixedState, *, atol: float = 1e-12) -> float:
    """Shannon entropy of a mixed state in bits."""
    from pensive.generators.stochastic import shannon_entropy

    return shannon_entropy(state.as_array(), atol=atol)


class MixedStatePresentation(MealyHMM):
    """Unifilar presentation whose states are beliefs over a source HMM."""

    basis_states: tuple[Hashable, ...]
    initial_mixed_state: MixedState
    pure_states: frozenset[MixedState]
    recurrent_states: frozenset[MixedState]
    transient_states: frozenset[MixedState]

    def __init__(
        self,
        *,
        basis_states: Sequence[Hashable],
        initial_mixed_state: MixedState,
        pure_states: frozenset[MixedState],
        recurrent_states: frozenset[MixedState],
        transient_states: frozenset[MixedState],
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.basis_states = tuple(basis_states)
        self.initial_mixed_state = initial_mixed_state
        self.pure_states = pure_states
        self.recurrent_states = recurrent_states
        self.transient_states = transient_states

    @classmethod
    def from_presentation(
        cls,
        hmm: MealyHMM,
        *,
        initial_mixed_state: MixedState | Mapping[Hashable, float] | Sequence[float] | None = None,
    ) -> MixedStatePresentation:
        from pensive.generators.mixed_state_construction import build_mixed_state_presentation

        return build_mixed_state_presentation(hmm, initial_mixed_state=initial_mixed_state)

    def to_recurrent(self) -> MealyHMM:
        """Return the recurrent component with stationary initial weights.

        Pure recurrent mixed states are relabeled by their basis states and returned
        as an :class:`~pensive.generators.epsilon_machine.EpsilonMachine`.  Otherwise
        the recurrent component remains a unifilar :class:`~pensive.generators.mealy.MealyHMM`
        over mixed states.
        """
        keep = frozenset(self.recurrent_states)
        if not keep:
            raise StochasticValidationError("mixed-state presentation has no recurrent states")

        state_map, is_epsilon_machine = self._recurrent_state_map(keep)
        graph = TransitionGraph()
        for state in keep:
            graph.add_state(state_map[state], **self.graph.state_attrs(state))
        for state in keep:
            for transition in self.graph.out_transitions(state):
                if transition.target in keep:
                    graph.add_transition(
                        state_map[state],
                        state_map[transition.target],
                        **transition.data,
                    )

        initial_distribution = self._recurrent_initial_distribution(keep, state_map)
        if is_epsilon_machine:
            from pensive.generators.epsilon_machine import EpsilonMachine

            recurrent = EpsilonMachine(
                graph=graph,
                initial_distribution=initial_distribution,
                observation_alphabet=self.observation_alphabet,
            )
            recurrent.validate()
            return recurrent

        recurrent = MealyHMM(
            graph=graph,
            initial_distribution=initial_distribution,
            observation_alphabet=self.observation_alphabet,
        )
        recurrent.validate_stochastic()
        recurrent._check_unifilar()
        return recurrent

    def _recurrent_state_map(
        self,
        keep: frozenset[MixedState],
    ) -> tuple[dict[MixedState, Hashable], bool]:
        labels: dict[MixedState, Hashable] = {}
        for state in keep:
            label = self.causal_state(state)
            if label is None:
                return {mixed_state: mixed_state for mixed_state in keep}, False
            labels[state] = label
        return labels, True

    def _recurrent_initial_distribution(
        self,
        keep: frozenset[MixedState],
        state_map: Mapping[MixedState, Hashable],
    ) -> dict[Hashable, float]:
        idx = self.reindex()
        stationary = self.stationary_distribution()
        initial: dict[Hashable, float] = {}
        for state in keep:
            mass = float(stationary[idx.index(state)])
            if mass > 0.0:
                label = state_map[state]
                initial[label] = initial.get(label, 0.0) + mass
        if not initial:
            raise StochasticValidationError("recurrent component has no positive stationary mass")
        total = sum(initial.values())
        return {state: mass / total for state, mass in initial.items()}

    def belief(self, state: MixedState) -> tuple[float, ...]:
        return state.belief

    def causal_state(self, state: MixedState) -> Hashable | None:
        index = pure_state_index(state)
        if index is None:
            return None
        return self.basis_states[index]

    def is_pure(self, state: MixedState) -> bool:
        return state in self.pure_states

    def is_recurrent(self, state: MixedState) -> bool:
        return state in self.recurrent_states

    def is_transient(self, state: MixedState) -> bool:
        return state in self.transient_states
