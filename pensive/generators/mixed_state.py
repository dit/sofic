"""Mixed-state presentations of hidden Markov models."""

from __future__ import annotations

from collections.abc import Hashable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from pensive.generators.mealy import MealyHMM


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
    belief = state.as_array()
    mask = belief > atol
    if not np.any(mask):
        return 0.0
    positive = belief[mask]
    return float(-np.sum(positive * np.log2(positive)))


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
