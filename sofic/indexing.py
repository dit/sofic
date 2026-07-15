"""Dense state indexing for NumPy-backed algorithms."""

from __future__ import annotations

from collections.abc import Hashable, Iterable


class StateIndex:
    """Bidirectional mapping between hashable states and ``0..n-1`` indices."""

    __slots__ = ("_states", "_index")

    def __init__(self, states: Iterable[Hashable]) -> None:
        self._states = tuple(dict.fromkeys(states))
        self._index = {state: i for i, state in enumerate(self._states)}

    def index(self, state: Hashable) -> int:
        return self._index[state]

    def state(self, i: int) -> Hashable:
        return self._states[i]

    @property
    def states(self) -> tuple[Hashable, ...]:
        return self._states

    def __len__(self) -> int:
        return len(self._states)

    def __contains__(self, state: Hashable) -> bool:
        return state in self._index
