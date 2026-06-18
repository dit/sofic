"""Finite-state transducers."""

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Hashable, Sequence
from typing import Any

from pensive.base import StateMachine
from pensive.graph import ATTR_OUTPUT, ATTR_SYMBOL


class Transducer(StateMachine):
    """Non-probabilistic input-to-output machine."""

    input_alphabet: frozenset[Any]
    output_alphabet: frozenset[Any]
    initial_states: frozenset[Hashable]

    def __init__(
        self,
        input_alphabet: frozenset[Any] | None = None,
        output_alphabet: frozenset[Any] | None = None,
        initial_states: frozenset[Hashable] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.input_alphabet = input_alphabet if input_alphabet is not None else frozenset()
        self.output_alphabet = output_alphabet if output_alphabet is not None else frozenset()
        self.initial_states = initial_states if initial_states is not None else frozenset()

    def validate(self) -> None:
        for state in self.initial_states:
            self._require(self.graph.has_state(state), f"missing initial state {state!r}")

    def is_deterministic(self) -> bool:
        """Return whether each state has at most one transition per input symbol."""
        from pensive.properties import is_deterministic_transducer

        return is_deterministic_transducer(self)

    @abstractmethod
    def transduce(self, word: Sequence[Any]) -> set[tuple[Any, ...]]:
        """Return possible output sequences for ``word``."""


class MealyMachine(Transducer):
    """Output symbols on transitions."""

    def transduce(self, word: Sequence[Any]) -> set[tuple[Any, ...]]:
        from pensive.automata.transducer_simulation import transduce_mealy

        return transduce_mealy(self, word)

    def validate(self) -> None:
        super().validate()
        for transition in self.transitions():
            if ATTR_OUTPUT in transition.data:
                out = transition.data[ATTR_OUTPUT]
                self._require(out in self.output_alphabet, f"output {out!r} not in output alphabet")
            if ATTR_SYMBOL in transition.data:
                sym = transition.data[ATTR_SYMBOL]
                self._require(sym in self.input_alphabet, f"symbol {sym!r} not in input alphabet")


class MooreMachine(Transducer):
    """Output symbols on states."""

    def transduce(self, word: Sequence[Any]) -> set[tuple[Any, ...]]:
        from pensive.automata.transducer_simulation import transduce_moore

        return transduce_moore(self, word)

    def validate(self) -> None:
        super().validate()
        for state in self.states():
            out = self.graph.state_attrs(state).get(ATTR_OUTPUT)
            if out is not None:
                self._require(out in self.output_alphabet, f"output {out!r} not in output alphabet")
