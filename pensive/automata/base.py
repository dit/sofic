"""Labeled finite automata base class."""

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Hashable, Iterator, Sequence
from typing import TYPE_CHECKING, Any

from pensive.base import StateMachine
from pensive.graph import ATTR_SYMBOL, EPSILON

if TYPE_CHECKING:
    from pensive.automata.nfa import NFA


class LabeledAutomaton(StateMachine):
    """Finite-word recognizer over an input alphabet."""

    input_alphabet: frozenset[Any]
    initial_states: frozenset[Hashable]
    accepting_states: frozenset[Hashable]

    def __init__(
        self,
        input_alphabet: frozenset[Any] | None = None,
        initial_states: frozenset[Hashable] | None = None,
        accepting_states: frozenset[Hashable] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.input_alphabet = input_alphabet if input_alphabet is not None else frozenset()
        self.initial_states = initial_states if initial_states is not None else frozenset()
        self.accepting_states = accepting_states if accepting_states is not None else frozenset()

    def validate(self) -> None:
        for state in self.initial_states | self.accepting_states:
            self._require(self.graph.has_state(state), f"missing state {state!r}")
        for transition in self.transitions():
            symbol = transition.data.get(ATTR_SYMBOL)
            if symbol is not None and symbol is not EPSILON:
                self._require(symbol in self.input_alphabet, f"symbol {symbol!r} not in input alphabet")

    @abstractmethod
    def recognizes(self, word: Sequence[Any]) -> bool:
        """Return whether ``word`` is accepted."""

    def is_deterministic(self) -> bool:
        """Return whether this automaton is DFA-deterministic."""
        from pensive.properties import is_deterministic_automaton

        return is_deterministic_automaton(self)

    def delta(self, state: Hashable, symbol: Any) -> set[Hashable]:
        targets: set[Hashable] = set()
        for transition in self.graph.out_transitions(state):
            if transition.data.get(ATTR_SYMBOL) == symbol:
                targets.add(transition.target)
        return targets

    def epsilon_closure(self, states: set[Hashable]) -> set[Hashable]:
        closure = set(states)
        stack = list(states)
        while stack:
            state = stack.pop()
            for target in self.delta(state, EPSILON):
                if target not in closure:
                    closure.add(target)
                    stack.append(target)
        return closure

    def _run_nfa(self, word: Sequence[Any]) -> set[Hashable]:
        current = self.epsilon_closure(set(self.initial_states))
        for symbol in word:
            next_states: set[Hashable] = set()
            for state in current:
                next_states.update(self.delta(state, symbol))
            current = self.epsilon_closure(next_states)
        return current

    def reverse(self) -> NFA:
        """Return an NFA recognizing the reversed language."""
        from pensive.automata.nfa import NFA

        return NFA(
            input_alphabet=self.input_alphabet,
            initial_states=frozenset(self.accepting_states),
            accepting_states=frozenset(self.epsilon_closure(set(self.initial_states))),
            graph=self.graph.reverse(),
        )
