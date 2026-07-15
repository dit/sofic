"""Labeled finite automata base class."""

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Hashable, Iterator, Sequence
from typing import TYPE_CHECKING, Any

from sofic.base import StateMachine
from sofic.graph import ATTR_SYMBOL, EPSILON

if TYPE_CHECKING:
    from sofic.automata.dfa import DFA
    from sofic.automata.nfa import NFA


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

    def words_of_length(self, length: int) -> Iterator[tuple[Any, ...]]:
        """Yield accepted words of exactly ``length`` symbols."""
        from sofic.automata.enumeration import words_of_length

        yield from words_of_length(self, length)

    def iter_language(self, max_length: int | None = None) -> Iterator[tuple[Any, ...]]:
        """Yield accepted words in nondecreasing length order.

        If ``max_length`` is omitted, the iterator is unbounded and may not
        terminate for finite languages after yielding their last word.
        """
        from sofic.automata.enumeration import iter_language

        yield from iter_language(self, max_length=max_length)

    def to_regex(self) -> str:
        """Return a regular expression for the accepted language."""
        from sofic.automata.regex import automaton_to_regex

        return automaton_to_regex(self)

    def union(self, other: LabeledAutomaton) -> NFA:
        """Return an NFA recognizing the union of this language and ``other``."""
        from sofic.automata.languages.automaton_ops import union_nfa

        return union_nfa(self, other)

    def intersection(self, other: LabeledAutomaton) -> DFA:
        """Return a DFA recognizing the intersection with ``other``."""
        from sofic.automata.languages.automaton_ops import intersection_dfa

        return intersection_dfa(self, other)

    def intersect(self, other: LabeledAutomaton) -> DFA:
        """Alias for :meth:`intersection`."""
        return self.intersection(other)

    def complement(self, alphabet: frozenset[Any] | None = None) -> DFA:
        """Return a complete DFA recognizing the complement over ``alphabet``."""
        from sofic.automata.languages.automaton_ops import complement_dfa

        return complement_dfa(self, self.input_alphabet if alphabet is None else alphabet)

    def difference(self, other: LabeledAutomaton, alphabet: frozenset[Any] | None = None) -> DFA:
        """Return a DFA recognizing this language minus ``other``."""
        from sofic.automata.languages.automaton_ops import difference_dfa

        return difference_dfa(self, other, alphabet=alphabet)

    def concat(self, other: LabeledAutomaton) -> NFA:
        """Return an NFA recognizing concatenation with ``other``."""
        from sofic.automata.languages.automaton_ops import concat_nfa

        return concat_nfa(self, other)

    def concatenate(self, other: LabeledAutomaton) -> NFA:
        """Alias for :meth:`concat`."""
        return self.concat(other)

    def kleene_star(self) -> NFA:
        """Return an NFA recognizing the Kleene star of this language."""
        from sofic.automata.languages.automaton_ops import kleene_star_nfa

        return kleene_star_nfa(self)

    def star(self) -> NFA:
        """Alias for :meth:`kleene_star`."""
        return self.kleene_star()

    def is_deterministic(self) -> bool:
        """Return whether this automaton is DFA-deterministic."""
        from sofic.properties import is_deterministic_automaton

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

    def _run_nfa(self, word: Sequence[Any], start: set[Hashable] | None = None) -> set[Hashable]:
        seed = set(self.initial_states) if start is None else set(start)
        current = self.epsilon_closure(seed)
        for symbol in word:
            next_states: set[Hashable] = set()
            for state in current:
                next_states.update(self.delta(state, symbol))
            current = self.epsilon_closure(next_states)
        return current

    def reverse(self) -> NFA:
        """Return an NFA recognizing the reversed language."""
        from sofic.automata.nfa import NFA

        return NFA(
            input_alphabet=self.input_alphabet,
            initial_states=frozenset(self.accepting_states),
            accepting_states=frozenset(self.epsilon_closure(set(self.initial_states))),
            graph=self.graph.reverse(),
        )
