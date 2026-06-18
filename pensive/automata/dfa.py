"""Deterministic finite automata."""

from __future__ import annotations

from collections.abc import Hashable, Sequence
from typing import TYPE_CHECKING, Any

from pensive.automata.base import LabeledAutomaton
from pensive.exceptions import NonDeterministicError
from pensive.graph import ATTR_SYMBOL, EPSILON

if TYPE_CHECKING:
    from pensive.automata.algorithms import MinimizationAlgorithm
    from pensive.automata.nfa import NFA


class DFA(LabeledAutomaton):
    """Deterministic finite automaton."""

    def validate(self) -> None:
        super().validate()
        self._check_determinism()

    def _check_determinism(self) -> None:
        if self.is_deterministic():
            return
        for transition in self.transitions():
            if transition.data.get(ATTR_SYMBOL) is EPSILON:
                raise NonDeterministicError("DFA does not allow epsilon transitions")
        if len(self.initial_states) != 1:
            raise NonDeterministicError("DFA requires exactly one initial state")
        seen: dict[tuple[Hashable, Any], Hashable] = {}
        for transition in self.transitions():
            symbol = transition.data.get(ATTR_SYMBOL)
            if symbol is None:
                continue
            key = (transition.source, symbol)
            if key in seen and seen[key] != transition.target:
                raise NonDeterministicError(f"non-deterministic transition on {key}")
            seen[key] = transition.target
        raise NonDeterministicError("DFA determinism check failed")

    def add_transition(self, source: Hashable, target: Hashable, symbol: Any, **attrs: Any) -> int:
        if symbol is EPSILON:
            raise NonDeterministicError("DFA does not allow epsilon transitions")
        key = (source, symbol)
        for transition in self.graph.out_transitions(source):
            if transition.data.get(ATTR_SYMBOL) == symbol and transition.target != target:
                raise NonDeterministicError(f"non-deterministic transition on {key}")
        return self.graph.add_transition(source, target, **{ATTR_SYMBOL: symbol, **attrs})

    def recognizes(self, word: Sequence[Any]) -> bool:
        if not self.initial_states:
            return False
        state = next(iter(self.initial_states))
        for symbol in word:
            successors = self.delta(state, symbol)
            if len(successors) != 1:
                return False
            state = next(iter(successors))
        return state in self.accepting_states

    def determinize(self) -> DFA:
        from pensive.automata.algorithms import trim

        return trim(self)

    @classmethod
    def from_nfa(cls, nfa: NFA, **kwargs: Any) -> DFA:
        from pensive.automata.algorithms import determinize
        from pensive.automata.nfa import NFA as NFAClass

        if not isinstance(nfa, NFAClass):
            raise TypeError("from_nfa requires an NFA")
        return determinize(nfa)

    def minimize(
        self,
        algorithm: MinimizationAlgorithm = "hopcroft",
        *,
        alphabet: frozenset[Any] | None = None,
    ) -> DFA:
        from pensive.automata.algorithms import minimize

        return minimize(self, algorithm=algorithm, alphabet=alphabet)
