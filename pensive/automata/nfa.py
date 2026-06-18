"""Nondeterministic finite automata."""

from __future__ import annotations

from collections.abc import Hashable, Sequence
from typing import TYPE_CHECKING, Any

from pensive.automata.base import LabeledAutomaton
from pensive.graph import ATTR_SYMBOL, EPSILON

if TYPE_CHECKING:
    from pensive.automata.algorithms import MinimizationAlgorithm
    from pensive.automata.dfa import DFA


class NFA(LabeledAutomaton):
    """Nondeterministic finite automaton with first-class epsilon transitions."""

    def add_transition(self, source: Hashable, target: Hashable, symbol: Any = EPSILON, **attrs: Any) -> int:
        return self.graph.add_transition(source, target, **{ATTR_SYMBOL: symbol, **attrs})

    def recognizes(self, word: Sequence[Any]) -> bool:
        final = self._run_nfa(word)
        return bool(final & self.accepting_states)

    def determinize(self, *, alphabet: frozenset[Any] | None = None) -> DFA:
        from pensive.automata.algorithms import determinize

        return determinize(self, alphabet=alphabet)

    def minimize(
        self,
        algorithm: MinimizationAlgorithm = "hopcroft",
        *,
        alphabet: frozenset[Any] | None = None,
    ) -> DFA:
        from pensive.automata.algorithms import minimize

        return minimize(self, algorithm=algorithm, alphabet=alphabet)
