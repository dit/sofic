"""Unifilar labeled automata."""

from __future__ import annotations

from collections.abc import Hashable, Sequence
from typing import Any

from sofic.automata.base import LabeledAutomaton
from sofic.exceptions import UnifilarityError
from sofic.graph import ATTR_SYMBOL, EPSILON


class UnifilarAutomaton(LabeledAutomaton):
    """At most one outgoing edge per (state, symbol), epsilon excluded."""

    def is_unifilar(self) -> bool:
        """Return whether this presentation is right-resolving (unifilar)."""
        from sofic.properties import is_unifilar_symbols

        return is_unifilar_symbols(self)

    def validate(self) -> None:
        super().validate()
        if self.is_unifilar():
            return
        seen: set[tuple[Hashable, Any]] = set()
        for transition in self.transitions():
            symbol = transition.data.get(ATTR_SYMBOL)
            if symbol is EPSILON or symbol is None:
                continue
            key = (transition.source, symbol)
            if key in seen:
                raise UnifilarityError(f"duplicate unifilar transition on {key}")
            seen.add(key)

    def recognizes(self, word: Sequence[Any]) -> bool:
        return bool(self._run_nfa(word) & self.accepting_states)

    def markov_order(self) -> int | float:
        """Markov order ``R`` for this right-resolving (unifilar) presentation."""
        from sofic.generators.synchronization import graph_from_unifilar_automaton, markov_order_from_graph

        return markov_order_from_graph(graph_from_unifilar_automaton(self))

    def cryptic_order(self) -> int | float:
        """Cryptic order ``k_chi`` for this right-resolving presentation."""
        from sofic.generators.synchronization import cryptic_order_from_graph, graph_from_unifilar_automaton

        return cryptic_order_from_graph(graph_from_unifilar_automaton(self))

    def is_exactly_synchronizable(self) -> bool:
        """Return whether this presentation has finite Markov order."""
        from sofic.generators.synchronization import graph_from_unifilar_automaton, is_exactly_synchronizable

        return is_exactly_synchronizable(graph_from_unifilar_automaton(self))
