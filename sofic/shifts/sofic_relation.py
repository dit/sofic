"""Sofic relations: subshifts over a product alphabet ``X x Y``.

The topological support of a transducer is a subshift of the product shift on
``X x Y`` -- a "sofic relation" whose input and output projections are the
transducer's domain and range subshifts. This is the symbolic-dynamics reading
of a transducer, complementary to the sliding block code (Lind & Marcus, *An
Introduction to Symbolic Dynamics and Coding* (1995), ch. 6).
"""

from __future__ import annotations

from typing import Any

from sofic.graph import ATTR_OUTPUT, ATTR_SYMBOL, EPSILON
from sofic.shifts.sofic import SoficShift


class SoficRelation(SoficShift):
    """A sofic subshift whose symbols are ``(input, output)`` pairs."""

    @classmethod
    def from_transducer(cls, transducer: Any) -> SoficRelation:
        """Build the topological support of a transducer (probabilities dropped)."""
        relation = cls()
        pairs: set[tuple[Any, Any]] = set()
        for state in transducer.states():
            relation.graph.add_state(state)
        for transition in transducer.transitions():
            input_symbol = transition.data.get(ATTR_SYMBOL, EPSILON)
            output_symbol = transition.data.get(ATTR_OUTPUT, EPSILON)
            pair = (input_symbol, output_symbol)
            relation.add_transition(transition.source, transition.target, pair)
            pairs.add(pair)
        relation.symbol_alphabet = frozenset(pairs)
        return relation

    def input_alphabet(self) -> frozenset[Any]:
        """Return the set of input symbols appearing in the relation."""
        return frozenset(pair[0] for pair in self.symbol_alphabet)

    def output_alphabet(self) -> frozenset[Any]:
        """Return the set of output symbols appearing in the relation."""
        return frozenset(pair[1] for pair in self.symbol_alphabet)

    def input_shift(self) -> SoficShift:
        """Return the projection onto the input coordinate as a sofic shift."""
        return self._project(0)

    def output_shift(self) -> SoficShift:
        """Return the projection onto the output coordinate as a sofic shift."""
        return self._project(1)

    def _project(self, index: int) -> SoficShift:
        projected = SoficShift()
        symbols: set[Any] = set()
        for state in self.states():
            projected.graph.add_state(state)
        for transition in self.transitions():
            pair = transition.data.get(ATTR_SYMBOL)
            symbol = pair[index] if isinstance(pair, tuple) else pair
            projected.add_transition(transition.source, transition.target, symbol)
            symbols.add(symbol)
        projected.symbol_alphabet = frozenset(symbols)
        return projected
