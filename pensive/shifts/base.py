"""Symbolic dynamical systems base class."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any, Self

from pensive.base import StateMachine
from pensive.graph import ATTR_SYMBOL


class SymbolicModel(StateMachine):
    """Labeled transition system for shift presentations."""

    symbol_alphabet: frozenset[Any]

    def __init__(self, symbol_alphabet: frozenset[Any] | None = None, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.symbol_alphabet = symbol_alphabet if symbol_alphabet is not None else frozenset()

    def validate(self) -> None:
        for transition in self.transitions():
            symbol = transition.data.get(ATTR_SYMBOL)
            if symbol is not None:
                self._require(symbol in self.symbol_alphabet, f"symbol {symbol!r} not in alphabet")

    def factor_language(self, length: int) -> Iterator[tuple[Any, ...]]:
        from pensive.shifts.algorithms import factor_language

        yield from factor_language(self, length)

    def is_unifilar(self) -> bool:
        """Return whether this presentation is right-resolving (unifilar)."""
        from pensive.properties import is_unifilar_symbols

        return is_unifilar_symbols(self)

    def trim_transient(self) -> Self:
        from pensive.shifts.algorithms import trim_transient

        return trim_transient(self)
