"""Shifts of finite type."""

from __future__ import annotations

from typing import Any, Self

from pensive.shifts.base import SymbolicModel
from pensive.graph import TransitionGraph


class ShiftOfFiniteType(SymbolicModel):
    """SFT specified by forbidden words or an explicit presentation."""

    @classmethod
    def from_forbidden_words(
        cls,
        forbidden: set[tuple[Any, ...]],
        symbol_alphabet: frozenset[Any],
        **kwargs: Any,
    ) -> ShiftOfFiniteType:
        from pensive.shifts.sft_construction import from_forbidden_words

        return from_forbidden_words(forbidden, symbol_alphabet, **kwargs)

    @classmethod
    def from_presentation(
        cls,
        graph: TransitionGraph,
        symbol_alphabet: frozenset[Any],
        **kwargs: Any,
    ) -> ShiftOfFiniteType:
        return cls(graph=graph, symbol_alphabet=symbol_alphabet, **kwargs)
