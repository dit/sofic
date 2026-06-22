"""Finite-word enumeration for automata."""

from __future__ import annotations

from collections.abc import Iterator
from itertools import product
from typing import Any

from pensive.automata.base import LabeledAutomaton
from pensive.graph import ATTR_SYMBOL, EPSILON


def words_of_length(automaton: LabeledAutomaton, length: int) -> Iterator[tuple[Any, ...]]:
    """Yield accepted words of exactly ``length`` symbols."""
    if length < 0:
        raise ValueError("length must be nonnegative")
    alphabet = sorted(_effective_alphabet(automaton), key=repr)
    if length == 0:
        if automaton.recognizes(()):
            yield ()
        return
    if not alphabet:
        return
    for word in product(alphabet, repeat=length):
        if automaton.recognizes(word):
            yield word


def iter_language(
    automaton: LabeledAutomaton,
    max_length: int | None = None,
) -> Iterator[tuple[Any, ...]]:
    """Yield accepted words in nondecreasing length order."""
    if max_length is not None and max_length < 0:
        raise ValueError("max_length must be nonnegative")
    length = 0
    while max_length is None or length <= max_length:
        yield from words_of_length(automaton, length)
        length += 1


def _effective_alphabet(automaton: LabeledAutomaton) -> frozenset[Any]:
    symbols = {symbol for symbol in automaton.input_alphabet if symbol is not EPSILON}
    if symbols:
        return frozenset(symbols)
    for transition in automaton.transitions():
        symbol = transition.data.get(ATTR_SYMBOL)
        if symbol is not None and symbol is not EPSILON:
            symbols.add(symbol)
    return frozenset(symbols)
