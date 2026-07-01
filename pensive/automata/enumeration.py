"""Finite-word enumeration for automata."""

from __future__ import annotations

from collections.abc import Iterator
from itertools import product
from typing import Any

from pensive.automata.base import LabeledAutomaton


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
    from pensive.automata.algorithms import _effective_alphabet as _shared

    return _shared(automaton)
