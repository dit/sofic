"""Shared word enumeration for automata modules."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from typing import Any

Word = tuple[Any, ...]


def _words_up_to(max_length: int, alphabet: Iterable[Any]) -> Iterator[Word]:
    """Yield every word of length at most ``max_length`` in breadth-first order."""
    symbols = tuple(alphabet)
    frontier: list[Word] = [()]
    yield ()
    for _ in range(max_length):
        nxt: list[Word] = []
        for word in frontier:
            for symbol in symbols:
                extended = (*word, symbol)
                yield extended
                nxt.append(extended)
        frontier = nxt
