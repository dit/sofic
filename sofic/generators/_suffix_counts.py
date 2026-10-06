"""Sliding-window suffix scan shared by the CSSR history counters."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import Any


def infer_alphabet(tokens: Sequence[Any], alphabet: Sequence[Any] | None) -> tuple[Any, ...]:
    """``alphabet`` as a tuple, or the distinct ``tokens`` sorted by ``repr``."""
    return tuple(sorted(set(tokens), key=repr)) if alphabet is None else tuple(alphabet)


def iter_suffixes(tokens: tuple[Any, ...], max_length: int) -> Iterator[tuple[int, tuple[Any, ...]]]:
    """Yield ``(t, tokens[t - L : t])`` for each position ``t`` and each ``L`` in ``0..min(t, max_length)``.

    These are the pasts, up to ``max_length`` long, that precede the token at ``t``.
    """
    for t in range(len(tokens)):
        for length in range(min(t, max_length) + 1):
            yield t, tokens[t - length : t]
