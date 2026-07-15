"""SFT construction from forbidden words."""

from __future__ import annotations

from typing import Any

from sofic.graph import ATTR_SYMBOL, TransitionGraph
from sofic.shifts.sft import ShiftOfFiniteType


def from_forbidden_words(
    forbidden: set[tuple[Any, ...]],
    symbol_alphabet: frozenset[Any],
    *,
    max_states: int = 256,
) -> ShiftOfFiniteType:
    """Build an SFT presentation via a follower automaton on allowed prefixes."""
    alphabet = tuple(symbol_alphabet)
    forbidden_set = set(forbidden)
    max_len = max((len(word) for word in forbidden_set), default=0)

    def is_allowed(prefix: tuple[Any, ...]) -> bool:
        return all(not (len(word) <= len(prefix) and prefix[-len(word) :] == word) for word in forbidden_set)

    graph = TransitionGraph()
    start: tuple[Any, ...] = ()
    graph.add_state(start)
    queue = [start]
    seen = {start}

    while queue:
        prefix = queue.pop(0)
        if len(seen) >= max_states:
            break
        for symbol in alphabet:
            extended = prefix + (symbol,)
            if not is_allowed(extended):
                continue
            trimmed = extended
            if max_len > 0:
                trimmed = extended[-max_len:]
            if trimmed not in seen:
                seen.add(trimmed)
                graph.add_state(trimmed)
                queue.append(trimmed)
            graph.add_transition(prefix, trimmed, **{ATTR_SYMBOL: symbol})

    return ShiftOfFiniteType.from_presentation(
        graph,
        symbol_alphabet=symbol_alphabet,
        forbidden_words=forbidden_set,
    )
