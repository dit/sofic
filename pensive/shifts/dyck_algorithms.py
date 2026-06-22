"""Algorithms for Dyck shift presentations."""

from __future__ import annotations

from collections import deque
from collections.abc import Hashable, Iterator, Sequence
from typing import TYPE_CHECKING, Any

from pensive.graph import ATTR_KIND, ATTR_SYMBOL, KIND_CALL, KIND_INTERNAL, KIND_RETURN, Transition
from pensive.shifts.sofic_dyck import TransitionRef, transition_ref

if TYPE_CHECKING:
    from pensive.shifts.sofic_dyck import SoficDyckShift

Configuration = tuple[Hashable, tuple[TransitionRef, ...]]


def is_admissible_word(shift: SoficDyckShift, word: Sequence[Any]) -> bool:
    """Return whether ``word`` labels an admissible finite Dyck path."""
    word = tuple(word)
    if not word:
        return True
    if any(symbol not in shift.symbol_alphabet for symbol in word):
        return False

    current: set[Configuration] = {(state, ()) for state in shift.states()}
    for symbol in word:
        next_configs: set[Configuration] = set()
        for state, stack in current:
            for transition in shift.graph.out_transitions(state):
                if transition.data.get(ATTR_SYMBOL) != symbol:
                    continue
                _advance_transition(shift, transition, stack, next_configs)
        if not next_configs:
            return False
        current = next_configs

    return bool(current)


def admissible_words(shift: SoficDyckShift, length: int) -> Iterator[tuple[Any, ...]]:
    """Yield distinct admissible words of exactly ``length`` symbols."""
    if length <= 0:
        yield ()
        return

    seen_words: set[tuple[Any, ...]] = set()
    seen_configs: set[tuple[Hashable, tuple[Any, ...], tuple[TransitionRef, ...]]] = set()
    queue: deque[tuple[Hashable, tuple[Any, ...], tuple[TransitionRef, ...]]] = deque()
    for state in shift.states():
        config = (state, (), ())
        queue.append(config)
        seen_configs.add(config)

    while queue:
        state, prefix, stack = queue.popleft()
        if len(prefix) == length:
            if prefix not in seen_words:
                seen_words.add(prefix)
                yield prefix
            continue

        for transition in shift.graph.out_transitions(state):
            symbol = transition.data.get(ATTR_SYMBOL)
            if symbol is None:
                continue
            for target, next_stack in _successors(shift, transition, stack):
                config = (target, prefix + (symbol,), next_stack)
                if config in seen_configs:
                    continue
                seen_configs.add(config)
                queue.append(config)


def _advance_transition(
    shift: SoficDyckShift,
    transition: Transition,
    stack: tuple[TransitionRef, ...],
    next_configs: set[Configuration],
) -> None:
    for target, next_stack in _successors(shift, transition, stack):
        next_configs.add((target, next_stack))


def _successors(
    shift: SoficDyckShift,
    transition: Transition,
    stack: tuple[TransitionRef, ...],
) -> Iterator[tuple[Hashable, tuple[TransitionRef, ...]]]:
    kind = transition.data.get(ATTR_KIND)
    ref = transition_ref(transition)
    if kind == KIND_CALL:
        yield transition.target, stack + (ref,)
    elif kind == KIND_RETURN:
        if not stack:
            yield transition.target, stack
        elif (stack[-1], ref) in shift.matched_edges:
            yield transition.target, stack[:-1]
    elif kind == KIND_INTERNAL:
        yield transition.target, stack
