"""Simulation for nested word automata."""

from __future__ import annotations

from collections.abc import Hashable
from typing import Any

from pensive.automata.nwa import NestedWord, NestedWordAutomaton
from pensive.graph import ATTR_HIER_STATE, ATTR_KIND, ATTR_SYMBOL, KIND_CALL, KIND_INTERNAL, KIND_RETURN


def recognizes_nwa(nwa: NestedWordAutomaton, word: NestedWord) -> bool:
    """Return whether ``nwa`` accepts ``word``."""
    if nwa.initial_state is None:
        return False

    word.validate()
    current: set[tuple[Hashable, tuple[Any, ...]]] = {(nwa.initial_state, ())}
    for symbol, kind, partner in zip(word.symbols, word.kinds, word.matching, strict=True):
        next_configs: set[tuple[Hashable, tuple[Any, ...]]] = set()
        for state, stack in current:
            for transition in nwa.graph.out_transitions(state):
                data = transition.data
                if data.get(ATTR_SYMBOL) != symbol or data.get(ATTR_KIND) != kind:
                    continue
                if kind == KIND_CALL:
                    hier_state = data.get(ATTR_HIER_STATE)
                    next_stack = stack + (hier_state,) if partner is not None else stack
                    next_configs.add((transition.target, next_stack))
                elif kind == KIND_RETURN:
                    _advance_return(nwa, transition.target, data.get(ATTR_HIER_STATE), partner, stack, next_configs)
                elif kind == KIND_INTERNAL:
                    next_configs.add((transition.target, stack))
        if not next_configs:
            return False
        current = next_configs

    return any(state in nwa.accepting_states for state, _stack in current)


def _advance_return(
    nwa: NestedWordAutomaton,
    target: Hashable,
    hier_state: Any,
    partner: int | None,
    stack: tuple[Any, ...],
    next_configs: set[tuple[Hashable, tuple[Any, ...]]],
) -> None:
    if partner is None:
        if nwa.bottom_hier_state is not None and hier_state == nwa.bottom_hier_state:
            next_configs.add((target, stack))
        return
    if stack and stack[-1] == hier_state:
        next_configs.add((target, stack[:-1]))
