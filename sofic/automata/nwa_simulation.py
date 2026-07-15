"""Simulation for nested word automata."""

from __future__ import annotations

from collections.abc import Hashable, Iterator
from typing import Any

from sofic.automata._config_simulation import simulate_configs
from sofic.automata.nwa import NestedWord, NestedWordAutomaton
from sofic.graph import ATTR_HIER_STATE, ATTR_KIND, ATTR_SYMBOL, KIND_CALL, KIND_INTERNAL, KIND_RETURN

Config = tuple[Hashable, tuple[Any, ...]]


def recognizes_nwa(nwa: NestedWordAutomaton, word: NestedWord) -> bool:
    """Return whether ``nwa`` accepts ``word``."""
    if nwa.initial_state is None:
        return False

    word.validate()
    initial: set[Config] = {(nwa.initial_state, ())}
    steps = zip(word.symbols, word.kinds, word.matching, strict=True)

    def step(config: Config, item: tuple[Any, Any, int | None]) -> Iterator[Config]:
        state, stack = config
        symbol, kind, partner = item
        for transition in nwa.graph.out_transitions(state):
            data = transition.data
            if data.get(ATTR_SYMBOL) != symbol or data.get(ATTR_KIND) != kind:
                continue
            if kind == KIND_CALL:
                hier_state = data.get(ATTR_HIER_STATE)
                next_stack = stack + (hier_state,) if partner is not None else stack
                yield (transition.target, next_stack)
            elif kind == KIND_RETURN:
                yield from _advance_return(nwa, transition.target, data.get(ATTR_HIER_STATE), partner, stack)
            elif kind == KIND_INTERNAL:
                yield (transition.target, stack)

    current = simulate_configs(initial, steps, step)
    return any(state in nwa.accepting_states for state, _stack in current)


def _advance_return(
    nwa: NestedWordAutomaton,
    target: Hashable,
    hier_state: Any,
    partner: int | None,
    stack: tuple[Any, ...],
) -> Iterator[Config]:
    if partner is None:
        if nwa.bottom_hier_state is not None and hier_state == nwa.bottom_hier_state:
            yield (target, stack)
        return
    if stack and stack[-1] == hier_state:
        yield (target, stack[:-1])
