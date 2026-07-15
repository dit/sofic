"""Stack simulation for visibly pushdown automata."""

from __future__ import annotations

from collections.abc import Hashable, Iterator, Sequence
from typing import Any

from sofic.automata._config_simulation import simulate_configs
from sofic.automata.vpa import VisiblyPushdownAutomaton
from sofic.graph import ATTR_KIND, ATTR_STACK_SYMBOL, ATTR_SYMBOL, KIND_CALL, KIND_INTERNAL, KIND_RETURN

_BOTTOM = object()

Config = tuple[Hashable, tuple[Any, ...]]


def recognizes_vpa(vpa: VisiblyPushdownAutomaton, word: Sequence[Any]) -> bool:
    """Return whether ``vpa`` accepts ``word``."""
    if vpa.initial_state is None:
        return False

    bottom = _BOTTOM if vpa.bottom_stack_symbol is None else vpa.bottom_stack_symbol
    initial: set[Config] = {(vpa.initial_state, (bottom,))}

    def step(config: Config, symbol: Any) -> Iterator[Config]:
        state, stack = config
        for transition in vpa.graph.out_transitions(state):
            data = transition.data
            if data.get(ATTR_SYMBOL) != symbol:
                continue
            kind = data.get(ATTR_KIND)
            if kind == KIND_CALL:
                stack_sym = data.get(ATTR_STACK_SYMBOL)
                if stack_sym is None:
                    continue
                yield (transition.target, stack + (stack_sym,))
            elif kind == KIND_RETURN:
                stack_sym = data.get(ATTR_STACK_SYMBOL)
                if len(stack) <= 1:
                    if vpa.bottom_stack_symbol is None:
                        continue
                    if stack_sym is not None and stack_sym != stack[-1]:
                        continue
                    yield (transition.target, stack)
                    continue
                if stack_sym is not None and stack_sym != stack[-1]:
                    continue
                yield (transition.target, stack[:-1])
            elif kind == KIND_INTERNAL:
                yield (transition.target, stack)

    current = simulate_configs(initial, word, step)
    return any(state in vpa.accepting_states for state, _ in current)
