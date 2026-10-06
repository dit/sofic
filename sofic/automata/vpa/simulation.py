"""Stack simulation for visibly pushdown automata."""

from __future__ import annotations

from collections.abc import Hashable, Iterator, Sequence
from typing import Any

from sofic.automata._config_simulation import simulate_configs
from sofic.automata.vpa.base import VisiblyPushdownAutomaton
from sofic.automata.vpa.operations import BOTTOM, normalize

Config = tuple[Hashable, tuple[Any, ...]]


def recognizes_vpa(vpa: VisiblyPushdownAutomaton, word: Sequence[Any]) -> bool:
    """Return whether ``vpa`` accepts ``word``.

    Simulates the normalized form (:func:`~sofic.automata.vpa.operations.normalize`):
    the stack holds only pushed symbols, its emptiness stands for the explicit
    bottom, and a return guarded by ``BOTTOM`` fires only on the empty stack.
    """
    machine = normalize(vpa)
    calls, internals, returns = machine.call_map(), machine.internal_map(), machine.return_map()

    def step(config: Config, symbol: Any) -> Iterator[Config]:
        state, stack = config
        for target, pushed in calls.get((state, symbol), ()):
            yield (target, (*stack, pushed))
        for target in internals.get((state, symbol), ()):
            yield (target, stack)
        for guard, target in returns.get((state, symbol), ()):
            if guard == BOTTOM:
                if not stack:
                    yield (target, stack)
            elif stack and stack[-1] == guard:
                yield (target, stack[:-1])

    initial: set[Config] = {(state, ()) for state in machine.initial}
    current = simulate_configs(initial, word, step)
    return any(state in machine.accepting for state, _ in current)
