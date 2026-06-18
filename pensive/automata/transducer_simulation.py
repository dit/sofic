"""Input/output simulation for Mealy and Moore transducers."""

from __future__ import annotations

from collections.abc import Hashable, Sequence
from typing import Any

from pensive.automata.transducers import MealyMachine, MooreMachine, Transducer
from pensive.graph import ATTR_OUTPUT, ATTR_SYMBOL


def transduce_mealy(mealy: MealyMachine, word: Sequence[Any]) -> set[tuple[Any, ...]]:
    """Return possible output sequences for ``word`` on a Mealy machine.

    Each transition appends its ``ATTR_OUTPUT`` after the corresponding input symbol.
    """
    return _transduce(transducer=mealy, word=word, moore=False)


def transduce_moore(moore: MooreMachine, word: Sequence[Any]) -> set[tuple[Any, ...]]:
    """Return possible output sequences for ``word`` on a Moore machine.

    Outputs follow the standard Moore convention: emit the initial state's output,
    then after each input symbol emit the output of the entered state (length
    ``len(word) + 1`` when states carry outputs).
    """
    return _transduce(transducer=moore, word=word, moore=True)


def _state_output(transducer: Transducer, state: Hashable) -> tuple[Any, ...]:
    out = transducer.graph.state_attrs(state).get(ATTR_OUTPUT)
    return (out,) if out is not None else ()


def _transduce(transducer: Transducer, word: Sequence[Any], *, moore: bool) -> set[tuple[Any, ...]]:
    if not transducer.initial_states:
        return set()

    stack: list[tuple[Hashable, tuple[Any, ...]]] = []
    for initial in transducer.initial_states:
        prefix = _state_output(transducer, initial) if moore else ()
        stack.append((initial, prefix))

    for symbol in word:
        next_stack: list[tuple[Hashable, tuple[Any, ...]]] = []
        for state, output_prefix in stack:
            for transition in transducer.graph.out_transitions(state):
                if transition.data.get(ATTR_SYMBOL) != symbol:
                    continue
                if moore:
                    extended = output_prefix + _state_output(transducer, transition.target)
                else:
                    out = transition.data.get(ATTR_OUTPUT)
                    extended = output_prefix + ((out,) if out is not None else ())
                next_stack.append((transition.target, extended))
        stack = next_stack
        if not stack:
            return set()

    return {output_prefix for _, output_prefix in stack}
