"""Input/output simulation for Mealy and Moore transducers."""

from __future__ import annotations

from collections.abc import Hashable, Sequence
from typing import Any

import networkx as nx

from pensive.automata.transducers import MealyMachine, MooreMachine, Transducer
from pensive.exceptions import InfiniteTransductionError
from pensive.graph import ATTR_OUTPUT, ATTR_SYMBOL, EPSILON


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
    return (out,) if out is not None and out is not EPSILON else ()


def _transduce(transducer: Transducer, word: Sequence[Any], *, moore: bool) -> set[tuple[Any, ...]]:
    if not transducer.initial_states:
        return set()

    stack: set[tuple[Hashable, tuple[Any, ...]]] = set()
    for initial in transducer.initial_states:
        prefix = _state_output(transducer, initial) if moore else ()
        stack.add((initial, prefix))
    stack = _epsilon_closure(transducer, stack, moore=moore)

    for symbol in word:
        next_stack: set[tuple[Hashable, tuple[Any, ...]]] = set()
        for state, output_prefix in stack:
            for transition in transducer.graph.out_transitions(state):
                if transition.data.get(ATTR_SYMBOL) != symbol:
                    continue
                if moore:
                    extended = output_prefix + _state_output(transducer, transition.target)
                else:
                    out = transition.data.get(ATTR_OUTPUT)
                    extended = output_prefix + ((out,) if out is not None and out is not EPSILON else ())
                next_stack.add((transition.target, extended))
        stack = _epsilon_closure(transducer, next_stack, moore=moore)
        if not stack:
            return set()

    return {output_prefix for _, output_prefix in stack}


def _epsilon_closure(
    transducer: Transducer,
    configs: set[tuple[Hashable, tuple[Any, ...]]],
    *,
    moore: bool,
) -> set[tuple[Hashable, tuple[Any, ...]]]:
    if not configs:
        return set()
    starts = {state for state, _prefix in configs}
    if _has_productive_epsilon_cycle(transducer, starts, moore=moore):
        raise InfiniteTransductionError("productive epsilon-input cycle gives infinitely many outputs")

    closure = set(configs)
    stack = list(configs)
    while stack:
        state, output_prefix = stack.pop()
        for transition in transducer.graph.out_transitions(state):
            if transition.data.get(ATTR_SYMBOL, EPSILON) is not EPSILON:
                continue
            if moore:
                extended = output_prefix + _state_output(transducer, transition.target)
            else:
                out = transition.data.get(ATTR_OUTPUT)
                extended = output_prefix + ((out,) if out is not None and out is not EPSILON else ())
            config = (transition.target, extended)
            if config not in closure:
                closure.add(config)
                stack.append(config)
    return closure


def _has_productive_epsilon_cycle(
    transducer: Transducer,
    starts: set[Hashable],
    *,
    moore: bool,
) -> bool:
    eps_graph = nx.MultiDiGraph()
    reachable: set[Hashable] = set(starts)
    stack = list(starts)
    while stack:
        state = stack.pop()
        eps_graph.add_node(state)
        for transition in transducer.graph.out_transitions(state):
            if transition.data.get(ATTR_SYMBOL, EPSILON) is not EPSILON:
                continue
            target = transition.target
            eps_graph.add_edge(state, target, transition=transition)
            if target not in reachable:
                reachable.add(target)
                stack.append(target)

    for component in nx.strongly_connected_components(eps_graph):
        if len(component) == 1:
            state = next(iter(component))
            has_cycle = eps_graph.has_edge(state, state)
        else:
            has_cycle = True
        if not has_cycle:
            continue
        for source in component:
            for target in eps_graph.successors(source):
                if target not in component:
                    continue
                for edge_data in eps_graph.get_edge_data(source, target).values():
                    transition = edge_data["transition"]
                    if _epsilon_edge_is_productive(transducer, transition.target, transition.data, moore=moore):
                        return True
    return False


def _epsilon_edge_is_productive(
    transducer: Transducer,
    target: Hashable,
    data: dict[str, Any],
    *,
    moore: bool,
) -> bool:
    if moore:
        return bool(_state_output(transducer, target))
    output = data.get(ATTR_OUTPUT)
    return output is not None and output is not EPSILON
