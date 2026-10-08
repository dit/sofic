"""Emptiness, universality, and inclusion for finite-word automata.

Universality and inclusion use the forward antichain algorithms of De Wulf,
Doyen, Henzinger, and Raskin :cite:`DeWulf2006`: the subset construction is
explored on the fly, and a subset is pruned when a subset of it has already been
reached, because every word rejected from the larger subset is also rejected
from the smaller one.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Hashable, Iterable
from typing import TYPE_CHECKING, Any

from sofic.graph import ATTR_SYMBOL, EPSILON

if TYPE_CHECKING:
    from sofic.automata.base import LabeledAutomaton

Successors = dict[Hashable, dict[Any, frozenset[Hashable]]]


def _successors(aut: LabeledAutomaton) -> Successors:
    table: dict[Hashable, dict[Any, set[Hashable]]] = {state: {} for state in aut.states()}
    for transition in aut.transitions():
        symbol = transition.data.get(ATTR_SYMBOL)
        if symbol is None:
            continue
        table.setdefault(transition.source, {}).setdefault(symbol, set()).add(transition.target)
    return {state: {symbol: frozenset(targets) for symbol, targets in row.items()} for state, row in table.items()}


def _closure(table: Successors, states: Iterable[Hashable]) -> frozenset[Hashable]:
    closure = set(states)
    stack = list(closure)
    while stack:
        for target in table.get(stack.pop(), {}).get(EPSILON, ()):
            if target not in closure:
                closure.add(target)
                stack.append(target)
    return frozenset(closure)


def _post(table: Successors, states: Iterable[Hashable], symbol: Any) -> frozenset[Hashable]:
    targets: set[Hashable] = set()
    for state in states:
        targets.update(table.get(state, {}).get(symbol, ()))
    return _closure(table, targets)


def _symbols(table: Successors, state: Hashable) -> list[Any]:
    return sorted((symbol for symbol in table.get(state, {}) if symbol is not EPSILON), key=repr)


def _dominated(antichain: list[frozenset[Hashable]], candidate: frozenset[Hashable]) -> bool:
    return any(member <= candidate for member in antichain)


def _insert(antichain: list[frozenset[Hashable]], candidate: frozenset[Hashable]) -> None:
    antichain[:] = [member for member in antichain if not candidate <= member]
    antichain.append(candidate)


def accepted_word(aut: LabeledAutomaton) -> tuple[Any, ...] | None:
    """Return a shortest accepted word, or ``None`` when the language is empty.

    Breadth-first search over states, layered by the number of non-ε symbols
    read :cite:`HopcroftUllman1979`; ties are broken by ``repr`` order of the
    symbols.
    """
    table = _successors(aut)
    parent: dict[Hashable, tuple[Hashable, Any] | None] = dict.fromkeys(aut.initial_states)

    def close(layer: list[Hashable]) -> list[Hashable]:
        stack = list(layer)
        while stack:
            state = stack.pop()
            for target in table.get(state, {}).get(EPSILON, ()):
                if target not in parent:
                    parent[target] = (state, EPSILON)
                    layer.append(target)
                    stack.append(target)
        return layer

    layer = close(list(parent))
    while layer:
        for state in layer:
            if state in aut.accepting_states:
                return _trace(parent, state)
        following: list[Hashable] = []
        for state in layer:
            for symbol in _symbols(table, state):
                for target in table[state][symbol]:
                    if target not in parent:
                        parent[target] = (state, symbol)
                        following.append(target)
        layer = close(following)
    return None


def _trace(parent: dict[Hashable, tuple[Hashable, Any] | None], state: Hashable) -> tuple[Any, ...]:
    word: list[Any] = []
    step = parent[state]
    while step is not None:
        source, symbol = step
        if symbol is not EPSILON:
            word.append(symbol)
        step = parent[source]
    return tuple(reversed(word))


def is_empty(aut: LabeledAutomaton) -> bool:
    """Return whether ``aut`` accepts no finite word :cite:`HopcroftUllman1979`."""
    return accepted_word(aut) is None


def is_universal(aut: LabeledAutomaton, alphabet: frozenset[Any]) -> bool:
    """Return whether ``aut`` accepts every finite word over ``alphabet``.

    Forward antichain search on the subset construction :cite:`DeWulf2006`:
    a reachable subset without an accepting state witnesses a rejected word,
    and only ``⊆``-minimal subsets need to be explored.
    """
    table = _successors(aut)
    accepting = aut.accepting_states
    symbols = sorted((symbol for symbol in alphabet if symbol is not EPSILON), key=repr)
    start = _closure(table, aut.initial_states)
    if not start & accepting:
        return False
    antichain = [start]
    queue = deque([start])
    while queue:
        current = queue.popleft()
        for symbol in symbols:
            successor = _post(table, current, symbol)
            if not successor & accepting:
                return False
            if _dominated(antichain, successor):
                continue
            _insert(antichain, successor)
            queue.append(successor)
    return True


def includes(larger: LabeledAutomaton, smaller: LabeledAutomaton) -> bool:
    """Return whether ``L(smaller) ⊆ L(larger)``.

    Forward antichain search over pairs ``(q, S)`` of a state ``q`` of
    ``smaller`` and the subset ``S`` of ``larger`` reached on the same word
    :cite:`DeWulf2006`. A pair with ``q`` accepting and ``S`` non-accepting
    witnesses a word in the difference; ``(q, S)`` is pruned when some
    ``(q, S')`` with ``S' ⊆ S`` has already been reached.
    """
    big = _successors(larger)
    small = _successors(smaller)
    antichains: dict[Hashable, list[frozenset[Hashable]]] = {}
    queue: deque[tuple[Hashable, frozenset[Hashable]]] = deque()

    def visit(state: Hashable, subset: frozenset[Hashable]) -> bool:
        if state in smaller.accepting_states and not subset & larger.accepting_states:
            return False
        antichain = antichains.setdefault(state, [])
        if not _dominated(antichain, subset):
            _insert(antichain, subset)
            queue.append((state, subset))
        return True

    start = _closure(big, larger.initial_states)
    for state in _closure(small, smaller.initial_states):
        if not visit(state, start):
            return False
    while queue:
        state, subset = queue.popleft()
        for symbol in _symbols(small, state):
            successor = _post(big, subset, symbol)
            for target in _post(small, (state,), symbol):
                if not visit(target, successor):
                    return False
    return True
