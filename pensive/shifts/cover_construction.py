"""Fischer and Krieger cover constructions."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from pensive.graph import ATTR_SYMBOL, TransitionGraph
from pensive.shifts.covers import LeftFischerCover, LeftKriegerCover, RightFischerCover, RightKriegerCover
from pensive.shifts.sofic import SoficShift
from pensive.states import sequential_labels


def _follower_language(shift: SoficShift, vertex: Any, max_len: int = 8) -> frozenset[tuple[Any, ...]]:
    from collections import deque

    seen: set[tuple[Any, ...]] = set()
    queue: deque[tuple[Any, tuple[Any, ...]]] = deque([(vertex, ())])
    while queue:
        state, prefix = queue.popleft()
        if len(prefix) > max_len:
            continue
        if prefix:
            seen.add(prefix)
        for transition in shift.graph.out_transitions(state):
            symbol = transition.data.get(ATTR_SYMBOL)
            if symbol is None:
                continue
            queue.append((transition.target, prefix + (symbol,)))
    return frozenset(seen)


def _build_left_fischer(shift: SoficShift) -> LeftFischerCover:
    followers: dict[Any, frozenset[tuple[Any, ...]]] = {
        vertex: _follower_language(shift, vertex) for vertex in shift.states()
    }
    classes: dict[frozenset[tuple[Any, ...]], list[Any]] = defaultdict(list)
    for vertex, language in followers.items():
        classes[language].append(vertex)

    graph = TransitionGraph()
    class_for_vertex = {vertex: language for vertex, language in followers.items()}
    state_ids = {
        language: sequential_labels(len(classes))[index]
        for index, language in enumerate(classes)
    }
    for _language, state_id in state_ids.items():
        graph.add_state(state_id)

    for transition in shift.transitions():
        source_lang = class_for_vertex[transition.source]
        target_lang = class_for_vertex[transition.target]
        symbol = transition.data.get(ATTR_SYMBOL)
        graph.add_transition(state_ids[source_lang], state_ids[target_lang], **{ATTR_SYMBOL: symbol})

    return LeftFischerCover(
        graph=graph,
        symbol_alphabet=shift.symbol_alphabet,
    )


def left_fischer_from_sofic(shift: SoficShift) -> LeftFischerCover:
    return _build_left_fischer(shift.trim_transient())


def right_fischer_from_sofic(shift: SoficShift) -> RightFischerCover:
    left = left_fischer_from_sofic(shift.reverse())
    return RightFischerCover(
        graph=left.graph.copy(),
        symbol_alphabet=left.symbol_alphabet,
    )


def left_krieger_from_sofic(shift: SoficShift) -> LeftKriegerCover:
    raise NotImplementedError(
        "Left Krieger cover construction is not yet implemented; use left_fischer_from_sofic"
    )


def right_krieger_from_sofic(shift: SoficShift) -> RightKriegerCover:
    raise NotImplementedError(
        "Right Krieger cover construction is not yet implemented; use right_fischer_from_sofic"
    )
