"""Wheeler presentations of sofic shifts.

A shift presents its factor language with every state both initial and
accepting, which is the "all states initial" case of :cite:`Gagie2017`
Theorem 6. The minimal right-resolving presentation of a shift is rarely
Wheeler on its own -- a state reachable on two different symbols already
violates the axioms -- but remembering the last few symbols restores input
consistency, and for many shifts that is enough.

Not every sofic shift has a Wheeler presentation at any order: Wheeler
languages are star-free :cite:`ShyrThierrin1974` :cite:`Alanko2021`, so a shift
whose syntactic monoid contains a nontrivial group, such as the even shift,
is excluded outright.
"""

from __future__ import annotations

from collections import defaultdict, deque
from collections.abc import Hashable
from typing import Any

from sofic.automata.wheeler import WheelerError, WheelerOrder, labeled_graph, wheeler_order
from sofic.graph import ATTR_SYMBOL, TransitionGraph
from sofic.shifts.covers import WheelerCover
from sofic.shifts.sofic import SoficShift

#: How many symbols of memory :func:`wheeler_cover` will add before giving up.
DEFAULT_MAX_ORDER = 6

#: Word length used to compare follower languages when merging states.
DEFAULT_FOLLOWER_DEPTH = 8

BlockState = tuple[tuple[Any, ...], Hashable]


def higher_block_presentation(shift: SoficShift, order: int) -> SoficShift:
    """Return the order-``order`` refinement of ``shift``.

    States become ``(last order symbols, original state)``. Every edge into
    such a state carries the last symbol of its word, so any refinement of
    order at least one is input-consistent -- the cheapest necessary condition
    for the Wheeler axioms.
    """
    if order < 0:
        raise ValueError("order must be nonnegative")
    if order == 0:
        return shift

    outgoing: dict[Hashable, list[tuple[Any, Hashable]]] = defaultdict(list)
    for transition in shift.transitions():
        symbol = transition.data.get(ATTR_SYMBOL)
        if symbol is not None:
            outgoing[transition.source].append((symbol, transition.target))

    current: set[BlockState] = {((), state) for state in shift.states()}
    for _step in range(order):
        current = {
            ((word + (symbol,))[-order:], target) for word, state in current for symbol, target in outgoing[state]
        }

    # Close under transitions so the refinement is a genuine presentation.
    reachable = set(current)
    queue = deque(current)
    edges: set[tuple[BlockState, Any, BlockState]] = set()
    while queue:
        word, state = queue.popleft()
        for symbol, target in outgoing[state]:
            successor = ((word + (symbol,))[-order:], target)
            edges.add(((word, state), symbol, successor))
            if successor not in reachable:
                reachable.add(successor)
                queue.append(successor)

    graph = TransitionGraph()
    for block in reachable:
        graph.add_state(block)
    for source, symbol, target in edges:
        graph.add_transition(source, target, **{ATTR_SYMBOL: symbol})
    return SoficShift(graph=graph, symbol_alphabet=shift.symbol_alphabet).trim_transient()


def right_resolve(shift: SoficShift) -> SoficShift:
    """Return a right-resolving presentation of the same factor language.

    Subset construction seeded with every state, since a shift presents its
    factor language with all states initial. Returns ``shift`` unchanged when
    it is already right-resolving.
    """
    if shift.is_unifilar():
        return shift

    outgoing: dict[Hashable, list[tuple[Any, Hashable]]] = defaultdict(list)
    for transition in shift.transitions():
        symbol = transition.data.get(ATTR_SYMBOL)
        if symbol is not None:
            outgoing[transition.source].append((symbol, transition.target))

    start = frozenset(shift.states())
    subsets = {start}
    queue = deque([start])
    edges: set[tuple[frozenset[Hashable], Any, frozenset[Hashable]]] = set()
    while queue:
        current = queue.popleft()
        successors: dict[Any, set[Hashable]] = defaultdict(set)
        for state in current:
            for symbol, target in outgoing[state]:
                successors[symbol].add(target)
        for symbol, targets in successors.items():
            successor = frozenset(targets)
            edges.add((current, symbol, successor))
            if successor not in subsets:
                subsets.add(successor)
                queue.append(successor)

    graph = TransitionGraph()
    for subset in subsets:
        graph.add_state(subset)
    for source, symbol, target in edges:
        graph.add_transition(source, target, **{ATTR_SYMBOL: symbol})
    return SoficShift(graph=graph, symbol_alphabet=shift.symbol_alphabet).trim_transient()


def wheeler_cover(
    shift: SoficShift,
    *,
    max_order: int = DEFAULT_MAX_ORDER,
    follower_depth: int = DEFAULT_FOLLOWER_DEPTH,
) -> WheelerCover:
    """Return the smallest Wheeler presentation found for ``shift``.

    Right-resolves the presentation, then tries it as given followed by its
    order-1, order-2, ... refinements up to ``max_order``, merging the first
    Wheeler one down. Raises :class:`~sofic.automata.wheeler.WheelerError` when
    no refinement in range is Wheeler; that is evidence, not proof, that the
    shift is non-Wheeler.
    """
    trimmed = right_resolve(shift.trim_transient())
    for order in range(max_order + 1):
        candidate = higher_block_presentation(trimmed, order)
        found = wheeler_order(candidate)
        if found is not None:
            return _merge_wheeler_runs(candidate, found, follower_depth=follower_depth)
    raise WheelerError(
        f"no Wheeler presentation of order <= {max_order}; Wheeler languages are star-free, "
        "so a shift that counts modulo anything (the even shift, for one) has none at any order"
    )


def is_wheeler_shift(shift: SoficShift, *, max_order: int = DEFAULT_MAX_ORDER) -> bool:
    """Return whether some refinement of ``shift`` up to ``max_order`` is Wheeler."""
    try:
        wheeler_cover(shift, max_order=max_order)
    except WheelerError:
        return False
    return True


def wheeler_order_of_shift(shift: SoficShift) -> int | None:
    """Smallest refinement order at which ``shift`` becomes Wheeler, or ``None``.

    Zero means the right-resolving presentation is already Wheeler. This is a
    property of the presentation as much as of the shift, since refining a
    non-minimal presentation can need more memory than refining a minimal one.
    """
    trimmed = right_resolve(shift.trim_transient())
    for order in range(DEFAULT_MAX_ORDER + 1):
        if wheeler_order(higher_block_presentation(trimmed, order)) is not None:
            return order
    return None


def _follower_signature(shift: SoficShift, depth: int) -> dict[Hashable, frozenset[tuple[Any, ...]]]:
    """Bounded follower language of every state, the shift analogue of Nerode classes."""
    outgoing: dict[Hashable, list[tuple[Any, Hashable]]] = defaultdict(list)
    for transition in shift.transitions():
        symbol = transition.data.get(ATTR_SYMBOL)
        if symbol is not None:
            outgoing[transition.source].append((symbol, transition.target))

    signatures: dict[Hashable, frozenset[tuple[Any, ...]]] = {}
    for state in shift.states():
        words: set[tuple[Any, ...]] = set()
        queue: deque[tuple[Hashable, tuple[Any, ...]]] = deque([(state, ())])
        while queue:
            current, prefix = queue.popleft()
            if len(prefix) >= depth:
                continue
            for symbol, target in outgoing[current]:
                word = prefix + (symbol,)
                words.add(word)
                queue.append((target, word))
        signatures[state] = frozenset(words)
    return signatures


def _merge_wheeler_runs(shift: SoficShift, order: WheelerOrder, *, follower_depth: int) -> WheelerCover:
    """Collapse Wheeler-consecutive states that share an incoming label and followers.

    Merging a range of states whose incoming edges all carry one label keeps the
    graph Wheeler :cite:`Gagie2017`, and equal follower languages make the merge
    language-preserving; together these are the merges of the minimum-WDFA
    construction :cite:`Alanko2020`. Only *adjacent* runs collapse, so a
    presentation carrying states the Wheeler order separates -- a source state
    with no incoming edges, for one -- keeps them.
    """
    in_labels = labeled_graph(shift).in_labels()
    followers = _follower_signature(shift, follower_depth)

    runs: list[list[Hashable]] = []
    for state in order.states:
        signature = (in_labels[state], followers[state])
        if runs and (in_labels[runs[-1][-1]], followers[runs[-1][-1]]) == signature:
            runs[-1].append(state)
        else:
            runs.append([state])

    representative = {state: run[0] for run in runs for state in run}
    graph = TransitionGraph()
    for run in runs:
        graph.add_state(run[0])
    seen: set[tuple[Hashable, Any, Hashable]] = set()
    for transition in shift.transitions():
        symbol = transition.data.get(ATTR_SYMBOL)
        if symbol is None:
            continue
        edge = (representative[transition.source], symbol, representative[transition.target])
        if edge in seen:
            continue
        seen.add(edge)
        graph.add_transition(edge[0], edge[2], **{ATTR_SYMBOL: symbol})
    return WheelerCover(graph=graph, symbol_alphabet=shift.symbol_alphabet)


def wheeler_index_of_shift(shift: SoficShift, **kwargs: Any) -> Any:
    """Build a :class:`~sofic.automata.wheeler_index.WheelerIndex` over a Wheeler cover.

    Gives ``O(|w| log |A|)`` factor-language membership in place of scanning
    :meth:`~sofic.shifts.base.SymbolicModel.factor_language`.
    """
    from sofic.automata.wheeler_index import WheelerIndex

    return WheelerIndex.from_model(wheeler_cover(shift, **kwargs))
