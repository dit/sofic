"""Fischer and Krieger cover constructions.

Both covers are computed exactly from a presentation ``G`` with vertex set
``Q``. Write ``S . w`` for the set of vertices reached from ``S`` along paths
labeled ``w``, and ``F(S)`` for the follower set (future language) of ``S``.

* The **right Fischer cover** of an irreducible sofic shift is its unique
  minimal right-resolving presentation :cite:`Fischer1975`
  :cite:`LindMarcus1995`. It is the unique terminal strongly connected
  component of the subset construction from ``Q`` after merging subsets with
  equal follower sets: an intrinsically synchronizing word ``m`` sends every
  subset to the follower class ``F(m)``, so that class is reachable from all
  others.
* The **right Krieger cover** (future cover) has one vertex per follower set
  ``F(x^-)`` of a left-infinite ray, with ``F(x^-) --a--> F(x^- a)``
  :cite:`Krieger1984` :cite:`LindMarcus1995`. ``F(x^-) = F(T(x^-))`` where
  ``T(x^-) = Q . s`` for every long enough suffix ``s`` of ``x^-``. Reading a
  ray right to left composes path relations ``rho_{cs} = rho_c o rho_s`` in a
  finite monoid, so the sets ``T(x^-)`` are exactly the images ``Q . rho`` of
  relations ``rho`` that lie on a cycle reachable from the identity.

The left covers are the mirror images: the right cover of the reversed shift,
reversed back.
"""

from __future__ import annotations

from collections import defaultdict, deque
from collections.abc import Hashable, Iterable
from typing import Any

import networkx as nx

from sofic.exceptions import SoficValidationError
from sofic.graph import ATTR_SYMBOL, TransitionGraph
from sofic.shifts.covers import LeftFischerCover, LeftKriegerCover, RightFischerCover, RightKriegerCover
from sofic.shifts.sofic import SoficShift
from sofic.states import sequential_labels

Subset = frozenset[Hashable]
Relation = frozenset[tuple[Hashable, Hashable]]


def _labeled_successors(shift: SoficShift) -> dict[Hashable, dict[Any, set[Hashable]]]:
    successors: dict[Hashable, dict[Any, set[Hashable]]] = {state: defaultdict(set) for state in shift.states()}
    for transition in shift.transitions():
        symbol = transition.data.get(ATTR_SYMBOL)
        if symbol is not None:
            successors[transition.source][symbol].add(transition.target)
    return successors


def _subset_automaton(
    successors: dict[Hashable, dict[Any, set[Hashable]]],
) -> dict[Subset, dict[Any, Subset]]:
    """Deterministic subset automaton reachable from the full vertex set."""
    start: Subset = frozenset(successors)
    delta: dict[Subset, dict[Any, Subset]] = {}
    queue = deque([start])
    while queue:
        subset = queue.popleft()
        if subset in delta:
            continue
        moves: dict[Any, set[Hashable]] = defaultdict(set)
        for state in subset:
            for symbol, targets in successors[state].items():
                moves[symbol] |= targets
        delta[subset] = {symbol: frozenset(targets) for symbol, targets in moves.items() if targets}
        queue.extend(target for target in delta[subset].values() if target not in delta)
    return delta


def _follower_classes(delta: dict[Subset, dict[Any, Subset]]) -> dict[Subset, int]:
    """Moore refinement: subsets with equal follower sets share a class."""
    block = dict.fromkeys(delta, 0)
    while True:
        signatures = {
            subset: (block[subset], tuple(sorted(((repr(a), block[t]) for a, t in moves.items()))))
            for subset, moves in delta.items()
        }
        ids: dict[Any, int] = {}
        refined = {subset: ids.setdefault(signature, len(ids)) for subset, signature in signatures.items()}
        if len(ids) == len(set(block.values())):
            return refined
        block = refined


def _labels(count: int) -> tuple[Hashable, ...]:
    return sequential_labels(count) if count <= 26 else tuple(range(count))


def _quotient_shift(
    cls: type[SoficShift],
    shift: SoficShift,
    vertices: Iterable[Subset],
    delta: dict[Subset, dict[Any, Subset]],
    classes: dict[Subset, int],
) -> SoficShift:
    keep = set(vertices)
    used = sorted({classes[subset] for subset in keep})
    name = dict(zip(used, _labels(len(used)), strict=True))
    graph = TransitionGraph()
    for class_id in used:
        graph.add_state(name[class_id])
    edges = {
        (classes[subset], symbol, classes[target])
        for subset in keep
        for symbol, target in delta[subset].items()
        if target in keep
    }
    for source, symbol, target in sorted(edges, key=repr):
        graph.add_transition(name[source], name[target], **{ATTR_SYMBOL: symbol})
    return cls(graph=graph, symbol_alphabet=shift.symbol_alphabet)


def _mirror(cls: type[SoficShift], cover: SoficShift) -> SoficShift:
    return cls(graph=cover.graph.reverse(), symbol_alphabet=cover.symbol_alphabet)


def right_fischer_cover(shift: SoficShift) -> RightFischerCover:
    """Return the minimal right-resolving presentation of an irreducible sofic shift.

    Raises :class:`~sofic.exceptions.SoficValidationError` when the shift is
    reducible (more than one terminal component), in which case a minimal
    right-resolving presentation need not be unique :cite:`LindMarcus1995`.
    """
    trimmed = shift.trim_transient()
    delta = _subset_automaton(_labeled_successors(trimmed))
    if not delta or not any(delta.values()):
        return RightFischerCover(symbol_alphabet=shift.symbol_alphabet)
    classes = _follower_classes(delta)

    quotient = nx.DiGraph()
    quotient.add_nodes_from(set(classes.values()))
    for subset, moves in delta.items():
        for target in moves.values():
            quotient.add_edge(classes[subset], classes[target])
    condensation = nx.condensation(quotient)
    terminal = [node for node in condensation.nodes if condensation.out_degree(node) == 0]
    if len(terminal) != 1:
        raise SoficValidationError("the Fischer cover is defined for irreducible sofic shifts; this shift is reducible")
    members = set(condensation.nodes[terminal[0]]["members"])
    if not any(classes[s] in members and delta[s] for s in delta):
        return RightFischerCover(symbol_alphabet=shift.symbol_alphabet)
    if not _presents_every_word(frozenset(trimmed.states()), delta, classes, members):
        raise SoficValidationError("the Fischer cover is defined for irreducible sofic shifts; this shift is reducible")
    vertices = [subset for subset in delta if classes[subset] in members]
    return _quotient_shift(RightFischerCover, shift, vertices, delta, classes)


def _presents_every_word(
    start: Subset,
    delta: dict[Subset, dict[Any, Subset]],
    classes: dict[Subset, int],
    members: set[int],
) -> bool:
    """Whether the terminal component reads every word the whole presentation reads.

    A unique terminal component does not make the shift irreducible (``0^inf``,
    ``1^inf`` and ``0...01...1`` have one), but the shift is irreducible exactly
    when that component presents all of it. Both sides are deterministic, so
    walk their product from (all vertices, all component classes).
    """
    moves: dict[int, dict[Any, int]] = defaultdict(dict)
    for subset, edges in delta.items():
        if classes[subset] in members:
            for symbol, target in edges.items():
                moves[classes[subset]][symbol] = classes[target]
    initial = (start, frozenset(members))
    seen = {initial}
    queue = deque([initial])
    while queue:
        subset, component = queue.popleft()
        for symbol, target in delta[subset].items():
            following = frozenset(moves[c][symbol] for c in component if symbol in moves[c])
            if not following:
                return False
            pair = (target, following)
            if pair not in seen:
                seen.add(pair)
                queue.append(pair)
    return True


def left_fischer_cover(shift: SoficShift) -> LeftFischerCover:
    """Return the minimal left-resolving presentation (mirror of the right Fischer cover)."""
    return _mirror(LeftFischerCover, right_fischer_cover(shift.reverse()))


def _ray_terminal_sets(successors: dict[Hashable, dict[Any, set[Hashable]]]) -> set[Subset]:
    """Return ``{T(x^-)}``: images of path relations lying on reachable cycles."""
    symbols = {symbol for moves in successors.values() for symbol in moves}
    letter: dict[Any, Relation] = {
        symbol: frozenset((p, q) for p, moves in successors.items() for q in moves.get(symbol, ()))
        for symbol in symbols
    }
    identity: Relation = frozenset((q, q) for q in successors)

    def prepend(symbol: Any, relation: Relation) -> Relation:
        after: dict[Hashable, set[Hashable]] = defaultdict(set)
        for q, r in relation:
            after[q].add(r)
        return frozenset((p, r) for p, q in letter[symbol] for r in after.get(q, ()))

    graph = nx.DiGraph()
    graph.add_node(identity)
    queue = deque([identity])
    while queue:
        relation = queue.popleft()
        for symbol in symbols:
            extended = prepend(symbol, relation)
            if not extended:
                continue
            if extended not in graph:
                queue.append(extended)
            graph.add_edge(relation, extended)

    recurrent: set[Relation] = set()
    for component in nx.strongly_connected_components(graph):
        node = next(iter(component))
        if len(component) > 1 or graph.has_edge(node, node):
            recurrent |= component
    return {frozenset(r for _q, r in relation) for relation in recurrent}


def right_krieger_cover(shift: SoficShift) -> RightKriegerCover:
    """Return the right Krieger (future) cover of ``shift``."""
    trimmed = shift.trim_transient()
    successors = _labeled_successors(trimmed)
    if not successors:
        return RightKriegerCover(symbol_alphabet=shift.symbol_alphabet)
    delta = _subset_automaton(successors)
    classes = _follower_classes(delta)
    vertices = [subset for subset in _ray_terminal_sets(successors) if subset in delta]
    return _quotient_shift(RightKriegerCover, shift, vertices, delta, classes)


def left_krieger_cover(shift: SoficShift) -> LeftKriegerCover:
    """Return the left Krieger (past) cover: the mirror of the right Krieger cover."""
    return _mirror(LeftKriegerCover, right_krieger_cover(shift.reverse()))
