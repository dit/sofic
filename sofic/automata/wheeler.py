"""Co-lexicographic (Wheeler) ordering of labeled state machines.

A labeled graph is *Wheeler* when its states admit a total order in which
states with no incoming edges come first and, for edges ``(u, v)`` labeled
``a`` and ``(u', v')`` labeled ``a'``, ``a < a'`` implies ``v < v'`` while
``a == a'`` and ``u < u'`` imply ``v <= v'`` :cite:`Gagie2017`. Equivalently,
the set of words reaching each state is an interval of the co-lexicographically
sorted prefixes of the recognized language :cite:`Alanko2020`.

Wheeler orders are the width-one case of the co-lexicographic partial orders of
:cite:`CotumaccioPrezza2021`; :func:`colex_width` measures how far an arbitrary
machine is from being Wheeler.
"""

from __future__ import annotations

import itertools
import math
from collections import defaultdict, deque
from collections.abc import Callable, Hashable, Iterable, Sequence
from dataclasses import dataclass
from functools import partial
from typing import Any

import networkx as nx

from sofic.exceptions import SoficValidationError
from sofic.graph import ATTR_EMISSION, ATTR_SYMBOL, EPSILON

#: Cap on the number of tie-breaking permutations tried by :func:`wheeler_order`.
MAX_TIE_PERMUTATIONS = 20_000

LabeledEdge = tuple[Hashable, Any, Hashable]


class WheelerError(SoficValidationError):
    """Raised when a Wheeler invariant is violated or cannot be established."""


@dataclass(frozen=True, slots=True)
class LabeledGraph:
    """Symbol-labeled view of a model, shared by every Wheeler routine.

    ``alphabet`` is ordered; ``edges`` are ``(source, symbol, target)`` triples.
    """

    states: tuple[Hashable, ...]
    alphabet: tuple[Any, ...]
    edges: tuple[LabeledEdge, ...]

    def symbol_rank(self) -> dict[Any, int]:
        return {symbol: index for index, symbol in enumerate(self.alphabet)}

    def in_labels(self) -> dict[Hashable, frozenset[Any]]:
        """Map each state to the set of labels on its incoming edges."""
        labels: dict[Hashable, set[Any]] = {state: set() for state in self.states}
        for _source, symbol, target in self.edges:
            labels[target].add(symbol)
        return {state: frozenset(symbols) for state, symbols in labels.items()}

    def predecessors(self) -> dict[Hashable, tuple[Hashable, ...]]:
        """Map each state to the sources of its incoming edges."""
        preds: dict[Hashable, list[Hashable]] = {state: [] for state in self.states}
        for source, _symbol, target in self.edges:
            preds[target].append(source)
        return {state: tuple(sources) for state, sources in preds.items()}


def labeled_graph(model: Any, *, symbol_key: Callable[[Any], Any] | None = None) -> LabeledGraph:
    """Extract the symbol-labeled graph of ``model``.

    Reads :data:`~sofic.graph.ATTR_SYMBOL` for automata and shift presentations
    and :data:`~sofic.graph.ATTR_EMISSION` for Mealy HMMs and epsilon-machines.
    Epsilon transitions have no Wheeler semantics and raise.
    """
    key = repr if symbol_key is None else symbol_key
    edges: list[LabeledEdge] = []
    symbols: set[Any] = set()
    for transition in model.transitions():
        symbol = transition.data.get(ATTR_SYMBOL, transition.data.get(ATTR_EMISSION))
        if symbol is EPSILON:
            raise WheelerError("epsilon transitions have no Wheeler order; remove them first")
        if symbol is None:
            continue
        edges.append((transition.source, symbol, transition.target))
        symbols.add(symbol)

    for attribute in ("input_alphabet", "symbol_alphabet", "observation_alphabet"):
        declared = getattr(model, attribute, None)
        if declared:
            symbols.update(declared)

    return LabeledGraph(
        states=tuple(model.states()),
        alphabet=tuple(sorted(symbols, key=key)),
        edges=tuple(edges),
    )


@dataclass(frozen=True, slots=True)
class WheelerOrder:
    """A total order on states witnessing the Wheeler property."""

    states: tuple[Hashable, ...]
    rank: dict[Hashable, int]

    @classmethod
    def from_sequence(cls, states: Sequence[Hashable]) -> WheelerOrder:
        ordered = tuple(states)
        return cls(states=ordered, rank={state: index for index, state in enumerate(ordered)})

    def __len__(self) -> int:
        return len(self.states)

    def __iter__(self) -> Any:
        return iter(self.states)

    def validate(self, graph: LabeledGraph) -> None:
        """Raise :class:`WheelerError` unless the axioms hold for ``graph``."""
        violation = _first_axiom_violation(graph, self.rank)
        if violation is not None:
            raise WheelerError(violation)


def is_input_consistent(model: Any, *, symbol_key: Callable[[Any], Any] | None = None) -> bool:
    """Return whether every state's incoming edges share a single label.

    A necessary condition for the Wheeler property: axiom one forces states
    with distinct incoming labels apart, so a state entered on two different
    symbols can never be placed in a total co-lex order. This rejects the even
    process, Nemo, and butterfly epsilon-machines outright.
    """
    graph = labeled_graph(model, symbol_key=symbol_key)
    return all(len(labels) <= 1 for labels in graph.in_labels().values())


def _first_axiom_violation(graph: LabeledGraph, rank: dict[Hashable, int]) -> str | None:
    """Return a description of the first Wheeler axiom violation, else ``None``."""
    symbol_rank = graph.symbol_rank()
    in_labels = graph.in_labels()

    sourceless = [rank[state] for state, labels in in_labels.items() if not labels]
    targeted = [rank[state] for state, labels in in_labels.items() if labels]
    if sourceless and targeted and max(sourceless) > min(targeted):
        return "a state with no incoming edges is ordered after a state with incoming edges"

    # Axiom 1: distinct incoming labels force the order, so every state entered
    # on symbol ``a`` must precede every state entered on a later symbol.
    by_symbol: dict[Any, list[int]] = defaultdict(list)
    for state, labels in in_labels.items():
        if len(labels) > 1:
            return f"state {state!r} is entered on more than one symbol: {sorted(labels, key=repr)}"
        for symbol in labels:
            by_symbol[symbol].append(rank[state])
    ordered_symbols = sorted(by_symbol, key=lambda symbol: symbol_rank[symbol])
    for earlier, later in zip(ordered_symbols, ordered_symbols[1:], strict=False):
        if max(by_symbol[earlier]) > min(by_symbol[later]):
            return f"states entered on {earlier!r} are not all before states entered on {later!r}"

    # Axiom 2: among equally labeled edges the target order follows the source
    # order, so sorting one symbol's edges by source rank must leave the target
    # ranks non-decreasing. Edges sharing a source are mutually unconstrained,
    # so they are compared as a group against the groups before them.
    state_at = {index: state for state, index in rank.items()}
    ranked: dict[Any, list[tuple[int, int]]] = defaultdict(list)
    for source, symbol, target in graph.edges:
        ranked[symbol].append((rank[source], rank[target]))
    for symbol, pairs in ranked.items():
        pairs.sort()
        ceiling, ceiling_source = -1, -1
        position = 0
        while position < len(pairs):
            source_rank = pairs[position][0]
            targets = []
            while position < len(pairs) and pairs[position][0] == source_rank:
                targets.append(pairs[position][1])
                position += 1
            if min(targets) < ceiling:
                return (
                    f"edges {state_at[ceiling_source]!r} -{symbol!r}-> {state_at[ceiling]!r} and "
                    f"{state_at[source_rank]!r} -{symbol!r}-> {state_at[min(targets)]!r} invert the order"
                )
            if max(targets) > ceiling:
                ceiling, ceiling_source = max(targets), source_rank
    return None


def check_wheeler_axioms(graph: LabeledGraph, order: Sequence[Hashable]) -> bool:
    """Return whether ``order`` satisfies the Wheeler axioms on ``graph``."""
    rank = {state: index for index, state in enumerate(order)}
    return _first_axiom_violation(graph, rank) is None


def _predecessor_range(
    state: Hashable,
    *,
    predecessors: dict[Hashable, tuple[Hashable, ...]],
    block_of: dict[Hashable, int],
) -> tuple[int, int]:
    """Lowest and highest block index among ``state``'s predecessors."""
    sources = [block_of[source] for source in predecessors[state]]
    return (min(sources), max(sources)) if sources else (-1, -1)


def _refine_colex_blocks(graph: LabeledGraph) -> list[list[Hashable]]:
    """Refine states into co-lex ordered blocks by predecessor block range.

    Seeds an ordered partition from the incoming label (axiom one) and then
    repeatedly splits each block by the lowest and highest block index among
    its members' predecessors. In a deterministic automaton two states sharing
    an incoming label must have *separated* predecessor sets -- ``u < v``
    forces every predecessor of ``u`` below every predecessor of ``v`` -- so
    both ends of that range move monotonically with the Wheeler order.
    """
    symbol_rank = graph.symbol_rank()
    in_labels = graph.in_labels()
    predecessors = graph.predecessors()

    def seed_key(state: Hashable) -> tuple[int, int]:
        labels = in_labels[state]
        if not labels:
            return (0, -1)
        return (1, min(symbol_rank[symbol] for symbol in labels))

    blocks: list[list[Hashable]] = []
    for _key, group in itertools.groupby(sorted(graph.states, key=seed_key), key=seed_key):
        blocks.append(list(group))

    for _round in range(len(graph.states) + 1):
        block_of = {state: index for index, block in enumerate(blocks) for state in block}
        split_key = partial(_predecessor_range, predecessors=predecessors, block_of=block_of)
        refined: list[list[Hashable]] = []
        for block in blocks:
            if len(block) == 1:
                refined.append(block)
                continue
            for _key, group in itertools.groupby(sorted(block, key=split_key), key=split_key):
                refined.append(list(group))
        if len(refined) == len(blocks):
            return refined
        blocks = refined
    return blocks


def wheeler_order(
    model: Any,
    *,
    symbol_key: Callable[[Any], Any] | None = None,
    max_tie_permutations: int = MAX_TIE_PERMUTATIONS,
) -> WheelerOrder | None:
    """Return a Wheeler order for ``model``, or ``None`` if it is not Wheeler."""
    return wheeler_order_of_graph(
        labeled_graph(model, symbol_key=symbol_key),
        max_tie_permutations=max_tie_permutations,
    )


def wheeler_order_of_graph(
    graph: LabeledGraph,
    *,
    max_tie_permutations: int = MAX_TIE_PERMUTATIONS,
) -> WheelerOrder | None:
    """Return a Wheeler order for ``graph``, or ``None`` if it is not Wheeler.

    Tries three increasingly expensive steps. Co-lex partition refinement
    usually pins the order outright. Otherwise the maximum co-lex relation is
    linearized, which also *disproves* Wheelerness whenever it leaves a pair
    incomparable, since it contains every co-lex order. Only states the
    relation ranks as mutually comparable are permuted, bounded by
    ``max_tie_permutations`` -- deciding Wheelerness is NP-complete for NFAs
    :cite:`GibneyThankachan2019`, so some input must stay expensive.
    """
    if not graph.states:
        return WheelerOrder.from_sequence(())
    if any(len(labels) > 1 for labels in graph.in_labels().values()):
        return None

    blocks = _refine_colex_blocks(graph)
    candidate = [state for block in blocks for state in block]
    if check_wheeler_axioms(graph, candidate):
        return WheelerOrder.from_sequence(candidate)

    if _is_deterministic(graph) and all(len(block) == 1 for block in blocks):
        # Separated predecessor sets make the refinement order the only
        # candidate for a deterministic graph, so a failure here is decisive
        # and the maximum co-lex relation has nothing left to contribute.
        return None

    chain = _linearize_colex_relation(graph)
    if chain is None:
        return None
    groups = [list(group) for group in chain]
    ordered = [state for group in groups for state in group]
    if check_wheeler_axioms(graph, ordered):
        return WheelerOrder.from_sequence(ordered)

    tied = [index for index, group in enumerate(groups) if len(group) > 1]
    total = math.prod(math.factorial(len(groups[index])) for index in tied)
    if total > max_tie_permutations:
        raise WheelerError(
            f"{total} tie-breaking permutations exceed max_tie_permutations="
            f"{max_tie_permutations}; Wheelerness is undecided for this model"
        )
    for arrangement in itertools.product(*(itertools.permutations(groups[index]) for index in tied)):
        replacement = dict(zip(tied, arrangement, strict=True))
        attempt: list[Hashable] = []
        for index, group in enumerate(groups):
            attempt.extend(replacement.get(index, tuple(group)))
        if check_wheeler_axioms(graph, attempt):
            return WheelerOrder.from_sequence(attempt)
    return None


def _linearize_colex_relation(graph: LabeledGraph) -> list[tuple[Hashable, ...]] | None:
    """Order the states into a chain of mutually comparable groups, or ``None``.

    ``None`` means some pair is incomparable in the maximum co-lex relation, so
    no co-lex order compares them and no *total* one exists: the graph is not
    Wheeler. This keeps the hard case polynomial instead of leaving unrelated
    states, sourceless ones above all, to a factorial search.
    """
    relation = maximum_colex_relation_of_graph(graph)
    strict = nx.DiGraph()
    strict.add_nodes_from(relation)
    for left, rights in relation.items():
        for right in rights:
            if left != right and left not in relation[right]:
                strict.add_edge(left, right)

    # A DAG's reachability order is total exactly when its topological order is
    # unique, i.e. Kahn's algorithm never has two ready nodes at once. That is
    # linear, where materializing the transitive closure is not.
    condensation = nx.condensation(strict)
    remaining = dict(condensation.in_degree())
    ready = [node for node, degree in remaining.items() if degree == 0]
    chain: list[int] = []
    while len(ready) == 1:
        node = ready.pop()
        chain.append(node)
        for successor in condensation.successors(node):
            remaining[successor] -= 1
            if remaining[successor] == 0:
                ready.append(successor)
    if len(chain) != condensation.number_of_nodes():
        return None
    return [tuple(condensation.nodes[node]["members"]) for node in chain]


def _is_deterministic(graph: LabeledGraph) -> bool:
    """Whether each state has at most one out-edge per symbol."""
    seen: dict[tuple[Hashable, Any], Hashable] = {}
    for source, symbol, target in graph.edges:
        key = (source, symbol)
        if seen.setdefault(key, target) != target:
            return False
    return True


def is_wheeler(model: Any, *, symbol_key: Callable[[Any], Any] | None = None) -> bool:
    """Return whether ``model`` admits a Wheeler order.

    A property of the presentation. Whether the *language* is Wheeler -- whether
    some equivalent automaton is -- is a strictly weaker and much more expensive
    question, decidable in ``O(mn)`` for a DFA :cite:`Becker2023` and
    PSPACE-complete for an NFA :cite:`DAgostino2023`. See
    :func:`~sofic.shifts.wheeler.wheeler_cover` and
    :func:`~sofic.generators.wheeler_epsilon.wheeler_presentation` for the
    search over presentations.
    """
    return wheeler_order(model, symbol_key=symbol_key) is not None


def maximum_colex_relation(
    model: Any,
    *,
    symbol_key: Callable[[Any], Any] | None = None,
) -> dict[Hashable, frozenset[Hashable]]:
    """Return the maximum co-lexicographic relation of ``model``.

    Maps each state ``u`` to the states ``v`` with ``u <= v``. Computed as the
    greatest fixpoint of the co-lex axioms of :cite:`CotumaccioPrezza2021`:
    start from every pair permitted by axiom one, then discard ``u <= v``
    whenever some pair of incoming edges makes axiom two fail. Unlike a minimum
    width co-lex *order*, whose computation is NP-hard for NFAs, this relation
    always exists and is computable in polynomial time :cite:`Cotumaccio2023`.
    """
    return maximum_colex_relation_of_graph(labeled_graph(model, symbol_key=symbol_key))


def maximum_colex_relation_of_graph(graph: LabeledGraph) -> dict[Hashable, frozenset[Hashable]]:
    """Greatest fixpoint of the co-lex axioms on ``graph``.

    Every co-lex order of ``graph`` is contained in this relation, since the
    fixpoint only ever discards pairs that violate an axiom. A pair left
    incomparable is therefore a pair *no* co-lex order can compare.
    """
    symbol_rank = graph.symbol_rank()
    in_labels = graph.in_labels()
    successors: dict[Hashable, set[Hashable]] = {state: set() for state in graph.states}
    for source, _symbol, target in graph.edges:
        successors[source].add(target)

    def label_precedes(left: Hashable, right: Hashable) -> bool:
        """Whether every incoming label of ``left`` is below every one of ``right``."""
        left_labels, right_labels = in_labels[left], in_labels[right]
        if not left_labels:
            return bool(right_labels)
        if not right_labels:
            return False
        return max(symbol_rank[s] for s in left_labels) < min(symbol_rank[s] for s in right_labels)

    relation: set[tuple[Hashable, Hashable]] = set()
    dropped: deque[tuple[Hashable, Hashable]] = deque()
    for left in graph.states:
        for right in graph.states:
            if left == right or not label_precedes(right, left):
                relation.add((left, right))
            else:
                dropped.append((left, right))

    # Axiom two fails for a pair exactly when one of its predecessor pairs has
    # already failed, so push removals forward along edges rather than
    # rescanning every pair on every round.
    while dropped:
        left_source, right_source = dropped.popleft()
        for left in successors[left_source]:
            for right in successors[right_source]:
                if left == right or (left, right) not in relation:
                    continue
                if in_labels[left] != in_labels[right]:
                    continue
                relation.discard((left, right))
                dropped.append((left, right))

    return {left: frozenset(right for right in graph.states if (left, right) in relation) for left in graph.states}


def colex_width(model: Any, *, symbol_key: Callable[[Any], Any] | None = None) -> int:
    """Return the co-lexicographic width of ``model``.

    The width is the size of the largest antichain of the partial order carried
    by the maximum co-lex relation, computed by Dilworth duality as the state
    count minus a maximum bipartite matching on the strict order. Width one
    means the order is total, i.e. the machine is Wheeler; larger widths bound
    the cost of indexing, encoding, and determinizing it
    :cite:`CotumaccioPrezza2021`.

    A co-lex order must be antisymmetric, so a pair the relation orients in
    both directions is a pair no co-lex order can compare and is counted here
    as incomparable. Minimum width over co-lex orders is NP-hard to compute for
    NFAs and the relation-based estimate can differ from it
    :cite:`Cotumaccio2023`.

    A Wheeler machine always has width one, but the converse needs
    :func:`is_input_consistent`: axiom one compares the incoming labels of two
    *distinct* states, so it cannot see a single state entered on two different
    symbols. The one-state presentation of the full shift is the smallest
    example -- width one, yet not Wheeler. Test Wheelerness with
    :func:`is_wheeler` rather than ``colex_width(...) == 1``.
    """
    relation = maximum_colex_relation(model, symbol_key=symbol_key)
    if not relation:
        return 0

    strict = nx.DiGraph()
    strict.add_nodes_from(relation)
    for left, rights in relation.items():
        for right in rights:
            if left != right and left not in relation[right]:
                strict.add_edge(left, right)

    # The strict part of a transitive relation is acyclic; condense defensively
    # so a non-transitive fixpoint still yields a DAG to close and match on.
    condensation = nx.condensation(strict)
    if condensation.number_of_nodes() <= 1:
        return condensation.number_of_nodes()

    closure = nx.transitive_closure_dag(condensation)
    bipartite = nx.Graph()
    bipartite.add_nodes_from((("out", node) for node in closure), bipartite=0)
    bipartite.add_nodes_from((("in", node) for node in closure), bipartite=1)
    bipartite.add_edges_from((("out", source), ("in", target)) for source, target in closure.edges)
    matching = nx.bipartite.hopcroft_karp_matching(bipartite, top_nodes=[("out", node) for node in closure])
    matched = sum(1 for node in matching if node[0] == "out")
    return closure.number_of_nodes() - matched


def _order_of(model: Any, symbol_key: Callable[[Any], Any] | None) -> tuple[LabeledGraph, WheelerOrder]:
    graph = labeled_graph(model, symbol_key=symbol_key)
    order = wheeler_order(model, symbol_key=symbol_key)
    if order is None:
        raise WheelerError(f"{type(model).__qualname__} does not admit a Wheeler order")
    return graph, order


def minimum_wdfa(
    dfa: Any,
    *,
    symbol_key: Callable[[Any], Any] | None = None,
) -> Any:
    """Return the minimum Wheeler DFA equivalent to the Wheeler DFA ``dfa``.

    Merges maximal runs of states that are consecutive in the Wheeler order,
    share an incoming label, and are Myhill-Nerode equivalent. Merging a range
    whose incoming edges carry one label preserves Wheelerness
    :cite:`Gagie2017`, and the resulting automaton is the unique smallest
    Wheeler DFA for the language :cite:`Alanko2020`.
    """
    from sofic.automata.algorithms import minimize
    from sofic.automata.dfa import DFA

    graph, order = _order_of(dfa, symbol_key)
    in_labels = graph.in_labels()
    classes = _nerode_classes(dfa, minimize(dfa))

    runs: list[list[Hashable]] = []
    for state in order.states:
        signature = (in_labels[state], classes[state])
        if runs and (in_labels[runs[-1][-1]], classes[runs[-1][-1]]) == signature:
            runs[-1].append(state)
        else:
            runs.append([state])

    merged: dict[Hashable, Hashable] = {}
    for run in runs:
        for state in run:
            merged[state] = run[0]

    result = DFA(
        input_alphabet=frozenset(dfa.input_alphabet),
        initial_states=frozenset(merged[state] for state in dfa.initial_states),
        accepting_states=frozenset(merged[state] for state in dfa.accepting_states),
    )
    for run in runs:
        result.graph.add_state(run[0])
    seen: set[tuple[Hashable, Any, Hashable]] = set()
    for source, symbol, target in graph.edges:
        edge = (merged[source], symbol, merged[target])
        if edge in seen:
            continue
        seen.add(edge)
        result.graph.add_transition(edge[0], edge[2], **{ATTR_SYMBOL: symbol})
    return result


def _nerode_classes(dfa: Any, minimal: Any) -> dict[Hashable, Hashable]:
    """Map each state of ``dfa`` to the ``minimal`` state it is equivalent to."""
    from sofic.graph import ATTR_SYMBOL as SYMBOL

    def step(automaton: Any, state: Hashable, symbol: Any) -> Hashable | None:
        for transition in automaton.graph.out_transitions(state):
            if transition.data.get(SYMBOL) == symbol:
                return transition.target
        return None

    start = next(iter(dfa.initial_states), None)
    minimal_start = next(iter(minimal.initial_states), None)
    if start is None or minimal_start is None:
        return dict.fromkeys(dfa.states(), 0)

    classes: dict[Hashable, Hashable] = {start: minimal_start}
    queue = [(start, minimal_start)]
    alphabet = sorted(dfa.input_alphabet, key=repr)
    while queue:
        state, image = queue.pop()
        for symbol in alphabet:
            target = step(dfa, state, symbol)
            if target is None or target in classes:
                continue
            classes[target] = step(minimal, image, symbol)
            queue.append((target, classes[target]))

    # States unreachable from the start form their own singleton classes.
    for state in dfa.states():
        classes.setdefault(state, ("unreachable", repr(state)))
    return classes


def wnfa_to_wdfa(
    nfa: Any,
    *,
    symbol_key: Callable[[Any], Any] | None = None,
) -> Any:
    """Determinize a Wheeler NFA into an equivalent Wheeler DFA.

    Path coherence makes every reachable subset an interval of the Wheeler
    order, so the subset construction ranges over intervals rather than subsets
    and yields at most ``2n - 1 - |Sigma|`` states :cite:`Alanko2020`.
    """
    from sofic.automata.dfa import DFA

    graph, order = _order_of(nfa, symbol_key)
    rank = order.rank
    by_symbol: dict[Any, list[tuple[int, int]]] = defaultdict(list)
    for source, symbol, target in graph.edges:
        by_symbol[symbol].append((rank[source], rank[target]))

    def successor(interval: tuple[int, int], symbol: Any) -> tuple[int, int] | None:
        targets = [target for source, target in by_symbol[symbol] if interval[0] <= source <= interval[1]]
        if not targets:
            return None
        return (min(targets), max(targets))

    initial_ranks = [rank[state] for state in nfa.initial_states]
    if not initial_ranks:
        return DFA(input_alphabet=frozenset(nfa.input_alphabet))
    start = (min(initial_ranks), max(initial_ranks))

    intervals = {start}
    queue = [start]
    edges: list[tuple[tuple[int, int], Any, tuple[int, int]]] = []
    while queue:
        interval = queue.pop()
        for symbol in graph.alphabet:
            target = successor(interval, symbol)
            if target is None:
                continue
            edges.append((interval, symbol, target))
            if target not in intervals:
                intervals.add(target)
                queue.append(target)

    accepting_ranks = {rank[state] for state in nfa.accepting_states}

    def label(interval: tuple[int, int]) -> Hashable:
        return tuple(order.states[index] for index in range(interval[0], interval[1] + 1))

    result = DFA(
        input_alphabet=frozenset(nfa.input_alphabet),
        initial_states=frozenset({label(start)}),
        accepting_states=frozenset(
            label(interval)
            for interval in intervals
            if any(index in accepting_ranks for index in range(interval[0], interval[1] + 1))
        ),
    )
    for interval in intervals:
        result.graph.add_state(label(interval))
    for source, symbol, target in edges:
        result.graph.add_transition(label(source), label(target), **{ATTR_SYMBOL: symbol})
    return result


def wheeler_canonical_form(model: Any, *, symbol_key: Callable[[Any], Any] | None = None) -> tuple[Any, ...]:
    """Return a canonical fingerprint of a Wheeler model.

    The Wheeler order is intrinsic, so serializing the machine in that order
    gives a representation independent of how its states happen to be named or
    inserted. Two Wheeler models share a fingerprint exactly when they are
    isomorphic as labeled graphs.
    """
    graph, order = _order_of(model, symbol_key)
    rank = order.rank
    symbol_rank = graph.symbol_rank()
    transitions = sorted((rank[source], symbol_rank[symbol], rank[target]) for source, symbol, target in graph.edges)
    marked = _marked_states(model, rank)
    return (len(order), len(graph.alphabet), tuple(transitions), marked)


def wheeler_isomorphic(
    left: Any,
    right: Any,
    *,
    symbol_key: Callable[[Any], Any] | None = None,
) -> bool:
    """Return whether two Wheeler models are isomorphic as labeled graphs.

    Because the Wheeler order is intrinsic, this compares canonical forms in
    time linear in the transition count, where
    :func:`~sofic.automata.algorithms.equivalent` has to minimize both inputs.
    It is strictly finer than language equivalence: two Wheeler DFAs for the
    same language differ here unless both are minimal. Raises
    :class:`WheelerError` if either model lacks a Wheeler order.
    """
    return wheeler_canonical_form(left, symbol_key=symbol_key) == wheeler_canonical_form(right, symbol_key=symbol_key)


def wheeler_state_index(model: Any, *, symbol_key: Callable[[Any], Any] | None = None) -> Any:
    """Return a :class:`~sofic.indexing.StateIndex` in Wheeler order.

    Indexing states this way makes derived matrices canonical: the same machine
    yields the same transition matrix no matter how its states were named or
    inserted, unlike the insertion order of :meth:`~sofic.base.StateMachine.reindex`.
    """
    from sofic.indexing import StateIndex

    _graph, order = _order_of(model, symbol_key)
    return StateIndex(order.states)


def _marked_states(model: Any, rank: dict[Hashable, int]) -> tuple[tuple[int, ...], ...]:
    """Ranks of initial and accepting states, when the model distinguishes them."""
    marks: list[tuple[int, ...]] = []
    for attribute in ("initial_states", "accepting_states"):
        states: Iterable[Hashable] = getattr(model, attribute, ()) or ()
        marks.append(tuple(sorted(rank[state] for state in states if state in rank)))
    return tuple(marks)
