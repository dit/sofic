"""Burrows-Wheeler index over a Wheeler-ordered machine.

Stores a Wheeler graph as the arrays of :cite:`Gagie2017`: out-degrees and
in-degrees in node order, plus the edge labels listed in node order. Because
the Wheeler axioms make the edges sorted by ``(label, source)`` coincide with
the edges sorted by target, following a label maps one node interval onto
another -- the generalization of FM-index backward search to labeled graphs.

Rank is served by binary search over per-symbol position arrays rather than a
succinct bitvector, which costs a logarithmic factor but keeps the dependency
surface at numpy.
"""

from __future__ import annotations

import random
from collections.abc import Callable, Hashable, Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from sofic.automata.wheeler import LabeledGraph, WheelerError, WheelerOrder, labeled_graph, wheeler_order_of_graph

Interval = tuple[int, int]


@dataclass(frozen=True, slots=True, eq=False)
class WheelerIndex:
    """Searchable Burrows-Wheeler representation of a Wheeler machine."""

    order: WheelerOrder
    alphabet: tuple[Any, ...]
    #: Prefix sums of out-degrees in Wheeler order; ``out_start[-1]`` is the edge count.
    out_start: np.ndarray
    #: Prefix sums of in-degrees in Wheeler order.
    in_start: np.ndarray
    #: Symbol index of each out-edge, listed in node order then symbol order.
    labels: np.ndarray
    #: Number of edges whose symbol precedes each symbol index.
    symbol_start: np.ndarray
    #: Positions in :attr:`labels` carrying each symbol index.
    label_positions: tuple[np.ndarray, ...]
    #: Maximal runs of consecutive initial ranks, in Wheeler order.
    initial: tuple[Interval, ...]
    accepting: frozenset[int]
    #: Reverse adjacency, used for co-lex word ranking.
    _backward: dict[tuple[Hashable, Any], frozenset[Hashable]]
    _symbol_index: dict[Any, int]
    _word_counts: dict[tuple[frozenset[Hashable], int], int] = field(default_factory=dict)

    @classmethod
    def from_model(
        cls,
        model: Any,
        *,
        symbol_key: Callable[[Any], Any] | None = None,
        initial_states: frozenset[Hashable] | None = None,
        accepting_states: frozenset[Hashable] | None = None,
    ) -> WheelerIndex:
        """Build an index for ``model``.

        Initial and accepting states default to the model's own. Presentations
        that do not mark them (shifts, or other models with empty marks) treat
        every state as both -- the "all states initial" case of
        :cite:`Gagie2017` Theorem 6, which is how a shift presents its factor
        language. A finite automaton's empty initial or accepting set is taken
        literally and yields an index of the empty language.
        """
        from sofic.automata.base import LabeledAutomaton

        graph = labeled_graph(model, symbol_key=symbol_key)
        order = wheeler_order_of_graph(graph)
        if order is None:
            raise WheelerError(f"{type(model).__qualname__} does not admit a Wheeler order")

        every = frozenset(graph.states)
        literal = isinstance(model, LabeledAutomaton)
        if initial_states is None:
            initial_states = model.initial_states if literal else getattr(model, "initial_states", None) or every
        if accepting_states is None:
            accepting_states = model.accepting_states if literal else getattr(model, "accepting_states", None) or every
        return cls._build(graph, order, initial_states, accepting_states)

    @classmethod
    def _build(
        cls,
        graph: LabeledGraph,
        order: WheelerOrder,
        initial_states: frozenset[Hashable],
        accepting_states: frozenset[Hashable],
    ) -> WheelerIndex:
        rank = order.rank
        symbol_rank = graph.symbol_rank()
        size = len(order.states)
        arity = len(graph.alphabet)

        # Edges in node order carry the labels; edges in (label, source) order
        # are the same multiset listed by target, which is what makes the
        # label-to-target hop a pair of array lookups.
        by_source = sorted((rank[source], symbol_rank[symbol], rank[target]) for source, symbol, target in graph.edges)
        out_degree = np.zeros(size, dtype=np.int64)
        in_degree = np.zeros(size, dtype=np.int64)
        for source, _symbol, target in by_source:
            out_degree[source] += 1
            in_degree[target] += 1

        labels = np.fromiter((symbol for _source, symbol, _target in by_source), dtype=np.int64, count=len(by_source))
        symbol_count = np.zeros(arity + 1, dtype=np.int64)
        for symbol in labels:
            symbol_count[symbol + 1] += 1

        backward: dict[tuple[Hashable, Any], set[Hashable]] = {}
        for source, symbol, target in graph.edges:
            backward.setdefault((target, symbol), set()).add(source)

        return cls(
            order=order,
            alphabet=graph.alphabet,
            out_start=np.concatenate(([0], np.cumsum(out_degree))),
            in_start=np.concatenate(([0], np.cumsum(in_degree))),
            labels=labels,
            symbol_start=np.cumsum(symbol_count)[:-1],
            label_positions=tuple(np.flatnonzero(labels == symbol) for symbol in range(arity)),
            initial=_runs_of(initial_states, rank),
            accepting=frozenset(rank[state] for state in accepting_states),
            _backward={key: frozenset(value) for key, value in backward.items()},
            _symbol_index=dict(symbol_rank),
        )

    def __len__(self) -> int:
        return len(self.order.states)

    def bits(self) -> int:
        """Size of the Burrows-Wheeler encoding in bits.

        The ``2(e + n) + e log |A|`` bound of :cite:`Gagie2017` Theorem 6,
        excluding the lower-order rank structures.
        """
        edges, nodes = int(self.labels.size), len(self)
        arity = max(len(self.alphabet), 1)
        return 2 * (edges + nodes) + int(np.ceil(edges * np.log2(arity)))

    def step(self, interval: Interval, symbol: Any) -> Interval | None:
        """Follow every ``symbol`` edge out of ``interval``, or ``None`` if there are none."""
        index = self._symbol_index.get(symbol)
        if index is None:
            return None
        low = int(self.out_start[interval[0]])
        high = int(self.out_start[interval[1] + 1])
        positions = self.label_positions[index]
        first = int(np.searchsorted(positions, low))
        last = int(np.searchsorted(positions, high))
        if first == last:
            return None
        base = int(self.symbol_start[index])
        return (self._node_of(base + first), self._node_of(base + last - 1))

    def _node_of(self, in_edge: int) -> int:
        """Node owning the given position in the by-target edge ordering."""
        return int(np.searchsorted(self.in_start, in_edge, side="right")) - 1

    def forward_search(self, word: Sequence[Any], interval: Interval | None = None) -> Interval | None:
        """Return the node interval reached by reading ``word``, or ``None``.

        Runs in ``O(|word| log |A|)`` regardless of how many states the word
        actually reaches, because path coherence keeps the image of an interval
        an interval :cite:`Gagie2017`. Starting from the initial states (the
        default) requires them to be consecutive in the Wheeler order whenever
        the reached set is not; use :meth:`states_reached` in that case.
        """
        runs = self._search_runs(word, (interval,) if interval is not None else self.initial)
        if not runs:
            return None
        if len(runs) > 1:
            raise WheelerError("the reached states are not an interval of the Wheeler order; use states_reached")
        return runs[0]

    def _search_runs(self, word: Sequence[Any], start: tuple[Interval, ...] | None = None) -> tuple[Interval, ...]:
        """Disjoint sorted node intervals reached by reading ``word`` from ``start``."""
        current = self.initial if start is None else start
        for symbol in word:
            if not current:
                break
            current = _merge_runs(image for run in current if (image := self.step(run, symbol)) is not None)
        return current

    def _nodes_reached(self, word: Sequence[Any]) -> Iterator[int]:
        for low, high in self._search_runs(word):
            yield from range(low, high + 1)

    def contains(self, word: Sequence[Any]) -> bool:
        """Return whether ``word`` is accepted.

        The indexed replacement for scanning
        :meth:`~sofic.shifts.base.SymbolicModel.factor_language`.
        """
        return any(node in self.accepting for node in self._nodes_reached(word))

    def count_states(self, word: Sequence[Any]) -> int:
        """Number of states reachable by reading ``word``."""
        return sum(high - low + 1 for low, high in self._search_runs(word))

    def states_reached(self, word: Sequence[Any]) -> tuple[Hashable, ...]:
        """The states reachable by reading ``word``, in Wheeler order."""
        return tuple(self.order.states[node] for node in self._nodes_reached(word))

    # -- Co-lexicographic word ranking -------------------------------------
    #
    # Co-lex compares words from the last symbol backwards, which is exactly
    # the order the Wheeler order imposes on states via the words reaching
    # them. Counting therefore walks the transition graph backwards, subset
    # determinizing as it goes so that each branch is a distinct word.

    def _predecessors(self, states: frozenset[Hashable], symbol: Any) -> frozenset[Hashable]:
        sources: set[Hashable] = set()
        for state in states:
            sources.update(self._backward.get((state, symbol), ()))
        return frozenset(sources)

    def _count_from(self, states: frozenset[Hashable], length: int) -> int:
        """Distinct words of ``length`` symbols leading from an initial state into ``states``."""
        if length == 0:
            return 1 if states & self._initial_states() else 0
        cached = self._word_counts.get((states, length))
        if cached is not None:
            return cached
        total = sum(
            self._count_from(sources, length - 1)
            for symbol in self.alphabet
            if (sources := self._predecessors(states, symbol))
        )
        self._word_counts[(states, length)] = total
        return total

    def _initial_states(self) -> frozenset[Hashable]:
        return frozenset(self.order.states[node] for low, high in self.initial for node in range(low, high + 1))

    def _accepting_states(self) -> frozenset[Hashable]:
        return frozenset(self.order.states[node] for node in sorted(self.accepting))

    def count_words(self, length: int) -> int:
        """Number of distinct accepted words of ``length`` symbols."""
        return self._count_from(self._accepting_states(), length)

    def words_of_length(self, length: int) -> Iterator[tuple[Any, ...]]:
        """Yield the accepted words of ``length`` symbols in co-lexicographic order."""
        for index in range(self.count_words(length)):
            yield self.unrank_word(index, length)

    def rank_word(self, word: Sequence[Any]) -> int:
        """Position of ``word`` among equally long accepted words, co-lex ordered."""
        states = self._accepting_states()
        position = 0
        for depth, symbol in enumerate(reversed(tuple(word))):
            remaining = len(word) - depth - 1
            for candidate in self.alphabet:
                sources = self._predecessors(states, candidate)
                if candidate == symbol:
                    if not sources:
                        raise ValueError(f"{tuple(word)!r} is not an accepted word")
                    states = sources
                    break
                position += self._count_from(sources, remaining)
            else:
                raise ValueError(f"symbol {symbol!r} is not in the alphabet")
        if not states & self._initial_states():
            raise ValueError(f"{tuple(word)!r} is not an accepted word")
        return position

    def unrank_word(self, index: int, length: int) -> tuple[Any, ...]:
        """Inverse of :meth:`rank_word`: the ``index``-th co-lex accepted word."""
        total = self.count_words(length)
        if not 0 <= index < total:
            raise IndexError(f"rank {index} out of range for {total} words of length {length}")
        states = self._accepting_states()
        suffix: list[Any] = []
        remaining = index
        for depth in range(length):
            for candidate in self.alphabet:
                sources = self._predecessors(states, candidate)
                block = self._count_from(sources, length - depth - 1)
                if remaining < block:
                    suffix.append(candidate)
                    states = sources
                    break
                remaining -= block
        return tuple(reversed(suffix))

    def sample_word(self, length: int, rng: random.Random | None = None) -> tuple[Any, ...]:
        """Sample uniformly from the accepted words of ``length`` symbols."""
        total = self.count_words(length)
        if total == 0:
            raise ValueError(f"no accepted words of length {length}")
        chooser = random.Random() if rng is None else rng
        return self.unrank_word(chooser.randrange(total), length)


def _runs_of(states: frozenset[Hashable], rank: dict[Hashable, int]) -> tuple[Interval, ...]:
    return _merge_runs((rank[state], rank[state]) for state in states if state in rank)


def _merge_runs(intervals: Iterable[Interval]) -> tuple[Interval, ...]:
    """Union of node intervals as sorted, disjoint, non-adjacent runs."""
    merged: list[list[int]] = []
    for low, high in sorted(intervals):
        if merged and low <= merged[-1][1] + 1:
            merged[-1][1] = max(merged[-1][1], high)
        else:
            merged.append([low, high])
    return tuple((low, high) for low, high in merged)


def wheeler_index(model: Any, **kwargs: Any) -> WheelerIndex:
    """Build a :class:`WheelerIndex` for ``model``."""
    return WheelerIndex.from_model(model, **kwargs)
