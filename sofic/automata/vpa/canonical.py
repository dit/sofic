"""Canonical visibly pushdown automata from the well-matched summary algebra."""

from __future__ import annotations

from collections import deque
from collections.abc import Hashable, Mapping, Sequence
from typing import Any

from sofic.automata.vpa.base import VisiblyPushdownAutomaton
from sofic.automata.vpa.deterministic import DeterministicVisiblyPushdownAutomaton
from sofic.exceptions import NonWellMatchedLanguageError


class CanonicalVisiblyPushdownAutomaton(DeterministicVisiblyPushdownAutomaton):
    """Canonical VPA built from the finite Myhill-Nerode summary algebra."""

    summary_representatives: dict[Hashable, tuple[int | None, ...]]

    def __init__(
        self,
        *,
        summary_representatives: Mapping[Hashable, tuple[int | None, ...]] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.summary_representatives = dict(summary_representatives or {})

    @classmethod
    def from_vpa(cls, vpa: VisiblyPushdownAutomaton) -> CanonicalVisiblyPushdownAutomaton:
        """Build the Myhill-Nerode canonical deterministic VPA of a well-matched language.

        States are classes of the summary algebra of well-matched factors, so
        the form is canonical for well-matched languages. Raises
        :class:`~sofic.exceptions.NonWellMatchedLanguageError` when ``vpa``
        accepts a word with a pending call or return; general VPLs have no
        unique minimal deterministic VPA :cite:`AlurKumarMadhusudanViswanathan2005`.
        With an empty call alphabet this is the minimal DFA.
        """
        if vpa.has_unmatched_word():
            raise NonWellMatchedLanguageError(
                "the canonical VPA is defined for well-matched languages; this one accepts a pending call or return"
            )
        det = DeterministicVisiblyPushdownAutomaton.from_vpa(vpa)
        return _SummaryAlgebra(det).to_canonical_vpa(cls)

    @classmethod
    def minimize(cls, vpa: VisiblyPushdownAutomaton) -> CanonicalVisiblyPushdownAutomaton:
        """Alias for :meth:`from_vpa`."""
        return cls.from_vpa(vpa)


class _SummaryAlgebra:
    """Finite algebra of well-matched summaries with top-level and nested congruences.

    A summary maps each source state to the state reached along a well-matched
    word (``None`` when the run dies). Top-level summaries are identified when
    every well-matched continuation accepts both or neither; nested summaries
    (inside a pending call) are identified when, for every enclosing context,
    returning from them lands in the same class. Both partitions are refined
    together until internal steps, calls, and returns are well defined on
    classes, which makes the quotient canonical.
    """

    def __init__(self, vpa: DeterministicVisiblyPushdownAutomaton) -> None:
        self.vpa = vpa
        self.state_order = tuple(sorted(vpa.states(), key=repr))
        state_index = {state: index for index, state in enumerate(self.state_order)}
        self.internal_summaries = {
            symbol: _internal_summary(vpa, self.state_order, state_index, symbol)
            for symbol in sorted(vpa.internal_alphabet, key=repr)
        }
        self.identity = tuple(range(len(self.state_order)))
        self.summaries = sorted(
            _close_summary_algebra(vpa, self.state_order, state_index, self.identity, self.internal_summaries),
            key=repr,
        )
        self.calls = sorted(vpa.call_alphabet, key=repr)
        self.returns = sorted(vpa.return_alphabet, key=repr)
        self._wrap_cache: dict[tuple, tuple] = {}
        self.top, self.nested = self._refine()

    def wrap(self, inner: tuple, call: Any, ret: Any) -> tuple:
        key = (inner, call, ret)
        if key not in self._wrap_cache:
            self._wrap_cache[key] = _wrap_summary(self.vpa, self.state_order, inner, call, ret)
        return self._wrap_cache[key]

    def accepts(self, summary: tuple) -> bool:
        return _summary_accepts(self.vpa, self.state_order, summary)

    def _refine(self) -> tuple[dict[tuple, int], dict[tuple, int]]:
        top = {s: int(self.accepts(s)) for s in self.summaries}
        nested = dict.fromkeys(self.summaries, 0)
        internals = list(self.internal_summaries.values())
        pairs = [(c, r) for c in self.calls for r in self.returns]
        while True:
            top_signature = {
                s: (
                    top[s],
                    tuple(top[_compose_summary(s, a)] for a in internals),
                    tuple(top[_compose_summary(s, self.wrap(x, c, r))] for x in self.summaries for c, r in pairs),
                )
                for s in self.summaries
            }
            nested_signature = {
                s: (
                    nested[s],
                    tuple(nested[_compose_summary(s, a)] for a in internals),
                    tuple(nested[_compose_summary(s, self.wrap(x, c, r))] for x in self.summaries for c, r in pairs),
                    tuple(
                        (top[_compose_summary(o, self.wrap(s, c, r))], nested[_compose_summary(o, self.wrap(s, c, r))])
                        for o in self.summaries
                        for c, r in pairs
                    ),
                )
                for s in self.summaries
            }
            new_top = _number_blocks(top_signature, first=top_signature[self.identity])
            new_nested = _number_blocks(nested_signature, first=nested_signature[self.identity])
            stable = len(set(new_top.values())) == len(set(top.values())) and len(set(new_nested.values())) == len(
                set(nested.values())
            )
            top, nested = new_top, new_nested
            if stable:
                return top, nested

    def to_canonical_vpa(self, cls: type[CanonicalVisiblyPushdownAutomaton]) -> CanonicalVisiblyPushdownAutomaton:
        representative: dict[Hashable, tuple] = {}
        for s in self.summaries:
            representative.setdefault(self.top[s], s)
            representative.setdefault(("nested", self.nested[s]), s)

        def label(summary: tuple, nested: bool) -> Hashable:
            return ("nested", self.nested[summary]) if nested else self.top[summary]

        initial = label(self.identity, False)
        entry = label(self.identity, True)
        reachable: set[Hashable] = set()
        internals, calls, returns = set(), set(), set()
        pushed: set[tuple[Hashable, Any]] = set()
        frontier = [initial]
        while frontier:
            while frontier:
                state = frontier.pop()
                if state in reachable:
                    continue
                reachable.add(state)
                summary = representative[state]
                for symbol, step in self.internal_summaries.items():
                    target = label(_compose_summary(summary, step), isinstance(state, tuple))
                    internals.add((state, symbol, target))
                    frontier.append(target)
                for call in self.calls:
                    calls.add((state, call, entry, (state, call)))
                    pushed.add((state, call))
                    frontier.append(entry)
            # Returns pop a pushed (outer, call) from a nested state; both sets
            # grow together, so repeat until no new state appears.
            for state in [s for s in reachable if isinstance(s, tuple)]:
                for outer, call in list(pushed):
                    for ret in self.returns:
                        combined = _compose_summary(representative[outer], self.wrap(representative[state], call, ret))
                        target = label(combined, isinstance(outer, tuple))
                        returns.add((state, ret, (outer, call), target))
                        if target not in reachable:
                            frontier.append(target)

        result = cls(
            input_alphabet=self.vpa.input_alphabet,
            call_alphabet=self.vpa.call_alphabet,
            return_alphabet=self.vpa.return_alphabet,
            internal_alphabet=self.vpa.internal_alphabet,
            stack_alphabet=frozenset(pushed),
            bottom_stack_symbol=None,
            initial_state=initial,
            accepting_states=frozenset(
                s for s in reachable if not isinstance(s, tuple) and self.accepts(representative[s])
            ),
            summary_representatives={s: representative[s] for s in reachable},
        )
        for state in sorted(reachable, key=repr):
            result.graph.add_state(state)
        for source, symbol, target in sorted(internals, key=repr):
            result.add_internal_transition(source, target, symbol)
        for source, symbol, target, push in sorted(calls, key=repr):
            result.add_call_transition(source, target, symbol, push)
        for source, symbol, guard, target in sorted(returns, key=repr):
            result.add_return_transition(source, target, symbol, guard)
        result.validate()
        return result


def _number_blocks(signatures: Mapping[tuple, Any], *, first: Any) -> dict[tuple, int]:
    ordered = sorted(set(signatures.values()), key=lambda sig: (sig != first, repr(sig)))
    index = {signature: position for position, signature in enumerate(ordered)}
    return {summary: index[signature] for summary, signature in signatures.items()}


def _internal_summary(
    vpa: DeterministicVisiblyPushdownAutomaton,
    state_order: Sequence[Hashable],
    state_index: Mapping[Hashable, int],
    symbol: Any,
) -> tuple[int | None, ...]:
    transitions = vpa.internal_transition_map()
    summary: list[int | None] = []
    for state in state_order:
        target = transitions.get((state, symbol))
        summary.append(None if target is None else state_index[target])
    return tuple(summary)


def _close_summary_algebra(
    vpa: DeterministicVisiblyPushdownAutomaton,
    state_order: Sequence[Hashable],
    state_index: Mapping[Hashable, int],
    identity: tuple[int | None, ...],
    internal_summaries: Mapping[Any, tuple[int | None, ...]],
) -> set[tuple[int | None, ...]]:
    summaries = {identity, *internal_summaries.values()}
    queue: deque[tuple[int | None, ...]] = deque(sorted(summaries, key=repr))
    while queue:
        summary = queue.popleft()
        current = list(summaries)
        candidates: list[tuple[int | None, ...]] = []
        for other in current:
            candidates.append(_compose_summary(summary, other))
            candidates.append(_compose_summary(other, summary))
        for call_symbol in sorted(vpa.call_alphabet, key=repr):
            for return_symbol in sorted(vpa.return_alphabet, key=repr):
                candidates.append(_wrap_summary(vpa, state_order, summary, call_symbol, return_symbol))
        for candidate in candidates:
            if candidate not in summaries:
                summaries.add(candidate)
                queue.append(candidate)
    return summaries


def _compose_summary(
    first: tuple[int | None, ...],
    second: tuple[int | None, ...],
) -> tuple[int | None, ...]:
    return tuple(None if state is None else second[state] for state in first)


def _wrap_summary(
    vpa: DeterministicVisiblyPushdownAutomaton,
    state_order: Sequence[Hashable],
    inner: tuple[int | None, ...],
    call_symbol: Any,
    return_symbol: Any,
) -> tuple[int | None, ...]:
    state_index = {state: index for index, state in enumerate(state_order)}
    call_map = vpa.call_transition_map()
    result: list[int | None] = []
    for state in state_order:
        call = call_map.get((state, call_symbol))
        if call is None:
            result.append(None)
            continue
        call_target, stack_symbol = call
        inner_target_index = inner[state_index[call_target]]
        if inner_target_index is None:
            result.append(None)
            continue
        return_target = vpa.return_successor(state_order[inner_target_index], return_symbol, stack_symbol)
        result.append(None if return_target is None else state_index[return_target])
    return tuple(result)


def _summary_accepts(
    vpa: DeterministicVisiblyPushdownAutomaton,
    state_order: Sequence[Hashable],
    summary: tuple[int | None, ...],
) -> bool:
    if vpa.initial_state is None:
        return False
    initial_index = {state: index for index, state in enumerate(state_order)}[vpa.initial_state]
    target = summary[initial_index]
    return target is not None and state_order[target] in vpa.accepting_states
