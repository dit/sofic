"""Visibly pushdown automata."""

from __future__ import annotations

from collections.abc import Hashable, Sequence
from typing import TYPE_CHECKING, Any

from sofic.automata.vpa import operations as vc
from sofic.base import StateMachine
from sofic.exceptions import NonDeterministicError
from sofic.graph import (
    ATTR_KIND,
    ATTR_STACK_SYMBOL,
    ATTR_SYMBOL,
    KIND_CALL,
    KIND_INTERNAL,
    KIND_RETURN,
)

if TYPE_CHECKING:
    from sofic.automata.vpa.deterministic import DeterministicVisiblyPushdownAutomaton

_MISSING = object()


class VisiblyPushdownAutomaton(StateMachine):
    """Standard 1-stack VPA with call / return / internal input partition."""

    input_alphabet: frozenset[Any]
    call_alphabet: frozenset[Any]
    return_alphabet: frozenset[Any]
    internal_alphabet: frozenset[Any]
    stack_alphabet: frozenset[Any]
    bottom_stack_symbol: Any | None
    initial_state: Hashable | None
    accepting_states: frozenset[Hashable]

    def __init__(
        self,
        input_alphabet: frozenset[Any] | None = None,
        call_alphabet: frozenset[Any] | None = None,
        return_alphabet: frozenset[Any] | None = None,
        internal_alphabet: frozenset[Any] | None = None,
        stack_alphabet: frozenset[Any] | None = None,
        bottom_stack_symbol: Any | None = None,
        initial_state: Hashable | None = None,
        accepting_states: frozenset[Hashable] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.call_alphabet = call_alphabet if call_alphabet is not None else frozenset()
        self.return_alphabet = return_alphabet if return_alphabet is not None else frozenset()
        self.internal_alphabet = internal_alphabet if internal_alphabet is not None else frozenset()
        self.stack_alphabet = stack_alphabet if stack_alphabet is not None else frozenset()
        self.bottom_stack_symbol = bottom_stack_symbol
        self.input_alphabet = (
            input_alphabet
            if input_alphabet is not None
            else self.call_alphabet | self.return_alphabet | self.internal_alphabet
        )
        self.initial_state = initial_state
        self.accepting_states = accepting_states if accepting_states is not None else frozenset()

    def validate(self) -> None:
        partition = self.call_alphabet | self.return_alphabet | self.internal_alphabet
        self._require(
            len(self.call_alphabet) + len(self.return_alphabet) + len(self.internal_alphabet) == len(partition),
            "call, return, and internal alphabets must be disjoint",
        )
        self._require(partition == self.input_alphabet, "input alphabet must equal partition of call/return/internal")
        if self.bottom_stack_symbol is not None:
            self._require(
                self.bottom_stack_symbol in self.stack_alphabet, "bottom_stack_symbol must be in stack alphabet"
            )
        if self.initial_state is not None:
            self._require(self.graph.has_state(self.initial_state), "missing initial state")
        for state in self.accepting_states:
            self._require(self.graph.has_state(state), f"missing accepting state {state!r}")
        for transition in self.transitions():
            kind = transition.data.get(ATTR_KIND)
            symbol = transition.data.get(ATTR_SYMBOL)
            self._require(kind in {KIND_CALL, KIND_RETURN, KIND_INTERNAL}, f"invalid VPA kind {kind!r}")
            if symbol is not None:
                if kind == KIND_CALL:
                    self._require(symbol in self.call_alphabet, f"{symbol!r} not in call alphabet")
                    stack_sym = transition.data.get(ATTR_STACK_SYMBOL)
                    self._require(stack_sym in self.stack_alphabet, "call edge requires stack_symbol in stack alphabet")
                    self._require(
                        stack_sym != self.bottom_stack_symbol,
                        "call edge cannot push the bottom_stack_symbol",
                    )
                elif kind == KIND_RETURN:
                    self._require(symbol in self.return_alphabet, f"{symbol!r} not in return alphabet")
                    stack_sym = transition.data.get(ATTR_STACK_SYMBOL)
                    if stack_sym is not None:
                        self._require(stack_sym in self.stack_alphabet, "return stack_symbol must be in stack alphabet")
                else:
                    self._require(symbol in self.internal_alphabet, f"{symbol!r} not in internal alphabet")

    def add_call_transition(
        self,
        source: Hashable,
        target: Hashable,
        symbol: Any,
        stack_symbol: Any,
        **attrs: Any,
    ) -> int:
        """Add a call transition that pushes ``stack_symbol``."""
        data = {**attrs, ATTR_KIND: KIND_CALL, ATTR_SYMBOL: symbol, ATTR_STACK_SYMBOL: stack_symbol}
        return self.graph.add_transition(source, target, **data)

    def add_return_transition(
        self,
        source: Hashable,
        target: Hashable,
        symbol: Any,
        stack_symbol: Any | None = None,
        **attrs: Any,
    ) -> int:
        """Add a return transition.

        If ``stack_symbol`` is omitted, the transition is a wildcard: it fires
        on every stack symbol, and also on the empty stack (leaving it empty)
        when the VPA has a ``bottom_stack_symbol``. A return guarded by the
        bottom symbol fires only on the empty stack.
        """
        data = {**attrs, ATTR_KIND: KIND_RETURN, ATTR_SYMBOL: symbol}
        if stack_symbol is not None:
            data[ATTR_STACK_SYMBOL] = stack_symbol
        return self.graph.add_transition(source, target, **data)

    def add_internal_transition(self, source: Hashable, target: Hashable, symbol: Any, **attrs: Any) -> int:
        """Add an internal transition."""
        data = {**attrs, ATTR_KIND: KIND_INTERNAL, ATTR_SYMBOL: symbol}
        return self.graph.add_transition(source, target, **data)

    def call_transition_map(self) -> dict[tuple[Hashable, Any], tuple[Hashable, Any]]:
        """Return deterministic call transitions keyed by ``(state, symbol)``."""
        result: dict[tuple[Hashable, Any], tuple[Hashable, Any]] = {}
        for transition in self.transitions():
            if transition.data.get(ATTR_KIND) != KIND_CALL:
                continue
            symbol = transition.data.get(ATTR_SYMBOL)
            if symbol is None:
                continue
            key = (transition.source, symbol)
            value = (transition.target, transition.data.get(ATTR_STACK_SYMBOL))
            if key in result and result[key] != value:
                raise NonDeterministicError(f"non-deterministic call transition on {key}")
            result[key] = value
        return result

    def return_transition_map(self) -> dict[tuple[Hashable, Any, Any | None], Hashable]:
        """Return deterministic return transitions keyed by ``(state, symbol, stack_symbol)``."""
        result: dict[tuple[Hashable, Any, Any | None], Hashable] = {}
        for transition in self.transitions():
            if transition.data.get(ATTR_KIND) != KIND_RETURN:
                continue
            symbol = transition.data.get(ATTR_SYMBOL)
            if symbol is None:
                continue
            key = (transition.source, symbol, transition.data.get(ATTR_STACK_SYMBOL))
            value = transition.target
            if key in result and result[key] != value:
                raise NonDeterministicError(f"non-deterministic return transition on {key}")
            result[key] = value
        return result

    def internal_transition_map(self) -> dict[tuple[Hashable, Any], Hashable]:
        """Return deterministic internal transitions keyed by ``(state, symbol)``."""
        result: dict[tuple[Hashable, Any], Hashable] = {}
        for transition in self.transitions():
            if transition.data.get(ATTR_KIND) != KIND_INTERNAL:
                continue
            symbol = transition.data.get(ATTR_SYMBOL)
            if symbol is None:
                continue
            key = (transition.source, symbol)
            value = transition.target
            if key in result and result[key] != value:
                raise NonDeterministicError(f"non-deterministic internal transition on {key}")
            result[key] = value
        return result

    def recognizes(self, word: Sequence[Any]) -> bool:
        from sofic.automata.vpa.simulation import recognizes_vpa

        return recognizes_vpa(self, word)

    def union(self, other: VisiblyPushdownAutomaton) -> VisiblyPushdownAutomaton:
        """Return a VPA recognizing the union with ``other``."""
        return _binary(vc.union, self, other)

    def intersection(self, other: VisiblyPushdownAutomaton) -> VisiblyPushdownAutomaton:
        """Return a VPA recognizing the intersection with ``other`` (synchronized product)."""
        return _binary(vc.intersection, self, other)

    def complement(self) -> DeterministicVisiblyPushdownAutomaton:
        """Return a deterministic VPA recognizing the complement over this visible alphabet."""
        from sofic.automata.vpa.deterministic import DeterministicVisiblyPushdownAutomaton

        return vc.denormalize(vc.complement(vc.normalize(self)), DeterministicVisiblyPushdownAutomaton)

    def difference(self, other: VisiblyPushdownAutomaton) -> VisiblyPushdownAutomaton:
        """Return a VPA recognizing this language minus ``other``."""
        return _binary(vc.difference, self, other)

    def concat(self, other: VisiblyPushdownAutomaton) -> VisiblyPushdownAutomaton:
        """Return a VPA recognizing concatenation with ``other``.

        Each factor is read from an empty stack of its own: a return in the right
        factor that would pop a pending call of the left factor is a pending
        return of the right factor.
        """
        return _binary(vc.concat, self, other)

    def kleene_star(self) -> VisiblyPushdownAutomaton:
        """Return a VPA recognizing the Kleene star; each factor starts from its own empty stack."""
        return vc.denormalize(vc.kleene_star(vc.normalize(self)))

    def determinize(self) -> DeterministicVisiblyPushdownAutomaton:
        """Return an equivalent complete deterministic VPA :cite:`AlurMadhusudan2009`."""
        return vc.determinize_vpa(self)

    def is_empty(self) -> bool:
        """Return whether the language is empty."""
        return vc.is_empty(vc.normalize(self))

    def accepted_word(self) -> tuple[Any, ...] | None:
        """Return a short accepted word, or ``None`` when the language is empty."""
        return vc.accepted_word(vc.normalize(self))

    def is_universal(self) -> bool:
        """Return whether every word over the visible alphabet is accepted."""
        return vc.is_empty(vc.complement(vc.normalize(self)))

    def includes(self, other: VisiblyPushdownAutomaton) -> bool:
        """Return whether ``other``'s language is contained in this one."""
        return vc.is_empty(vc.difference(vc.normalize(other), vc.normalize(self)))

    def equivalent(self, other: VisiblyPushdownAutomaton) -> bool:
        """Return whether both VPAs recognize the same language."""
        return self.includes(other) and other.includes(self)

    def has_unmatched_word(self) -> bool:
        """Return whether some accepted word has a pending call or a pending return."""
        return vc.has_unmatched_word(vc.normalize(self))


def _binary(operation, left: VisiblyPushdownAutomaton, right: VisiblyPushdownAutomaton) -> VisiblyPushdownAutomaton:
    return vc.denormalize(operation(vc.normalize(left), vc.normalize(right)))
