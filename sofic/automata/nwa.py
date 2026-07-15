"""Nested word automata."""

from __future__ import annotations

from collections.abc import Hashable, Iterable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from sofic.base import StateMachine
from sofic.exceptions import SoficValidationError
from sofic.graph import (
    ATTR_HIER_STATE,
    ATTR_KIND,
    ATTR_STACK_SYMBOL,
    ATTR_SYMBOL,
    KIND_CALL,
    KIND_INTERNAL,
    KIND_RETURN,
)

if TYPE_CHECKING:
    from sofic.automata.vpa import VisiblyPushdownAutomaton

_KINDS = frozenset({KIND_CALL, KIND_RETURN, KIND_INTERNAL})


@dataclass(frozen=True, slots=True)
class NestedWord:
    """Finite word together with an explicit call-return matching relation."""

    symbols: tuple[Any, ...]
    kinds: tuple[str, ...]
    matching: tuple[int | None, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "symbols", tuple(self.symbols))
        object.__setattr__(self, "kinds", tuple(self.kinds))
        object.__setattr__(self, "matching", tuple(self.matching))
        self.validate()

    @classmethod
    def from_visible_word(
        cls,
        symbols: Sequence[Any],
        *,
        call_alphabet: Iterable[Any],
        return_alphabet: Iterable[Any],
        internal_alphabet: Iterable[Any] = (),
    ) -> NestedWord:
        """Build the canonical nested word induced by a visible alphabet."""
        symbols = tuple(symbols)
        call_alphabet = frozenset(call_alphabet)
        return_alphabet = frozenset(return_alphabet)
        internal_alphabet = frozenset(internal_alphabet)
        _require_disjoint_visible_alphabets(call_alphabet, return_alphabet, internal_alphabet)

        kinds: list[str] = []
        matching: list[int | None] = [None] * len(symbols)
        stack: list[int] = []
        for index, symbol in enumerate(symbols):
            if symbol in call_alphabet:
                kinds.append(KIND_CALL)
                stack.append(index)
            elif symbol in return_alphabet:
                kinds.append(KIND_RETURN)
                if stack:
                    call_index = stack.pop()
                    matching[call_index] = index
                    matching[index] = call_index
            elif symbol in internal_alphabet:
                kinds.append(KIND_INTERNAL)
            else:
                raise ValueError(f"symbol {symbol!r} is not in the visible alphabet")

        return cls(symbols=symbols, kinds=tuple(kinds), matching=tuple(matching))

    def validate(self) -> None:
        """Raise if the matching relation is not a valid nested-word relation."""
        if len(self.symbols) != len(self.kinds) or len(self.symbols) != len(self.matching):
            raise SoficValidationError("symbols, kinds, and matching must have the same length")

        for index, kind in enumerate(self.kinds):
            if kind not in _KINDS:
                raise SoficValidationError(f"invalid nested-word kind {kind!r} at position {index}")

        for index, (kind, partner) in enumerate(zip(self.kinds, self.matching, strict=True)):
            if partner is None:
                continue
            if not isinstance(partner, int) or partner < 0 or partner >= len(self.symbols):
                raise SoficValidationError(f"matching partner {partner!r} out of range at position {index}")
            if self.matching[partner] != index:
                raise SoficValidationError("matching relation must be symmetric")
            partner_kind = self.kinds[partner]
            if kind == KIND_INTERNAL:
                raise SoficValidationError("internal positions cannot be matched")
            if kind == KIND_CALL and not (partner_kind == KIND_RETURN and index < partner):
                raise SoficValidationError("call positions must match later return positions")
            if kind == KIND_RETURN and not (partner_kind == KIND_CALL and partner < index):
                raise SoficValidationError("return positions must match earlier call positions")

        stack: list[int] = []
        for index, (kind, partner) in enumerate(zip(self.kinds, self.matching, strict=True)):
            if kind == KIND_CALL and partner is not None:
                stack.append(index)
            elif kind == KIND_RETURN and partner is not None:
                if not stack or stack[-1] != partner:
                    raise SoficValidationError("matching relation must be properly nested")
                stack.pop()


class NestedWordAutomaton(StateMachine):
    """Nondeterministic nested word automaton over explicit nested words."""

    input_alphabet: frozenset[Any]
    call_alphabet: frozenset[Any]
    return_alphabet: frozenset[Any]
    internal_alphabet: frozenset[Any]
    hier_alphabet: frozenset[Any]
    bottom_hier_state: Any | None
    initial_state: Hashable | None
    accepting_states: frozenset[Hashable]

    def __init__(
        self,
        input_alphabet: frozenset[Any] | None = None,
        call_alphabet: frozenset[Any] | None = None,
        return_alphabet: frozenset[Any] | None = None,
        internal_alphabet: frozenset[Any] | None = None,
        hier_alphabet: frozenset[Any] | None = None,
        bottom_hier_state: Any | None = None,
        initial_state: Hashable | None = None,
        accepting_states: frozenset[Hashable] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.call_alphabet = call_alphabet if call_alphabet is not None else frozenset()
        self.return_alphabet = return_alphabet if return_alphabet is not None else frozenset()
        self.internal_alphabet = internal_alphabet if internal_alphabet is not None else frozenset()
        self.hier_alphabet = hier_alphabet if hier_alphabet is not None else frozenset()
        self.bottom_hier_state = bottom_hier_state
        self.input_alphabet = (
            input_alphabet
            if input_alphabet is not None
            else self.call_alphabet | self.return_alphabet | self.internal_alphabet
        )
        self.initial_state = initial_state
        self.accepting_states = accepting_states if accepting_states is not None else frozenset()

    def validate(self) -> None:
        role_alphabet = self.call_alphabet | self.return_alphabet | self.internal_alphabet
        self._require(self.input_alphabet == role_alphabet, "input alphabet must equal the union of role alphabets")
        if self.bottom_hier_state is not None:
            self._require(self.bottom_hier_state in self.hier_alphabet, "bottom_hier_state must be in hier alphabet")
        if self.initial_state is not None:
            self._require(self.graph.has_state(self.initial_state), "missing initial state")
        for state in self.accepting_states:
            self._require(self.graph.has_state(state), f"missing accepting state {state!r}")

        for transition in self.transitions():
            kind = transition.data.get(ATTR_KIND)
            symbol = transition.data.get(ATTR_SYMBOL)
            self._require(kind in _KINDS, f"invalid NWA kind {kind!r}")
            if kind == KIND_CALL:
                self._require(symbol in self.call_alphabet, f"{symbol!r} not in call alphabet")
                self._require(ATTR_HIER_STATE in transition.data, "call edge requires a hier_state")
                hier_state = transition.data[ATTR_HIER_STATE]
                self._require(hier_state in self.hier_alphabet, "call hier_state must be in hier alphabet")
                self._require(hier_state != self.bottom_hier_state, "call edge cannot push the bottom_hier_state")
            elif kind == KIND_RETURN:
                self._require(symbol in self.return_alphabet, f"{symbol!r} not in return alphabet")
                self._require(ATTR_HIER_STATE in transition.data, "return edge requires a hier_state")
                self._require(
                    transition.data[ATTR_HIER_STATE] in self.hier_alphabet,
                    "return hier_state must be in hier alphabet",
                )
            else:
                self._require(symbol in self.internal_alphabet, f"{symbol!r} not in internal alphabet")
                self._require(ATTR_HIER_STATE not in transition.data, "internal edge cannot carry a hier_state")

    def add_call_transition(
        self,
        source: Hashable,
        target: Hashable,
        symbol: Any,
        hier_state: Any,
        **attrs: Any,
    ) -> int:
        """Add a call transition that stores ``hier_state`` for its matching return."""
        data = {**attrs, ATTR_KIND: KIND_CALL, ATTR_SYMBOL: symbol, ATTR_HIER_STATE: hier_state}
        return self.graph.add_transition(source, target, **data)

    def add_return_transition(
        self,
        source: Hashable,
        target: Hashable,
        symbol: Any,
        hier_state: Any,
        **attrs: Any,
    ) -> int:
        """Add a return transition guarded by ``hier_state``."""
        data = {**attrs, ATTR_KIND: KIND_RETURN, ATTR_SYMBOL: symbol, ATTR_HIER_STATE: hier_state}
        return self.graph.add_transition(source, target, **data)

    def add_internal_transition(self, source: Hashable, target: Hashable, symbol: Any, **attrs: Any) -> int:
        """Add an internal transition."""
        data = {**attrs, ATTR_KIND: KIND_INTERNAL, ATTR_SYMBOL: symbol}
        return self.graph.add_transition(source, target, **data)

    def recognizes(self, word: NestedWord) -> bool:
        from sofic.automata.nwa_simulation import recognizes_nwa

        return recognizes_nwa(self, word)

    def recognizes_visible(self, symbols: Sequence[Any]) -> bool:
        """Recognize an ordinary word using the NWA's visible role alphabets."""
        word = NestedWord.from_visible_word(
            symbols,
            call_alphabet=self.call_alphabet,
            return_alphabet=self.return_alphabet,
            internal_alphabet=self.internal_alphabet,
        )
        return self.recognizes(word)

    @classmethod
    def from_vpa(cls, vpa: VisiblyPushdownAutomaton) -> NestedWordAutomaton:
        """Copy a visibly pushdown automaton into an equivalent NWA view."""
        result = cls(
            input_alphabet=vpa.input_alphabet,
            call_alphabet=vpa.call_alphabet,
            return_alphabet=vpa.return_alphabet,
            internal_alphabet=vpa.internal_alphabet,
            hier_alphabet=vpa.stack_alphabet,
            bottom_hier_state=vpa.bottom_stack_symbol,
            initial_state=vpa.initial_state,
            accepting_states=vpa.accepting_states,
        )
        for state in vpa.states():
            result.graph.add_state(state, **vpa.graph.state_attrs(state))

        for transition in vpa.transitions():
            data = transition.data
            kind = data.get(ATTR_KIND)
            symbol = data.get(ATTR_SYMBOL)
            attrs = _without_transition_keys(data, ATTR_STACK_SYMBOL)
            if kind == KIND_CALL:
                result.add_call_transition(
                    transition.source,
                    transition.target,
                    symbol,
                    data.get(ATTR_STACK_SYMBOL),
                    **attrs,
                )
            elif kind == KIND_RETURN:
                for hier_state in _return_hier_states(vpa.stack_alphabet, data.get(ATTR_STACK_SYMBOL)):
                    result.add_return_transition(transition.source, transition.target, symbol, hier_state, **attrs)
            elif kind == KIND_INTERNAL:
                result.add_internal_transition(transition.source, transition.target, symbol, **attrs)

        return result

    def to_vpa(self, *, tag_symbols: bool = True) -> VisiblyPushdownAutomaton:
        """Encode this NWA as a visibly pushdown automaton.

        If ``tag_symbols`` is true, VPA symbols are role-tagged as
        ``("call", symbol)``, ``("return", symbol)``, and
        ``("internal", symbol)``. If false, the role alphabets must be disjoint.
        """
        from sofic.automata.vpa import VisiblyPushdownAutomaton

        if not tag_symbols:
            _require_disjoint_visible_alphabets(self.call_alphabet, self.return_alphabet, self.internal_alphabet)

        call_alphabet = frozenset(_encoded_symbol(KIND_CALL, symbol, tag_symbols) for symbol in self.call_alphabet)
        return_alphabet = frozenset(
            _encoded_symbol(KIND_RETURN, symbol, tag_symbols) for symbol in self.return_alphabet
        )
        internal_alphabet = frozenset(
            _encoded_symbol(KIND_INTERNAL, symbol, tag_symbols) for symbol in self.internal_alphabet
        )
        result = VisiblyPushdownAutomaton(
            call_alphabet=call_alphabet,
            return_alphabet=return_alphabet,
            internal_alphabet=internal_alphabet,
            stack_alphabet=self.hier_alphabet,
            bottom_stack_symbol=self.bottom_hier_state,
            initial_state=self.initial_state,
            accepting_states=self.accepting_states,
        )
        for state in self.states():
            result.graph.add_state(state, **self.graph.state_attrs(state))

        for transition in self.transitions():
            data = transition.data
            kind = data.get(ATTR_KIND)
            symbol = _encoded_symbol(kind, data.get(ATTR_SYMBOL), tag_symbols)
            attrs = _without_transition_keys(data, ATTR_HIER_STATE)
            if kind == KIND_CALL:
                result.add_call_transition(
                    transition.source,
                    transition.target,
                    symbol,
                    data.get(ATTR_HIER_STATE),
                    **attrs,
                )
            elif kind == KIND_RETURN:
                result.add_return_transition(
                    transition.source,
                    transition.target,
                    symbol,
                    data.get(ATTR_HIER_STATE),
                    **attrs,
                )
            elif kind == KIND_INTERNAL:
                result.add_internal_transition(transition.source, transition.target, symbol, **attrs)

        return result


def _require_disjoint_visible_alphabets(
    call_alphabet: frozenset[Any],
    return_alphabet: frozenset[Any],
    internal_alphabet: frozenset[Any],
) -> None:
    visible = call_alphabet | return_alphabet | internal_alphabet
    if len(call_alphabet) + len(return_alphabet) + len(internal_alphabet) != len(visible):
        raise ValueError("visible alphabets must be disjoint")


def _without_transition_keys(data: dict[str, Any], *extra: str) -> dict[str, Any]:
    excluded = {ATTR_KIND, ATTR_SYMBOL, *extra}
    return {key: value for key, value in data.items() if key not in excluded}


def _return_hier_states(hier_alphabet: frozenset[Any], hier_state: Any | None) -> frozenset[Any]:
    if hier_state is not None:
        return frozenset({hier_state})
    return hier_alphabet


def _encoded_symbol(kind: Any, symbol: Any, tag_symbols: bool) -> Any:
    if tag_symbols:
        return (kind, symbol)
    return symbol
