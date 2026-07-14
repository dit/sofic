"""Sofic-Dyck shift presentations."""

from __future__ import annotations

from collections.abc import Hashable, Iterator, Sequence
from typing import Any

from pensive.graph import ATTR_KIND, ATTR_SYMBOL, KIND_CALL, KIND_INTERNAL, KIND_RETURN, Transition
from pensive.shifts.base import SymbolicModel

TransitionRef = tuple[Hashable, Hashable, int]
MatchedEdge = tuple[TransitionRef, TransitionRef]

_KINDS = frozenset({KIND_CALL, KIND_RETURN, KIND_INTERNAL})


def transition_ref(transition: Transition) -> TransitionRef:
    """Return the stable graph reference for ``transition``."""
    return transition.source, transition.target, transition.key


class SoficDyckShift(SymbolicModel):
    """Sofic-Dyck shift presented by a finite Dyck automaton."""

    call_alphabet: frozenset[Any]
    return_alphabet: frozenset[Any]
    internal_alphabet: frozenset[Any]
    matched_edges: frozenset[MatchedEdge]

    def __init__(
        self,
        call_alphabet: frozenset[Any] | None = None,
        return_alphabet: frozenset[Any] | None = None,
        internal_alphabet: frozenset[Any] | None = None,
        matched_edges: set[MatchedEdge] | frozenset[MatchedEdge] | None = None,
        symbol_alphabet: frozenset[Any] | None = None,
        **kwargs: Any,
    ) -> None:
        self.call_alphabet = call_alphabet if call_alphabet is not None else frozenset()
        self.return_alphabet = return_alphabet if return_alphabet is not None else frozenset()
        self.internal_alphabet = internal_alphabet if internal_alphabet is not None else frozenset()
        inferred_alphabet = self.call_alphabet | self.return_alphabet | self.internal_alphabet
        super().__init__(
            symbol_alphabet=symbol_alphabet if symbol_alphabet is not None else inferred_alphabet, **kwargs
        )
        self.matched_edges = frozenset(matched_edges or frozenset())

    def validate(self) -> None:
        super().validate()
        role_alphabet = self.call_alphabet | self.return_alphabet | self.internal_alphabet
        self._require(
            len(self.call_alphabet) + len(self.return_alphabet) + len(self.internal_alphabet) == len(role_alphabet),
            "call, return, and internal alphabets must be disjoint",
        )
        self._require(self.symbol_alphabet == role_alphabet, "symbol_alphabet must equal the visible role alphabets")

        call_edges: set[TransitionRef] = set()
        return_edges: set[TransitionRef] = set()
        all_edges: set[TransitionRef] = set()
        for transition in self.transitions():
            ref = transition_ref(transition)
            all_edges.add(ref)
            kind = transition.data.get(ATTR_KIND)
            symbol = transition.data.get(ATTR_SYMBOL)
            self._require(kind in _KINDS, f"invalid Dyck edge kind {kind!r}")
            self._require(symbol is not None, "Dyck shift transitions require a symbol")
            if kind == KIND_CALL:
                self._require(symbol in self.call_alphabet, f"{symbol!r} not in call alphabet")
                call_edges.add(ref)
            elif kind == KIND_RETURN:
                self._require(symbol in self.return_alphabet, f"{symbol!r} not in return alphabet")
                return_edges.add(ref)
            else:
                self._require(symbol in self.internal_alphabet, f"{symbol!r} not in internal alphabet")

        for call_ref, return_ref in self.matched_edges:
            self._require(call_ref in all_edges, f"matched call edge {call_ref!r} is missing")
            self._require(return_ref in all_edges, f"matched return edge {return_ref!r} is missing")
            self._require(call_ref in call_edges, f"matched edge {call_ref!r} is not a call transition")
            self._require(return_ref in return_edges, f"matched edge {return_ref!r} is not a return transition")

    def add_call_transition(self, source: Hashable, target: Hashable, symbol: Any, **attrs: Any) -> TransitionRef:
        """Add a call transition and return its stable edge reference."""
        key = self.graph.add_transition(source, target, **{**attrs, ATTR_KIND: KIND_CALL, ATTR_SYMBOL: symbol})
        return source, target, key

    def add_return_transition(self, source: Hashable, target: Hashable, symbol: Any, **attrs: Any) -> TransitionRef:
        """Add a return transition and return its stable edge reference."""
        key = self.graph.add_transition(source, target, **{**attrs, ATTR_KIND: KIND_RETURN, ATTR_SYMBOL: symbol})
        return source, target, key

    def add_internal_transition(self, source: Hashable, target: Hashable, symbol: Any, **attrs: Any) -> TransitionRef:
        """Add an internal transition and return its stable edge reference."""
        key = self.graph.add_transition(source, target, **{**attrs, ATTR_KIND: KIND_INTERNAL, ATTR_SYMBOL: symbol})
        return source, target, key

    def add_matched_pair(self, call_ref: TransitionRef, return_ref: TransitionRef) -> None:
        """Mark ``call_ref`` and ``return_ref`` as a legal call-return pair."""
        self.matched_edges = frozenset({*self.matched_edges, (call_ref, return_ref)})

    def is_admissible_word(self, word: Sequence[Any]) -> bool:
        """Return whether ``word`` is a finite factor of this Dyck presentation."""
        from pensive.shifts.dyck_algorithms import is_admissible_word

        return is_admissible_word(self, word)

    def factor_language(self, length: int) -> Iterator[tuple[Any, ...]]:
        from pensive.shifts.dyck_algorithms import admissible_words

        yield from admissible_words(self, length)
