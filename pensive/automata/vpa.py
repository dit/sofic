"""Visibly pushdown automata."""

from __future__ import annotations

from collections.abc import Hashable, Sequence
from typing import Any

from pensive.base import StateMachine
from pensive.graph import ATTR_KIND, ATTR_STACK_SYMBOL, ATTR_SYMBOL, KIND_CALL, KIND_INTERNAL, KIND_RETURN


class VisiblyPushdownAutomaton(StateMachine):
    """Standard 1-stack VPA with call / return / internal input partition."""

    input_alphabet: frozenset[Any]
    call_alphabet: frozenset[Any]
    return_alphabet: frozenset[Any]
    internal_alphabet: frozenset[Any]
    stack_alphabet: frozenset[Any]
    initial_state: Hashable | None
    accepting_states: frozenset[Hashable]

    def __init__(
        self,
        input_alphabet: frozenset[Any] | None = None,
        call_alphabet: frozenset[Any] | None = None,
        return_alphabet: frozenset[Any] | None = None,
        internal_alphabet: frozenset[Any] | None = None,
        stack_alphabet: frozenset[Any] | None = None,
        initial_state: Hashable | None = None,
        accepting_states: frozenset[Hashable] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.call_alphabet = call_alphabet if call_alphabet is not None else frozenset()
        self.return_alphabet = return_alphabet if return_alphabet is not None else frozenset()
        self.internal_alphabet = internal_alphabet if internal_alphabet is not None else frozenset()
        self.stack_alphabet = stack_alphabet if stack_alphabet is not None else frozenset()
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
        if self.initial_state is not None:
            self._require(self.graph.has_state(self.initial_state), "missing initial state")
        for transition in self.transitions():
            kind = transition.data.get(ATTR_KIND)
            symbol = transition.data.get(ATTR_SYMBOL)
            self._require(kind in {KIND_CALL, KIND_RETURN, KIND_INTERNAL}, f"invalid VPA kind {kind!r}")
            if symbol is not None:
                if kind == KIND_CALL:
                    self._require(symbol in self.call_alphabet, f"{symbol!r} not in call alphabet")
                    stack_sym = transition.data.get(ATTR_STACK_SYMBOL)
                    self._require(stack_sym in self.stack_alphabet, "call edge requires stack_symbol in stack alphabet")
                elif kind == KIND_RETURN:
                    self._require(symbol in self.return_alphabet, f"{symbol!r} not in return alphabet")
                else:
                    self._require(symbol in self.internal_alphabet, f"{symbol!r} not in internal alphabet")

    def recognizes(self, word: Sequence[Any]) -> bool:
        from pensive.automata.vpa_simulation import recognizes_vpa

        return recognizes_vpa(self, word)
