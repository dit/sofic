"""Deterministic visibly pushdown automata."""

from __future__ import annotations

from collections.abc import Hashable
from typing import Any

from sofic.automata.vpa import operations as vc
from sofic.automata.vpa.base import _MISSING, VisiblyPushdownAutomaton
from sofic.exceptions import NonDeterministicError
from sofic.graph import (
    ATTR_KIND,
    ATTR_STACK_SYMBOL,
    ATTR_SYMBOL,
    KIND_CALL,
    KIND_INTERNAL,
    KIND_RETURN,
)


class DeterministicVisiblyPushdownAutomaton(VisiblyPushdownAutomaton):
    """VPA with at most one enabled transition for each visible configuration."""

    def validate(self) -> None:
        super().validate()
        if self.initial_state is None:
            raise NonDeterministicError("deterministic VPA requires an initial state")
        self._check_determinism()

    def _check_determinism(self) -> None:
        call_keys: set[tuple[Hashable, Any]] = set()
        internal_keys: set[tuple[Hashable, Any]] = set()
        return_keys: set[tuple[Hashable, Any, Any]] = set()
        wildcard_returns: set[tuple[Hashable, Any]] = set()

        for transition in self.transitions():
            kind = transition.data.get(ATTR_KIND)
            symbol = transition.data.get(ATTR_SYMBOL)
            if symbol is None:
                continue
            if kind == KIND_CALL:
                key = (transition.source, symbol)
                if key in call_keys:
                    raise NonDeterministicError(f"non-deterministic call transition on {key}")
                call_keys.add(key)
            elif kind == KIND_INTERNAL:
                key = (transition.source, symbol)
                if key in internal_keys:
                    raise NonDeterministicError(f"non-deterministic internal transition on {key}")
                internal_keys.add(key)
            elif kind == KIND_RETURN:
                stack_symbol = transition.data.get(ATTR_STACK_SYMBOL)
                wildcard_key = (transition.source, symbol)
                if stack_symbol is None:
                    if wildcard_key in wildcard_returns:
                        raise NonDeterministicError(f"duplicate wildcard return transition on {wildcard_key}")
                    if any(source == transition.source and ret == symbol for source, ret, _stack in return_keys):
                        raise NonDeterministicError(
                            f"wildcard return overlaps guarded return on {wildcard_key}",
                        )
                    wildcard_returns.add(wildcard_key)
                else:
                    key = (transition.source, symbol, stack_symbol)
                    if wildcard_key in wildcard_returns:
                        raise NonDeterministicError(
                            f"guarded return overlaps wildcard return on {wildcard_key}",
                        )
                    if key in return_keys:
                        raise NonDeterministicError(f"duplicate guarded return transition on {key}")
                    return_keys.add(key)

    def add_call_transition(
        self,
        source: Hashable,
        target: Hashable,
        symbol: Any,
        stack_symbol: Any,
        **attrs: Any,
    ) -> int:
        for transition in self.graph.out_transitions(source):
            if transition.data.get(ATTR_KIND) == KIND_CALL and transition.data.get(ATTR_SYMBOL) == symbol:
                raise NonDeterministicError(f"non-deterministic call transition on {(source, symbol)}")
        return super().add_call_transition(source, target, symbol, stack_symbol, **attrs)

    def add_return_transition(
        self,
        source: Hashable,
        target: Hashable,
        symbol: Any,
        stack_symbol: Any | None = None,
        **attrs: Any,
    ) -> int:
        for transition in self.graph.out_transitions(source):
            if transition.data.get(ATTR_KIND) != KIND_RETURN or transition.data.get(ATTR_SYMBOL) != symbol:
                continue
            existing_stack = transition.data.get(ATTR_STACK_SYMBOL)
            if existing_stack is None or stack_symbol is None or existing_stack == stack_symbol:
                raise NonDeterministicError(f"non-deterministic return transition on {(source, symbol)}")
        return super().add_return_transition(source, target, symbol, stack_symbol, **attrs)

    def add_internal_transition(self, source: Hashable, target: Hashable, symbol: Any, **attrs: Any) -> int:
        for transition in self.graph.out_transitions(source):
            if transition.data.get(ATTR_KIND) == KIND_INTERNAL and transition.data.get(ATTR_SYMBOL) == symbol:
                raise NonDeterministicError(f"non-deterministic internal transition on {(source, symbol)}")
        return super().add_internal_transition(source, target, symbol, **attrs)

    def call_successor(self, state: Hashable, symbol: Any) -> tuple[Hashable, Any] | None:
        """Return ``(target, pushed_stack_symbol)`` for a deterministic call."""
        return self.call_transition_map().get((state, symbol))

    def internal_successor(self, state: Hashable, symbol: Any) -> Hashable | None:
        """Return the deterministic internal successor, if present."""
        return self.internal_transition_map().get((state, symbol))

    def return_successor(self, state: Hashable, symbol: Any, stack_symbol: Any) -> Hashable | None:
        """Return the deterministic return successor for ``stack_symbol``, if present."""
        transitions = self.return_transition_map()
        explicit = transitions.get((state, symbol, stack_symbol), _MISSING)
        if explicit is not _MISSING:
            return explicit
        wildcard = transitions.get((state, symbol, None), _MISSING)
        if wildcard is not _MISSING:
            return wildcard
        return None

    @classmethod
    def from_vpa(cls, vpa: VisiblyPushdownAutomaton) -> DeterministicVisiblyPushdownAutomaton:
        """Return ``vpa`` as a deterministic VPA, determinizing it when needed.

        An already deterministic ``vpa`` is copied with its states unchanged;
        otherwise the summary construction of :cite:`AlurMadhusudan2009` is
        applied (see :func:`~sofic.automata.vpa.operations.determinize`).
        """
        if vpa.initial_state is None or not _is_deterministic(vpa):
            return vc.determinize_vpa(vpa)
        result = cls(
            input_alphabet=vpa.input_alphabet,
            call_alphabet=vpa.call_alphabet,
            return_alphabet=vpa.return_alphabet,
            internal_alphabet=vpa.internal_alphabet,
            stack_alphabet=vpa.stack_alphabet,
            bottom_stack_symbol=vpa.bottom_stack_symbol,
            initial_state=vpa.initial_state,
            accepting_states=vpa.accepting_states,
            graph=vpa.graph.copy(),
        )
        result.validate()
        return result


def _is_deterministic(vpa: VisiblyPushdownAutomaton) -> bool:
    probe = DeterministicVisiblyPushdownAutomaton(
        call_alphabet=vpa.call_alphabet,
        return_alphabet=vpa.return_alphabet,
        internal_alphabet=vpa.internal_alphabet,
        stack_alphabet=vpa.stack_alphabet,
        bottom_stack_symbol=vpa.bottom_stack_symbol,
        initial_state=vpa.initial_state,
        accepting_states=vpa.accepting_states,
        graph=vpa.graph,
    )
    try:
        probe._check_determinism()
    except NonDeterministicError:
        return False
    return True
