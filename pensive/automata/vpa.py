"""Visibly pushdown automata."""

from __future__ import annotations

from collections import deque
from collections.abc import Hashable, Iterable, Mapping, Sequence
from typing import Any

from pensive.base import StateMachine
from pensive.exceptions import NonDeterministicError
from pensive.graph import (
    ATTR_KIND,
    ATTR_STACK_SYMBOL,
    ATTR_SYMBOL,
    KIND_CALL,
    KIND_INTERNAL,
    KIND_RETURN,
)

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

        If ``stack_symbol`` is omitted, the transition is a wildcard over
        non-bottom stack symbols. This preserves the historical unguarded
        return behavior of :class:`VisiblyPushdownAutomaton`.
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
        from pensive.automata.vpa_simulation import recognizes_vpa

        return recognizes_vpa(self, word)

    def union(self, other: VisiblyPushdownAutomaton) -> CompositeVisiblyPushdownAutomaton:
        """Return a VPA recognizer for the union with ``other``."""
        return union_vpa(self, other)

    def intersection(self, other: VisiblyPushdownAutomaton) -> CompositeVisiblyPushdownAutomaton:
        """Return a VPA recognizer for the intersection with ``other``."""
        return intersection_vpa(self, other)

    def intersect(self, other: VisiblyPushdownAutomaton) -> CompositeVisiblyPushdownAutomaton:
        """Alias for :meth:`intersection`."""
        return self.intersection(other)

    def complement(self) -> CompositeVisiblyPushdownAutomaton:
        """Return a VPA recognizer for the complement over this visible alphabet."""
        return complement_vpa(self)

    def difference(self, other: VisiblyPushdownAutomaton) -> CompositeVisiblyPushdownAutomaton:
        """Return a VPA recognizer for this language minus ``other``."""
        return difference_vpa(self, other)

    def concat(self, other: VisiblyPushdownAutomaton) -> CompositeVisiblyPushdownAutomaton:
        """Return a VPA recognizer for concatenation with ``other``."""
        return concat_vpa(self, other)

    def concatenate(self, other: VisiblyPushdownAutomaton) -> CompositeVisiblyPushdownAutomaton:
        """Alias for :meth:`concat`."""
        return self.concat(other)

    def kleene_star(self) -> CompositeVisiblyPushdownAutomaton:
        """Return a VPA recognizer for the Kleene star of this language."""
        return kleene_star_vpa(self)

    def star(self) -> CompositeVisiblyPushdownAutomaton:
        """Alias for :meth:`kleene_star`."""
        return self.kleene_star()


class CompositeVisiblyPushdownAutomaton(VisiblyPushdownAutomaton):
    """Lazy VPA language expression built from standard closure operations.

    Composite VPAs keep exact language semantics for operations whose concrete
    graph construction would otherwise need a larger normalization pass. They
    still expose the regular VPA membership API through :meth:`recognizes`.
    """

    operation: str
    operands: tuple[VisiblyPushdownAutomaton, ...]

    def __init__(
        self,
        *,
        operation: str,
        operands: Iterable[VisiblyPushdownAutomaton],
    ) -> None:
        operands = tuple(operands)
        if not operands:
            raise ValueError("CompositeVisiblyPushdownAutomaton requires at least one operand")
        call_alphabet, return_alphabet, internal_alphabet = _merge_visible_alphabets(operands)
        super().__init__(
            call_alphabet=call_alphabet,
            return_alphabet=return_alphabet,
            internal_alphabet=internal_alphabet,
            stack_alphabet=frozenset(),
        )
        self.operation = operation
        self.operands = operands

    def validate(self) -> None:
        self._require(
            self.operation in {"union", "intersection", "complement", "difference", "concat", "kleene_star"},
            f"unknown composite VPA operation {self.operation!r}",
        )
        if self.operation in {"complement", "kleene_star"}:
            self._require(len(self.operands) == 1, f"{self.operation} requires one operand")
        elif self.operation in {"difference", "concat"}:
            self._require(len(self.operands) == 2, f"{self.operation} requires two operands")
        else:
            self._require(len(self.operands) >= 2, f"{self.operation} requires at least two operands")
        for operand in self.operands:
            operand.validate()
        call_alphabet, return_alphabet, internal_alphabet = _merge_visible_alphabets(self.operands)
        self._require(call_alphabet == self.call_alphabet, "composite call alphabet is stale")
        self._require(return_alphabet == self.return_alphabet, "composite return alphabet is stale")
        self._require(internal_alphabet == self.internal_alphabet, "composite internal alphabet is stale")

    def recognizes(self, word: Sequence[Any]) -> bool:
        word = tuple(word)
        if any(symbol not in self.input_alphabet for symbol in word):
            return False
        if self.operation == "union":
            return any(operand.recognizes(word) for operand in self.operands)
        if self.operation == "intersection":
            return all(operand.recognizes(word) for operand in self.operands)
        if self.operation == "complement":
            return not self.operands[0].recognizes(word)
        if self.operation == "difference":
            return self.operands[0].recognizes(word) and not self.operands[1].recognizes(word)
        if self.operation == "concat":
            left, right = self.operands
            return any(
                left.recognizes(word[:index]) and right.recognizes(word[index:]) for index in range(len(word) + 1)
            )
        if self.operation == "kleene_star":
            operand = self.operands[0]
            accepted = [False] * (len(word) + 1)
            accepted[0] = True
            for end in range(1, len(word) + 1):
                accepted[end] = any(accepted[start] and operand.recognizes(word[start:end]) for start in range(end))
            return accepted[-1]
        raise ValueError(f"unknown composite VPA operation {self.operation!r}")


def union_vpa(
    left: VisiblyPushdownAutomaton,
    right: VisiblyPushdownAutomaton,
    *rest: VisiblyPushdownAutomaton,
) -> CompositeVisiblyPushdownAutomaton:
    """Return a VPA recognizer for the union of the operands."""
    return CompositeVisiblyPushdownAutomaton(operation="union", operands=(left, right, *rest))


def intersection_vpa(
    left: VisiblyPushdownAutomaton,
    right: VisiblyPushdownAutomaton,
    *rest: VisiblyPushdownAutomaton,
) -> CompositeVisiblyPushdownAutomaton:
    """Return a VPA recognizer for the intersection of the operands."""
    return CompositeVisiblyPushdownAutomaton(operation="intersection", operands=(left, right, *rest))


def complement_vpa(vpa: VisiblyPushdownAutomaton) -> CompositeVisiblyPushdownAutomaton:
    """Return a VPA recognizer for complement over ``vpa``'s visible alphabet."""
    return CompositeVisiblyPushdownAutomaton(operation="complement", operands=(vpa,))


def difference_vpa(
    left: VisiblyPushdownAutomaton,
    right: VisiblyPushdownAutomaton,
) -> CompositeVisiblyPushdownAutomaton:
    """Return a VPA recognizer for ``left`` minus ``right``."""
    return CompositeVisiblyPushdownAutomaton(operation="difference", operands=(left, right))


def concat_vpa(
    left: VisiblyPushdownAutomaton,
    right: VisiblyPushdownAutomaton,
) -> CompositeVisiblyPushdownAutomaton:
    """Return a VPA recognizer for language concatenation."""
    return CompositeVisiblyPushdownAutomaton(operation="concat", operands=(left, right))


def kleene_star_vpa(vpa: VisiblyPushdownAutomaton) -> CompositeVisiblyPushdownAutomaton:
    """Return a VPA recognizer for Kleene star."""
    return CompositeVisiblyPushdownAutomaton(operation="kleene_star", operands=(vpa,))


def _merge_visible_alphabets(
    vpas: Iterable[VisiblyPushdownAutomaton],
) -> tuple[frozenset[Any], frozenset[Any], frozenset[Any]]:
    call_symbols: set[Any] = set()
    return_symbols: set[Any] = set()
    internal_symbols: set[Any] = set()
    owners: dict[Any, str] = {}
    for vpa in vpas:
        for kind, symbols in (
            ("call", vpa.call_alphabet),
            ("return", vpa.return_alphabet),
            ("internal", vpa.internal_alphabet),
        ):
            for symbol in symbols:
                existing = owners.get(symbol)
                if existing is not None and existing != kind:
                    raise ValueError(f"symbol {symbol!r} is both {existing} and {kind}")
                owners[symbol] = kind
                if kind == "call":
                    call_symbols.add(symbol)
                elif kind == "return":
                    return_symbols.add(symbol)
                else:
                    internal_symbols.add(symbol)
    return frozenset(call_symbols), frozenset(return_symbols), frozenset(internal_symbols)


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
        """Copy ``vpa`` into a deterministic VPA and validate determinism."""
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


class CallDrivenAutomaton(DeterministicVisiblyPushdownAutomaton):
    """Deterministic modular VPA whose call target depends only on the call symbol."""

    modules: dict[Hashable, frozenset[Hashable]]
    base_module: Hashable
    call_partition: dict[Any, Hashable]
    call_entries: dict[Any, Hashable]

    def __init__(
        self,
        *,
        modules: Mapping[Hashable, Iterable[Hashable]] | None = None,
        base_module: Hashable = 0,
        call_partition: Mapping[Any, Hashable] | None = None,
        call_entries: Mapping[Any, Hashable] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.modules = _normalize_modules(modules)
        self.base_module = base_module
        self.call_partition = dict(call_partition or {})
        self.call_entries = dict(call_entries or {})

    def validate(self) -> None:
        super().validate()
        state_modules = self._validate_modules()
        self._validate_call_partition()
        self._validate_internal_transitions_stay_in_module(state_modules)
        self._validate_call_driven_transitions()

    def call_entry_map(self) -> dict[Any, Hashable]:
        """Return configured or inferred entries for each call symbol with transitions."""
        entries = dict(self.call_entries)
        for transition in self.transitions():
            if transition.data.get(ATTR_KIND) != KIND_CALL:
                continue
            symbol = transition.data.get(ATTR_SYMBOL)
            if symbol is None:
                continue
            existing = entries.get(symbol, _MISSING)
            if existing is _MISSING:
                entries[symbol] = transition.target
            elif existing != transition.target:
                raise NonDeterministicError(f"call target for {symbol!r} depends on source state")
        return entries

    @classmethod
    def minimize(
        cls,
        vpa: VisiblyPushdownAutomaton,
        *,
        modules: Mapping[Hashable, Iterable[Hashable]] | None = None,
        call_partition: Mapping[Any, Hashable] | None = None,
        base_module: Hashable | None = None,
        call_entries: Mapping[Any, Hashable] | None = None,
    ) -> CallDrivenAutomaton:
        """Return the module-aware deterministic quotient as a CDA."""
        return _minimize_modular_vpa(
            cls,
            vpa,
            modules=modules,
            call_partition=call_partition,
            base_module=base_module,
            call_entries=call_entries,
            entry_states=None,
            form="cda",
        )

    def _validate_modules(self) -> dict[Hashable, Hashable]:
        self._require(bool(self.modules), "modular VPA requires modules")
        self._require(self.base_module in self.modules, "base_module must be present in modules")
        all_states = set(self.states())
        seen: dict[Hashable, Hashable] = {}
        for module, states in self.modules.items():
            self._require(bool(states), f"module {module!r} must contain at least one state")
            for state in states:
                self._require(state in all_states, f"module {module!r} contains unknown state {state!r}")
                self._require(state not in seen, f"state {state!r} appears in multiple modules")
                seen[state] = module
        self._require(set(seen) == all_states, "modules must cover exactly the VPA states")
        if self.initial_state is not None:
            self._require(
                self.initial_state in self.modules[self.base_module],
                "initial_state must lie in the base module",
            )
        return seen

    def _validate_call_partition(self) -> None:
        missing = self.call_alphabet - set(self.call_partition)
        extra = set(self.call_partition) - self.call_alphabet
        self._require(not missing, f"call_partition missing calls {sorted(missing, key=repr)!r}")
        self._require(not extra, f"call_partition contains non-call symbols {sorted(extra, key=repr)!r}")
        for symbol, module in self.call_partition.items():
            self._require(module in self.modules, f"call {symbol!r} targets unknown module {module!r}")

    def _validate_internal_transitions_stay_in_module(self, state_modules: Mapping[Hashable, Hashable]) -> None:
        for transition in self.transitions():
            if transition.data.get(ATTR_KIND) == KIND_INTERNAL:
                self._require(
                    state_modules[transition.source] == state_modules[transition.target],
                    "internal transitions must stay inside one module",
                )

    def _validate_call_driven_transitions(self) -> None:
        entries = self.call_entry_map()
        for symbol, target in entries.items():
            module = self.call_partition[symbol]
            self._require(target in self.modules[module], f"call {symbol!r} entry is not in module {module!r}")
        for transition in self.transitions():
            if transition.data.get(ATTR_KIND) != KIND_CALL:
                continue
            symbol = transition.data.get(ATTR_SYMBOL)
            if symbol is None:
                continue
            self._require(
                transition.target == entries[symbol],
                f"call target for {symbol!r} must be independent of source state",
            )


class MultipleEntryVisiblyPushdownAutomaton(CallDrivenAutomaton):
    """Modular VPA with multiple module entries and source-determined call pushes."""

    entry_states: dict[Hashable, frozenset[Hashable]]

    def __init__(
        self,
        *,
        entry_states: Mapping[Hashable, Iterable[Hashable] | Hashable] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.entry_states = _normalize_multi_entries(entry_states)

    def validate(self) -> None:
        super().validate()
        self._validate_entry_states()
        self._validate_call_targets_are_entries()
        self._validate_source_determined_pushes()

    @classmethod
    def minimize(
        cls,
        vpa: VisiblyPushdownAutomaton,
        *,
        modules: Mapping[Hashable, Iterable[Hashable]] | None = None,
        call_partition: Mapping[Any, Hashable] | None = None,
        base_module: Hashable | None = None,
        entry_states: Mapping[Hashable, Iterable[Hashable] | Hashable] | None = None,
        call_entries: Mapping[Any, Hashable] | None = None,
    ) -> MultipleEntryVisiblyPushdownAutomaton:
        """Return the module-aware deterministic quotient as an MEVPA."""
        return _minimize_modular_vpa(
            cls,
            vpa,
            modules=modules,
            call_partition=call_partition,
            base_module=base_module,
            call_entries=call_entries,
            entry_states=entry_states,
            form="mevpa",
        )

    def _validate_entry_states(self) -> None:
        self._require(bool(self.entry_states), "MEVPA requires entry_states")
        for module, states in self.entry_states.items():
            self._require(module in self.modules, f"entry_states has unknown module {module!r}")
            self._require(bool(states), f"module {module!r} must have at least one entry")
            for state in states:
                self._require(state in self.modules[module], f"entry {state!r} is not in module {module!r}")

    def _validate_call_targets_are_entries(self) -> None:
        entries = self.call_entry_map()
        for symbol, target in entries.items():
            module = self.call_partition[symbol]
            self._require(
                target in self.entry_states.get(module, frozenset()),
                f"call {symbol!r} must enter one of module {module!r}'s entries",
            )

    def _validate_source_determined_pushes(self) -> None:
        pushed_by_source: dict[Hashable, Any] = {}
        for transition in self.transitions():
            if transition.data.get(ATTR_KIND) != KIND_CALL:
                continue
            pushed = transition.data.get(ATTR_STACK_SYMBOL)
            existing = pushed_by_source.get(transition.source, _MISSING)
            if existing is _MISSING:
                pushed_by_source[transition.source] = pushed
            else:
                self._require(existing == pushed, "MEVPA call push must depend only on the source state")


class SingleEntryVisiblyPushdownAutomaton(CallDrivenAutomaton):
    """Modular VPA with one distinguished entry per non-base module."""

    entry_states: dict[Hashable, Hashable]

    def __init__(
        self,
        *,
        entry_states: Mapping[Hashable, Hashable] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.entry_states = dict(entry_states or {})

    def validate(self) -> None:
        super().validate()
        self._validate_single_entries()
        self._validate_single_entry_calls()

    @classmethod
    def minimize(
        cls,
        vpa: VisiblyPushdownAutomaton,
        *,
        call_partition: Mapping[Any, Hashable] | None = None,
        modules: Mapping[Hashable, Iterable[Hashable]] | None = None,
        base_module: Hashable | None = None,
        entry_states: Mapping[Hashable, Hashable] | None = None,
        call_entries: Mapping[Any, Hashable] | None = None,
    ) -> SingleEntryVisiblyPushdownAutomaton:
        """Return the module-aware deterministic quotient as an SEVPA.

        A fixed call partition and module structure are required. General VPA
        minimization is intentionally not attempted here.
        """
        return _minimize_modular_vpa(
            cls,
            vpa,
            modules=modules,
            call_partition=call_partition,
            base_module=base_module,
            call_entries=call_entries,
            entry_states=entry_states,
            form="sevpa",
        )

    def _validate_single_entries(self) -> None:
        missing = set(self.modules) - {self.base_module} - set(self.entry_states)
        self._require(not missing, f"SEVPA missing entries for modules {sorted(missing, key=repr)!r}")
        for module, state in self.entry_states.items():
            self._require(module in self.modules, f"entry_states has unknown module {module!r}")
            self._require(state in self.modules[module], f"entry {state!r} is not in module {module!r}")

    def _validate_single_entry_calls(self) -> None:
        for transition in self.transitions():
            if transition.data.get(ATTR_KIND) != KIND_CALL:
                continue
            symbol = transition.data.get(ATTR_SYMBOL)
            if symbol is None:
                continue
            module = self.call_partition[symbol]
            expected_entry = self.entry_states.get(module)
            self._require(
                transition.target == expected_entry,
                f"SEVPA call {symbol!r} must enter module {module!r}'s single entry",
            )
            self._require(
                transition.data.get(ATTR_STACK_SYMBOL) == (transition.source, symbol),
                "SEVPA call stack symbols must be (caller_state, call_symbol)",
            )


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
        """Build the Myhill-Nerode canonical deterministic VPA for ``vpa``.

        The construction is finite for deterministic VPAs because states are
        summary classes of well-matched factors. With an empty call alphabet,
        this specializes to the usual minimal DFA right-congruence construction.
        """
        det = DeterministicVisiblyPushdownAutomaton.from_vpa(vpa)
        algebra = _SummaryAlgebra.from_vpa(det)
        return algebra.to_canonical_vpa(cls, det)

    @classmethod
    def minimize(cls, vpa: VisiblyPushdownAutomaton) -> CanonicalVisiblyPushdownAutomaton:
        """Alias for :meth:`from_vpa`."""
        return cls.from_vpa(vpa)


def _normalize_modules(modules: Mapping[Hashable, Iterable[Hashable]] | None) -> dict[Hashable, frozenset[Hashable]]:
    if modules is None:
        return {}
    return {module: frozenset(states) for module, states in modules.items()}


def _normalize_multi_entries(
    entry_states: Mapping[Hashable, Iterable[Hashable] | Hashable] | None,
) -> dict[Hashable, frozenset[Hashable]]:
    if entry_states is None:
        return {}
    result: dict[Hashable, frozenset[Hashable]] = {}
    for module, states in entry_states.items():
        if isinstance(states, frozenset | set | list):
            result[module] = frozenset(states)
        else:
            result[module] = frozenset({states})
    return result


def _metadata_or_argument(vpa: VisiblyPushdownAutomaton, name: str, value: Any, default: Any) -> Any:
    if value is not None:
        return value
    return getattr(vpa, name, default)


def _deterministic_view(vpa: VisiblyPushdownAutomaton) -> DeterministicVisiblyPushdownAutomaton:
    return DeterministicVisiblyPushdownAutomaton.from_vpa(vpa)


def _minimize_modular_vpa(
    target_cls: type[CallDrivenAutomaton],
    vpa: VisiblyPushdownAutomaton,
    *,
    modules: Mapping[Hashable, Iterable[Hashable]] | None,
    call_partition: Mapping[Any, Hashable] | None,
    base_module: Hashable | None,
    call_entries: Mapping[Any, Hashable] | None,
    entry_states: Mapping[Hashable, Any] | None,
    form: str,
) -> Any:
    modules = _metadata_or_argument(vpa, "modules", modules, None)
    call_partition = _metadata_or_argument(vpa, "call_partition", call_partition, None)
    base_module = _metadata_or_argument(vpa, "base_module", base_module, 0)
    call_entries = _metadata_or_argument(vpa, "call_entries", call_entries, None)
    entry_states = _metadata_or_argument(vpa, "entry_states", entry_states, None)

    if modules is None or call_partition is None:
        raise NotImplementedError("modular VPA minimization requires fixed modules and call_partition")

    det = _deterministic_view(vpa)
    modules = _normalize_modules(modules)
    call_partition = dict(call_partition)
    call_entries = dict(call_entries or _infer_call_entries(det, call_partition))
    entry_states = _infer_entry_states(form, modules, base_module, call_partition, call_entries, entry_states, det)

    source = _construct_modular_view(
        target_cls,
        det,
        modules=modules,
        base_module=base_module,
        call_partition=call_partition,
        call_entries=call_entries,
        entry_states=entry_states,
    )
    source.validate()

    partition = _refine_modular_partition(source, modules)
    return _quotient_modular_vpa(
        target_cls,
        source,
        partition,
        modules=modules,
        base_module=base_module,
        call_partition=call_partition,
        call_entries=call_entries,
        entry_states=entry_states,
        form=form,
    )


def _construct_modular_view(
    target_cls: type[CallDrivenAutomaton],
    det: DeterministicVisiblyPushdownAutomaton,
    *,
    modules: Mapping[Hashable, frozenset[Hashable]],
    base_module: Hashable,
    call_partition: Mapping[Any, Hashable],
    call_entries: Mapping[Any, Hashable],
    entry_states: Mapping[Hashable, Any],
) -> CallDrivenAutomaton:
    kwargs = {
        "input_alphabet": det.input_alphabet,
        "call_alphabet": det.call_alphabet,
        "return_alphabet": det.return_alphabet,
        "internal_alphabet": det.internal_alphabet,
        "stack_alphabet": det.stack_alphabet,
        "bottom_stack_symbol": det.bottom_stack_symbol,
        "initial_state": det.initial_state,
        "accepting_states": det.accepting_states,
        "graph": det.graph.copy(),
        "modules": modules,
        "base_module": base_module,
        "call_partition": call_partition,
        "call_entries": call_entries,
    }
    if issubclass(target_cls, (SingleEntryVisiblyPushdownAutomaton, MultipleEntryVisiblyPushdownAutomaton)):
        kwargs["entry_states"] = entry_states
    return target_cls(**kwargs)


def _infer_call_entries(
    det: DeterministicVisiblyPushdownAutomaton,
    call_partition: Mapping[Any, Hashable],
) -> dict[Any, Hashable]:
    entries: dict[Any, Hashable] = {}
    for (_source, symbol), (target, _stack) in det.call_transition_map().items():
        if symbol not in call_partition:
            raise NotImplementedError(f"call_partition missing call symbol {symbol!r}")
        existing = entries.get(symbol, _MISSING)
        if existing is _MISSING:
            entries[symbol] = target
        elif existing != target:
            raise NotImplementedError(f"call target for {symbol!r} depends on source state")
    return entries


def _infer_entry_states(
    form: str,
    modules: Mapping[Hashable, frozenset[Hashable]],
    base_module: Hashable,
    call_partition: Mapping[Any, Hashable],
    call_entries: Mapping[Any, Hashable],
    entry_states: Mapping[Hashable, Any] | None,
    det: DeterministicVisiblyPushdownAutomaton,
) -> dict[Hashable, Any]:
    if entry_states is not None:
        if form == "mevpa":
            return _normalize_multi_entries(entry_states)
        return dict(entry_states)

    by_module: dict[Hashable, set[Hashable]] = {module: set() for module in modules}
    if det.initial_state is not None and base_module in by_module:
        by_module[base_module].add(det.initial_state)
    for symbol, entry in call_entries.items():
        by_module[call_partition[symbol]].add(entry)

    if form == "mevpa":
        return {module: frozenset(states) for module, states in by_module.items() if states}
    if form == "sevpa":
        result: dict[Hashable, Hashable] = {}
        for module, states in by_module.items():
            if module == base_module:
                continue
            if len(states) != 1:
                raise NotImplementedError("SEVPA minimization requires one inferred entry per non-base module")
            result[module] = next(iter(states))
        return result
    return {}


def _refine_modular_partition(
    vpa: CallDrivenAutomaton,
    modules: Mapping[Hashable, frozenset[Hashable]],
) -> list[frozenset[Hashable]]:
    partition: list[frozenset[Hashable]] = []
    for _module, states in sorted(modules.items(), key=lambda item: repr(item[0])):
        accepting = frozenset(states & vpa.accepting_states)
        rejecting = frozenset(states - vpa.accepting_states)
        if accepting:
            partition.append(accepting)
        if rejecting:
            partition.append(rejecting)

    changed = True
    while changed:
        changed = False
        block_of = _block_map(partition)
        context_groups = _stack_context_groups(vpa, block_of)
        new_partition: list[frozenset[Hashable]] = []
        for block in partition:
            pieces: dict[tuple[Any, ...], set[Hashable]] = {}
            for state in block:
                signature = _modular_state_signature(vpa, state, block_of, context_groups)
                pieces.setdefault(signature, set()).add(state)
            if len(pieces) > 1:
                changed = True
            new_partition.extend(frozenset(piece) for piece in pieces.values())
        partition = new_partition
    return partition


def _block_map(partition: Sequence[frozenset[Hashable]]) -> dict[Hashable, frozenset[Hashable]]:
    return {state: block for block in partition for state in block}


def _stack_context_groups(
    vpa: DeterministicVisiblyPushdownAutomaton,
    block_of: Mapping[Hashable, Hashable],
) -> list[tuple[Any, tuple[Any, ...]]]:
    stack_symbols = {stack for _key, (_target, stack) in vpa.call_transition_map().items()}
    if vpa.bottom_stack_symbol is not None:
        stack_symbols.add(vpa.bottom_stack_symbol)
    grouped: dict[Any, set[Any]] = {}
    for stack_symbol in stack_symbols:
        canonical = _canonical_stack_symbol(stack_symbol, block_of)
        grouped.setdefault(canonical, set()).add(stack_symbol)
    return [(canonical, tuple(sorted(actuals, key=repr))) for canonical, actuals in sorted(grouped.items(), key=repr)]


def _modular_state_signature(
    vpa: CallDrivenAutomaton,
    state: Hashable,
    block_of: Mapping[Hashable, Hashable],
    context_groups: Sequence[tuple[Any, tuple[Any, ...]]],
) -> tuple[Any, ...]:
    state_modules = {state: module for module, states in vpa.modules.items() for state in states}
    call_map = vpa.call_transition_map()
    internal_map = vpa.internal_transition_map()
    return_map = vpa.return_transition_map()

    internal = tuple(
        (
            symbol,
            None if (target := internal_map.get((state, symbol))) is None else block_of[target],
        )
        for symbol in sorted(vpa.internal_alphabet, key=repr)
    )
    calls = tuple(
        (
            symbol,
            None
            if (call := call_map.get((state, symbol))) is None
            else (block_of[call[0]], _canonical_stack_symbol(call[1], block_of)),
        )
        for symbol in sorted(vpa.call_alphabet, key=repr)
    )
    returns = []
    for symbol in sorted(vpa.return_alphabet, key=repr):
        for canonical_stack, actual_stacks in context_groups:
            targets = []
            for stack_symbol in actual_stacks:
                target = return_map.get((state, symbol, stack_symbol), return_map.get((state, symbol, None)))
                targets.append(None if target is None else block_of[target])
            returns.append((symbol, canonical_stack, tuple(sorted(set(targets), key=repr))))

    return (
        state_modules[state],
        state in vpa.accepting_states,
        internal,
        calls,
        tuple(returns),
    )


def _quotient_modular_vpa(
    target_cls: type[CallDrivenAutomaton],
    source: CallDrivenAutomaton,
    partition: Sequence[frozenset[Hashable]],
    *,
    modules: Mapping[Hashable, frozenset[Hashable]],
    base_module: Hashable,
    call_partition: Mapping[Any, Hashable],
    call_entries: Mapping[Any, Hashable],
    entry_states: Mapping[Hashable, Any],
    form: str,
) -> Any:
    block_of = _block_map(partition)
    blocks = list(partition)
    stack_alphabet: set[Any] = set()
    if source.bottom_stack_symbol is not None:
        stack_alphabet.add(source.bottom_stack_symbol)
    transitions: set[tuple[Hashable, Hashable, Any, Any, Any | None]] = set()
    stack_symbol_rewrite: dict[Any, set[Any]] = {}

    for transition in source.transitions():
        if transition.data.get(ATTR_KIND) != KIND_CALL:
            continue
        symbol = transition.data.get(ATTR_SYMBOL)
        stack_symbol = transition.data.get(ATTR_STACK_SYMBOL)
        quotient_stack_symbol = _quotient_call_stack_symbol(
            form,
            block_of[transition.source],
            symbol,
            stack_symbol,
            block_of,
        )
        stack_symbol_rewrite.setdefault(stack_symbol, set()).add(quotient_stack_symbol)

    for block in blocks:
        representative = min(block, key=repr)
        source_block = block_of[representative]
        for transition in source.graph.out_transitions(representative):
            kind = transition.data.get(ATTR_KIND)
            symbol = transition.data.get(ATTR_SYMBOL)
            target_block = block_of[transition.target]
            stack_symbol = transition.data.get(ATTR_STACK_SYMBOL)
            if kind == KIND_CALL:
                stack_symbol = _quotient_call_stack_symbol(form, source_block, symbol, stack_symbol, block_of)
                stack_alphabet.add(stack_symbol)
            elif kind == KIND_RETURN and stack_symbol is not None:
                rewritten = stack_symbol_rewrite.get(stack_symbol, {_canonical_stack_symbol(stack_symbol, block_of)})
                for rewritten_stack_symbol in rewritten:
                    stack_alphabet.add(rewritten_stack_symbol)
                    transitions.add((source_block, target_block, kind, symbol, rewritten_stack_symbol))
                continue
            transitions.add((source_block, target_block, kind, symbol, stack_symbol))

    quotient_modules = {
        module: frozenset(block for block in blocks if block & states)
        for module, states in modules.items()
        if any(block & states for block in blocks)
    }
    quotient_call_entries = {symbol: block_of[state] for symbol, state in call_entries.items() if state in block_of}
    quotient_entry_states = _quotient_entry_states(form, entry_states, block_of)

    kwargs = {
        "input_alphabet": source.input_alphabet,
        "call_alphabet": source.call_alphabet,
        "return_alphabet": source.return_alphabet,
        "internal_alphabet": source.internal_alphabet,
        "stack_alphabet": frozenset(stack_alphabet),
        "bottom_stack_symbol": source.bottom_stack_symbol,
        "initial_state": block_of[source.initial_state],
        "accepting_states": frozenset(block for block in blocks if block & source.accepting_states),
        "modules": quotient_modules,
        "base_module": base_module,
        "call_partition": dict(call_partition),
        "call_entries": quotient_call_entries,
    }
    if issubclass(target_cls, (SingleEntryVisiblyPushdownAutomaton, MultipleEntryVisiblyPushdownAutomaton)):
        kwargs["entry_states"] = quotient_entry_states

    result = target_cls(**kwargs)
    for block in blocks:
        result.graph.add_state(block)
    for source_block, target_block, kind, symbol, stack_symbol in sorted(transitions, key=repr):
        if kind == KIND_CALL:
            result.add_call_transition(source_block, target_block, symbol, stack_symbol)
        elif kind == KIND_RETURN:
            result.add_return_transition(source_block, target_block, symbol, stack_symbol)
        elif kind == KIND_INTERNAL:
            result.add_internal_transition(source_block, target_block, symbol)
    result.validate()
    return result


def _quotient_call_stack_symbol(
    form: str,
    source_block: Hashable,
    symbol: Any,
    stack_symbol: Any,
    block_of: Mapping[Hashable, Hashable],
) -> Any:
    if form == "sevpa":
        return (source_block, symbol)
    if form == "mevpa":
        return source_block
    return _canonical_stack_symbol(stack_symbol, block_of)


def _quotient_entry_states(
    form: str,
    entry_states: Mapping[Hashable, Any],
    block_of: Mapping[Hashable, Hashable],
) -> dict[Hashable, Any]:
    if form == "mevpa":
        normalized = _normalize_multi_entries(entry_states)
        return {
            module: frozenset(block_of[state] for state in states if state in block_of)
            for module, states in normalized.items()
        }
    if form == "sevpa":
        return {module: block_of[state] for module, state in entry_states.items() if state in block_of}
    return {}


def _canonical_stack_symbol(stack_symbol: Any, block_of: Mapping[Hashable, Hashable]) -> Any:
    if stack_symbol in block_of:
        return block_of[stack_symbol]
    if isinstance(stack_symbol, tuple):
        return tuple(_canonical_stack_symbol(part, block_of) for part in stack_symbol)
    return stack_symbol


class _SummaryAlgebra:
    def __init__(
        self,
        *,
        state_order: tuple[Hashable, ...],
        summaries: frozenset[tuple[int | None, ...]],
        identity: tuple[int | None, ...],
        internal_summaries: Mapping[Any, tuple[int | None, ...]],
        class_of: Mapping[tuple[int | None, ...], int],
        representatives: Mapping[int, tuple[int | None, ...]],
    ) -> None:
        self.state_order = state_order
        self.summaries = summaries
        self.identity = identity
        self.internal_summaries = dict(internal_summaries)
        self.class_of = dict(class_of)
        self.representatives = dict(representatives)

    @classmethod
    def from_vpa(cls, vpa: DeterministicVisiblyPushdownAutomaton) -> _SummaryAlgebra:
        state_order = tuple(sorted(vpa.states(), key=repr))
        state_index = {state: index for index, state in enumerate(state_order)}
        internal_summaries = {
            symbol: _internal_summary(vpa, state_order, state_index, symbol)
            for symbol in sorted(vpa.internal_alphabet, key=repr)
        }
        identity = tuple(range(len(state_order)))
        summaries = _close_summary_algebra(vpa, state_order, state_index, identity, internal_summaries)
        class_of, representatives = _quotient_summaries(vpa, state_order, state_index, summaries, identity)
        return cls(
            state_order=state_order,
            summaries=frozenset(summaries),
            identity=identity,
            internal_summaries=internal_summaries,
            class_of=class_of,
            representatives=representatives,
        )

    def to_canonical_vpa(
        self,
        cls: type[CanonicalVisiblyPushdownAutomaton],
        source: DeterministicVisiblyPushdownAutomaton,
    ) -> CanonicalVisiblyPushdownAutomaton:
        identity_class = self.class_of[self.identity]
        states = frozenset(self.representatives)
        stack_alphabet = frozenset(
            (summary_class, symbol) for summary_class in states for symbol in source.call_alphabet
        )
        accepting_states = frozenset(
            summary_class
            for summary_class, summary in self.representatives.items()
            if _summary_accepts(source, self.state_order, summary)
        )
        result = cls(
            input_alphabet=source.input_alphabet,
            call_alphabet=source.call_alphabet,
            return_alphabet=source.return_alphabet,
            internal_alphabet=source.internal_alphabet,
            stack_alphabet=stack_alphabet,
            bottom_stack_symbol=None,
            initial_state=identity_class,
            accepting_states=accepting_states,
            summary_representatives=self.representatives,
        )
        for state in states:
            result.graph.add_state(state)

        for summary_class, summary in sorted(self.representatives.items(), key=repr):
            for symbol, internal in sorted(self.internal_summaries.items(), key=lambda item: repr(item[0])):
                target_summary = _compose_summary(summary, internal)
                result.add_internal_transition(summary_class, self.class_of[target_summary], symbol)
            for symbol in sorted(source.call_alphabet, key=repr):
                result.add_call_transition(summary_class, identity_class, symbol, (summary_class, symbol))

        for inner_class, inner in sorted(self.representatives.items(), key=repr):
            for outer_class, outer in sorted(self.representatives.items(), key=repr):
                for call_symbol in sorted(source.call_alphabet, key=repr):
                    for return_symbol in sorted(source.return_alphabet, key=repr):
                        wrapped = _wrap_summary(source, self.state_order, inner, call_symbol, return_symbol)
                        target = _compose_summary(outer, wrapped)
                        result.add_return_transition(
                            inner_class,
                            self.class_of[target],
                            return_symbol,
                            (outer_class, call_symbol),
                        )

        result.validate()
        return result


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


def _quotient_summaries(
    vpa: DeterministicVisiblyPushdownAutomaton,
    state_order: Sequence[Hashable],
    state_index: Mapping[Hashable, int],
    summaries: set[tuple[int | None, ...]],
    identity: tuple[int | None, ...],
) -> tuple[dict[tuple[int | None, ...], int], dict[int, tuple[int | None, ...]]]:
    contexts = tuple(sorted(summaries, key=repr))
    signatures = {
        summary: tuple(_summary_accepts(vpa, state_order, _compose_summary(summary, context)) for context in contexts)
        for summary in summaries
    }
    identity_signature = signatures[identity]
    ordered_signatures = sorted(
        set(signatures.values()), key=lambda signature: (signature != identity_signature, signature)
    )
    signature_class = {signature: index for index, signature in enumerate(ordered_signatures)}
    class_of = {summary: signature_class[signature] for summary, signature in signatures.items()}
    representatives = {
        index: min(
            (summary for summary, signature in signatures.items() if signature_class[signature] == index), key=repr
        )
        for index in signature_class.values()
    }
    return class_of, representatives


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
