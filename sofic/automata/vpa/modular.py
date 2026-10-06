"""Modular (call-driven, single- and multiple-entry) visibly pushdown automata."""

from __future__ import annotations

from collections.abc import Hashable, Iterable, Mapping, Sequence
from typing import Any

from sofic.automata.vpa import operations as vc
from sofic.automata.vpa.base import _MISSING, VisiblyPushdownAutomaton
from sofic.automata.vpa.deterministic import DeterministicVisiblyPushdownAutomaton
from sofic.exceptions import NonDeterministicError
from sofic.graph import (
    ATTR_KIND,
    ATTR_STACK_SYMBOL,
    ATTR_SYMBOL,
    KIND_CALL,
    KIND_INTERNAL,
    KIND_RETURN,
)


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

    if modules is None:
        convert = vc.to_multiple_entry if form == "mevpa" else vc.to_single_entry
        converted = convert(vpa, call_partition)
        return _minimize_modular_vpa(
            target_cls,
            converted,
            modules=converted.modules,
            call_partition=converted.call_partition,
            base_module=converted.base_module,
            call_entries=converted.call_entries,
            entry_states=getattr(converted, "entry_states", None) if form != "cda" else None,
            form=form,
        )
    if call_partition is None:
        raise ValueError("a call_partition is required when modules are given")

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
            raise ValueError(f"call_partition missing call symbol {symbol!r}")
        existing = entries.get(symbol, _MISSING)
        if existing is _MISSING:
            entries[symbol] = target
        elif existing != target:
            raise ValueError(
                f"call target for {symbol!r} depends on the source state; omit modules to convert the VPA first"
            )
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
                raise ValueError(
                    "SEVPA minimization requires one entry per non-base module; omit modules to convert the VPA first"
                )
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
