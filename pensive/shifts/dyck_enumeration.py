"""Canonical strings and enumeration for small Sofic-Dyck topologies."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from itertools import product
from typing import Any

from pensive.exceptions import PensiveValidationError
from pensive.graph import ATTR_KIND, ATTR_SYMBOL, KIND_CALL, KIND_INTERNAL, KIND_RETURN
from pensive.shifts.dyck_algorithms import is_admissible_word
from pensive.shifts.sofic_dyck import SoficDyckShift, TransitionRef, transition_ref

__all__ = [
    "DyckGraphString",
    "count_dyck_graph_strings",
    "dyck_graph_string_to_shift",
    "iter_dyck_graph_strings",
    "iter_sofic_dyck_topologies",
    "shift_to_dyck_graph_string",
]

_KIND_CALL = 0
_KIND_RETURN = 1
_KIND_INTERNAL = 2


class DyckEnumerationError(PensiveValidationError):
    """Raised when Dyck-graph enumeration fails."""


@dataclass(frozen=True, slots=True)
class DyckGraphString:
    """Canonical encoding of a small matched-edge Dyck graph.

    ``transitions`` lists ``(source, kind, symbol_index, target)`` in
    lexicographic order. ``matched_pairs`` lists ``(call_transition_index,
    return_transition_index)`` into ``transitions``.
    """

    n: int
    call_symbols: tuple[Any, ...]
    return_symbols: tuple[Any, ...]
    internal_symbols: tuple[Any, ...]
    transitions: tuple[tuple[int, int, int, int], ...]
    matched_pairs: tuple[tuple[int, int], ...]


def shift_to_dyck_graph_string(shift: SoficDyckShift) -> DyckGraphString:
    """Serialize a ``SoficDyckShift`` into canonical edge-index form."""
    states = sorted(shift.states(), key=repr)
    state_index = {state: index for index, state in enumerate(states)}
    call_symbols = tuple(sorted(shift.call_alphabet, key=repr))
    return_symbols = tuple(sorted(shift.return_alphabet, key=repr))
    internal_symbols = tuple(sorted(shift.internal_alphabet, key=repr))
    call_index = {symbol: index for index, symbol in enumerate(call_symbols)}
    return_index = {symbol: index for index, symbol in enumerate(return_symbols)}
    internal_index = {symbol: index for index, symbol in enumerate(internal_symbols)}

    encoded: list[tuple[int, int, int, int]] = []
    call_positions: dict[TransitionRef, int] = {}
    return_positions: dict[TransitionRef, int] = {}

    for transition in sorted(shift.transitions(), key=lambda item: (repr(item.source), repr(item.target), item.key)):
        kind = transition.data.get(ATTR_KIND)
        symbol = transition.data.get(ATTR_SYMBOL)
        if kind == KIND_CALL:
            kind_code = _KIND_CALL
            symbol_index = call_index[symbol]
        elif kind == KIND_RETURN:
            kind_code = _KIND_RETURN
            symbol_index = return_index[symbol]
        elif kind == KIND_INTERNAL:
            kind_code = _KIND_INTERNAL
            symbol_index = internal_index[symbol]
        else:
            raise DyckEnumerationError(f"unknown edge kind {kind!r}")
        encoded.append((state_index[transition.source], kind_code, symbol_index, state_index[transition.target]))
        ref = transition_ref(transition)
        position = len(encoded) - 1
        if kind == KIND_CALL:
            call_positions[ref] = position
        elif kind == KIND_RETURN:
            return_positions[ref] = position

    matched = tuple(
        sorted((call_positions[call_ref], return_positions[return_ref]) for call_ref, return_ref in shift.matched_edges)
    )

    return DyckGraphString(
        n=len(states),
        call_symbols=call_symbols,
        return_symbols=return_symbols,
        internal_symbols=internal_symbols,
        transitions=tuple(encoded),
        matched_pairs=tuple(sorted(matched)),
    )


def dyck_graph_string_to_shift(spec: DyckGraphString) -> SoficDyckShift:
    """Build a ``SoficDyckShift`` from a ``DyckGraphString``."""
    shift = SoficDyckShift(
        call_alphabet=frozenset(spec.call_symbols),
        return_alphabet=frozenset(spec.return_symbols),
        internal_alphabet=frozenset(spec.internal_symbols),
    )
    states = [f"q{index}" for index in range(spec.n)]
    for state in states:
        shift.graph.add_state(state)

    edge_refs: list[TransitionRef] = []
    for source, kind_code, symbol_index, target in spec.transitions:
        source_state = states[source]
        target_state = states[target]
        if kind_code == _KIND_CALL:
            symbol = spec.call_symbols[symbol_index]
            edge_refs.append(shift.add_call_transition(source_state, target_state, symbol))
        elif kind_code == _KIND_RETURN:
            symbol = spec.return_symbols[symbol_index]
            edge_refs.append(shift.add_return_transition(source_state, target_state, symbol))
        else:
            symbol = spec.internal_symbols[symbol_index]
            edge_refs.append(shift.add_internal_transition(source_state, target_state, symbol))

    for call_index, return_index in spec.matched_pairs:
        shift.add_matched_pair(edge_refs[call_index], edge_refs[return_index])

    shift.validate()
    return shift


def _one_state_transition_choices(
  symbols_per_kind: Sequence[int],
  *,
  include_empty: bool,
) -> Iterator[tuple[tuple[int, int, int, int], ...]]:
    """Enumerate transition subsets on a single state looping to itself."""
    slots: list[tuple[int, int, int, int]] = []
    for kind_code, symbol_count in enumerate(symbols_per_kind):
        for symbol_index in range(symbol_count):
            slots.append((0, kind_code, symbol_index, 0))
    if not slots:
        if include_empty:
            yield ()
        return
    for mask in range(1 if not include_empty else 0, 1 << len(slots)):
        selected = tuple(slots[index] for index in range(len(slots)) if mask & (1 << index))
        if selected or include_empty:
            yield selected


def iter_dyck_graph_strings(
    *,
    n: int = 1,
    call_symbols: Sequence[Any] = ("a",),
    return_symbols: Sequence[Any] = ("A",),
    internal_symbols: Sequence[Any] = (),
    include_empty: bool = False,
) -> Iterator[DyckGraphString]:
    """Yield canonical Dyck-graph strings for small topologies."""
    if n != 1:
        raise DyckEnumerationError("iter_dyck_graph_strings currently supports n=1 only")
    calls = tuple(call_symbols)
    returns = tuple(return_symbols)
    internals = tuple(internal_symbols)
    if len(calls) != len(returns):
        raise DyckEnumerationError("call and return symbol counts must match")

    for transitions in _one_state_transition_choices(
        (len(calls), len(returns), len(internals)),
        include_empty=include_empty,
    ):
        call_edge_indices = [index for index, item in enumerate(transitions) if item[1] == _KIND_CALL]
        return_edge_indices = [index for index, item in enumerate(transitions) if item[1] == _KIND_RETURN]
        call_by_symbol: dict[int, list[int]] = defaultdict(list)
        return_by_symbol: dict[int, list[int]] = defaultdict(list)
        for edge_index in call_edge_indices:
            call_by_symbol[transitions[edge_index][2]].append(edge_index)
        for edge_index in return_edge_indices:
            return_by_symbol[transitions[edge_index][2]].append(edge_index)

        matched_options: list[list[tuple[int, int]]] = []
        for symbol_index in call_by_symbol:
            if symbol_index not in return_by_symbol:
                matched_options = []
                break
            pairs = [
                (call_index, return_index)
                for call_index, return_index in zip(
                    call_by_symbol[symbol_index],
                    return_by_symbol[symbol_index],
                    strict=False,
                )
            ]
            if not pairs:
                matched_options = []
                break
            matched_options.append(pairs)
        if call_by_symbol and not matched_options:
            continue

        if not matched_options:
            yield DyckGraphString(
                n=1,
                call_symbols=calls,
                return_symbols=returns,
                internal_symbols=internals,
                transitions=transitions,
                matched_pairs=(),
            )
            continue

        for combo in product(*matched_options):
            pairs: list[tuple[int, int]] = []
            for item in combo:
                if isinstance(item, tuple) and len(item) == 2:
                    pairs.append(item)
            matched = tuple(sorted(set(pairs)))
            yield DyckGraphString(
                n=1,
                call_symbols=calls,
                return_symbols=returns,
                internal_symbols=internals,
                transitions=transitions,
                matched_pairs=matched,
            )


def iter_sofic_dyck_topologies(
    *,
    n: int = 1,
    call_symbols: Sequence[Any] = ("a",),
    return_symbols: Sequence[Any] = ("A",),
    internal_symbols: Sequence[Any] = (),
) -> Iterator[SoficDyckShift]:
    """Yield valid ``SoficDyckShift`` topologies from canonical Dyck-graph strings."""
    seen: set[tuple[Any, ...]] = set()
    for spec in iter_dyck_graph_strings(
        n=n,
        call_symbols=call_symbols,
        return_symbols=return_symbols,
        internal_symbols=internal_symbols,
    ):
        try:
            shift = dyck_graph_string_to_shift(spec)
        except (PensiveValidationError, ValueError):
            continue
        if not is_admissible_word(shift, ()):
            continue
        key = (spec.transitions, spec.matched_pairs)
        if key in seen:
            continue
        seen.add(key)
        yield shift


def count_dyck_graph_strings(
    *,
    n: int = 1,
    call_symbols: Sequence[Any] = ("a",),
    return_symbols: Sequence[Any] = ("A",),
    internal_symbols: Sequence[Any] = (),
) -> int:
    """Count canonical Dyck-graph strings for the given signature."""
    return sum(
        1
        for _spec in iter_dyck_graph_strings(
            n=n,
            call_symbols=call_symbols,
            return_symbols=return_symbols,
            internal_symbols=internal_symbols,
        )
    )
