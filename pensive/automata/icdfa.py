"""Enumeration and canonical strings for initially-connected DFAs (ICDFAs).

Implements the string representation and exact generation algorithm of
Almeida, Moreira, and Reis (2007), *Enumeration and generation with a string
automata representation*, Theoretical Computer Science 387(2):93--102.
DOI: 10.1016/j.tcs.2007.07.029.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Hashable, Iterator, Sequence
from dataclasses import dataclass
from math import comb
from typing import Any

from pensive.automata.dfa import DFA
from pensive.exceptions import PensiveValidationError

__all__ = [
    "ICDFAString",
    "count_flag_sequences",
    "count_icdfa",
    "count_icdfa_empty",
    "dfa_to_icdfa_string",
    "first_icdfa_empty_string",
    "flags_from_string",
    "icdfa_string_to_dfa",
    "iter_icdfa",
    "iter_icdfa_empty_strings",
    "last_icdfa_empty_string",
    "next_flags",
    "next_icdfa_empty_string",
    "string_from_flags",
    "validate_icdfa_empty_string",
]


class ICDFAEnumerationError(PensiveValidationError):
    """Raised when ICDFA enumeration or string conversion fails."""


@dataclass(frozen=True, slots=True)
class ICDFAString:
    """Canonical transition string for an ICDFA structure (no final states)."""

    transitions: tuple[int, ...]
    n: int
    k: int

    def __post_init__(self) -> None:
        if self.n < 1 or self.k < 1:
            raise ValueError("n and k must be positive")
        if len(self.transitions) != self.k * self.n:
            raise ValueError(f"expected {self.k * self.n} transitions, got {len(self.transitions)}")


def validate_icdfa_empty_string(
    transitions: Sequence[int],
    *,
    n: int,
    k: int,
) -> None:
    """Validate that ``transitions`` satisfies rules R1 and R2."""
    length = k * n
    if len(transitions) != length:
        raise ICDFAEnumerationError(f"expected length {length}, got {len(transitions)}")
    for value in transitions:
        if not 0 <= value < n:
            raise ICDFAEnumerationError(f"transition value {value!r} not in [0, {n - 1}]")

    for state in range(2, n):
        first_index = next(i for i, value in enumerate(transitions) if value == state)
        if not any(transitions[j] == state - 1 for j in range(first_index)):
            raise ICDFAEnumerationError(f"state {state} at index {first_index} appears before state {state - 1}")

    for state in range(1, n):
        if state not in transitions[: k * state]:
            raise ICDFAEnumerationError(f"state {state} does not appear in the first {k * state} symbols")


def flags_from_string(transitions: Sequence[int], *, n: int) -> tuple[int, ...]:
    """Return first-occurrence indices ``(f_1, …, f_{n-1})`` for ``transitions``."""
    if n <= 1:
        return ()
    flags: list[int] = []
    for state in range(1, n):
        try:
            flags.append(transitions.index(state))
        except ValueError as exc:
            raise ICDFAEnumerationError(f"missing first occurrence of state {state}") from exc
    return tuple(flags)


def _validate_flags(flags: Sequence[int], *, n: int, k: int) -> None:
    if len(flags) != n - 1:
        raise ICDFAEnumerationError(f"expected {n - 1} flags, got {len(flags)}")
    if not 0 <= flags[0] < k:
        raise ICDFAEnumerationError(f"f_1={flags[0]} not in [0, {k - 1}]")
    for index in range(1, len(flags)):
        lower = flags[index - 1]
        upper = k * (index + 1) - 1
        if not lower < flags[index] <= upper:
            raise ICDFAEnumerationError(f"flag f_{index + 1}={flags[index]} not in ({lower}, {upper}]")


def string_from_flags(
    flags: Sequence[int],
    *,
    n: int,
    k: int,
    filler: int = 0,
) -> tuple[int, ...]:
    """Build the first ICDFA∅ string for a valid flag sequence."""
    _validate_flags(flags, n=n, k=k)
    transitions = [filler] * (k * n)
    for state, flag in enumerate(flags, start=1):
        transitions[flag] = state
    return tuple(transitions)


def first_icdfa_empty_string(*, n: int, k: int) -> tuple[int, ...]:
    """Return the first ICDFA∅ string in generation order."""
    if n == 1:
        return (0,) * k
    flags = tuple(k * state - 1 for state in range(1, n))
    return string_from_flags(flags, n=n, k=k)


def last_icdfa_empty_string(*, n: int, k: int) -> tuple[int, ...]:
    """Return the last ICDFA∅ string in generation order."""
    if n == 1:
        return (0,) * k
    flags = list(range(n - 1))
    transitions = list(string_from_flags(flags, n=n, k=k))
    flag_set = set(flags)
    for index in range(k * n):
        if index in flag_set:
            continue
        transitions[index] = _upper_bound_at(index, transitions, flags)
    return tuple(transitions)


def next_flags(flags: list[int], *, k: int) -> None:
    """Advance ``flags`` in-place to the next valid flag sequence, or raise ``StopIteration``."""

    def nextflags(index: int) -> None:
        if index == 0:
            if flags[0] == 0:
                raise StopIteration
            flags[0] -= 1
            return
        if flags[index] - 1 == flags[index - 1]:
            flags[index] = k * (index + 1) - 1
            nextflags(index - 1)
        else:
            flags[index] -= 1

    nextflags(len(flags) - 1)


def _nearest_flag(flags: Sequence[int], index: int) -> tuple[int, int]:
    label = 0
    position = -1
    for flag_index, flag_position in enumerate(flags):
        if flag_position <= index and flag_position >= position:
            label = flag_index + 1
            position = flag_position
    return label, position


def _upper_bound_at(index: int, transitions: Sequence[int], flags: Sequence[int]) -> int:
    """Return the maximum value allowed at ``index`` (non-flag positions only)."""
    if index < flags[0]:
        return 0
    _, flag_position = _nearest_flag(flags, index)
    return transitions[flag_position]


def _is_last_icdfa_empty_string(
    transitions: Sequence[int],
    flags: Sequence[int],
    *,
    n: int,
) -> bool:
    for index, value in enumerate(transitions):
        if index in flags:
            continue
        if value < _upper_bound_at(index, transitions, flags):
            return False
    return True


def next_icdfa_empty_string(
    transitions: list[int],
    flags: Sequence[int],
    *,
    n: int,
    k: int,
) -> None:
    """Advance ``transitions`` in-place to the next string for fixed ``flags``."""
    if _is_last_icdfa_empty_string(transitions, flags, n=n):
        raise StopIteration

    flag_set = set(flags)

    def nexticdfa(state: int, symbol: int) -> None:
        index = state * k + symbol
        while index in flag_set:
            for tail in range(index + 1, k * n):
                if tail not in flag_set:
                    transitions[tail] = 0
            symbol -= 1
            if symbol < 0:
                if state == 0:
                    raise ICDFAEnumerationError("cannot advance past first ICDFA string for flags")
                nexticdfa(state - 1, k - 1)
                return
            index -= 1

        upper = _upper_bound_at(index, transitions, flags)
        if transitions[index] == upper:
            transitions[index] = 0
            if symbol == 0:
                if state == 0:
                    raise ICDFAEnumerationError("cannot advance past first ICDFA string for flags")
                nexticdfa(state - 1, k - 1)
            else:
                nexticdfa(state, symbol - 1)
        else:
            transitions[index] += 1

    nexticdfa(n - 1, k - 1)


def count_flag_sequences(k: int, n: int) -> int:
    """Return ``F_{k,n}``, the number of valid flag sequences (Fuss--Catalan)."""
    if n <= 1:
        return 1
    return comb(k * n, n) // ((k - 1) * n + 1)


def count_icdfa_empty(k: int, n: int) -> int:
    """Return ``B_{k,n}``, the number of non-isomorphic ICDFA∅ structures."""
    if n == 1:
        return 1

    total = 0

    def visit(prefix: list[int]) -> None:
        nonlocal total
        depth = len(prefix)
        if depth == n - 1:
            product = 1
            previous = -1
            extended = prefix + [k * n]
            for label in range(1, n + 1):
                current = extended[label - 1]
                product *= label ** (current - previous - 1)
                previous = current
            total += product
            return

        lower = 0 if depth == 0 else prefix[-1] + 1
        upper = k * (depth + 1)
        for value in range(lower, upper):
            prefix.append(value)
            visit(prefix)
            prefix.pop()

    visit([])
    return total


def count_icdfa(k: int, n: int) -> int:
    """Return the number of non-isomorphic ICDFAs (with final states)."""
    return (2**n) * count_icdfa_empty(k, n)


def iter_icdfa_empty_strings(k: int, n: int) -> Iterator[tuple[int, ...]]:
    """Yield all ICDFA∅ transition strings in paper generation order."""
    if n == 1:
        yield (0,) * k
        return

    flags = [k * state - 1 for state in range(1, n)]
    transitions = list(string_from_flags(flags, n=n, k=k))
    while True:
        yield tuple(transitions)
        try:
            next_icdfa_empty_string(transitions, flags, n=n, k=k)
        except StopIteration:
            try:
                next_flags(flags, k=k)
            except StopIteration:
                break
            transitions[:] = list(string_from_flags(flags, n=n, k=k))


def iter_icdfa(k: int, n: int) -> Iterator[tuple[tuple[int, ...], frozenset[int]]]:
    """Yield ``(transitions, final_states)`` for all ICDFAs with ``n`` states and alphabet size ``k``."""
    for transitions in iter_icdfa_empty_strings(k, n):
        for mask in range(2**n):
            finals = frozenset(state for state in range(n) if (mask >> state) & 1)
            yield transitions, finals


def _ordered_alphabet(
    alphabet: Sequence[Any],
    *,
    symbol_order: Sequence[Any] | None,
) -> tuple[Any, ...]:
    if symbol_order is not None:
        order = tuple(symbol_order)
        if len(order) != len(alphabet) or len(set(order)) != len(order):
            raise ICDFAEnumerationError("symbol_order must be a permutation of alphabet")
        if set(order) != set(alphabet):
            raise ICDFAEnumerationError("symbol_order must match alphabet")
        return order
    return tuple(sorted(alphabet, key=lambda value: (type(value).__name__, value)))


def icdfa_string_to_dfa(
    transitions: Sequence[int],
    alphabet: Sequence[Any],
    *,
    n: int | None = None,
    k: int | None = None,
    final_states: frozenset[int] | None = None,
    symbol_order: Sequence[Any] | None = None,
    state_labels: Sequence[Hashable] | None = None,
) -> DFA:
    """Decode a canonical ICDFA string into a complete :class:`DFA`."""
    symbols = _ordered_alphabet(alphabet, symbol_order=symbol_order)
    inferred_k = len(symbols)
    inferred_n = len(transitions) // inferred_k if inferred_k else 0
    states_count = n if n is not None else inferred_n
    alphabet_size = k if k is not None else inferred_k
    if states_count * alphabet_size != len(transitions):
        raise ICDFAEnumerationError("transitions length does not match n and k")

    validate_icdfa_empty_string(transitions, n=states_count, k=alphabet_size)

    if state_labels is None:
        labels: tuple[Hashable, ...] = tuple(range(states_count))
    else:
        labels = tuple(state_labels)
        if len(labels) != states_count:
            raise ICDFAEnumerationError("state_labels length must equal n")

    acceptors = final_states if final_states is not None else frozenset()
    dfa = DFA(
        input_alphabet=frozenset(symbols),
        initial_states=frozenset({labels[0]}),
        accepting_states=frozenset(labels[state] for state in acceptors),
    )
    for label in labels:
        dfa.graph.add_state(label)
    for index, target in enumerate(transitions):
        source = labels[index // alphabet_size]
        symbol = symbols[index % alphabet_size]
        dfa.add_transition(source, labels[target], symbol)
    dfa.validate()
    return dfa


def dfa_to_icdfa_string(
    dfa: DFA,
    *,
    symbol_order: Sequence[Any] | None = None,
) -> ICDFAString:
    """Encode a complete initially-connected DFA as a canonical ICDFA string."""
    from pensive.automata.algorithms import _forward_reachable

    reachable = _forward_reachable(dfa)
    if reachable != set(dfa.states()):
        raise ICDFAEnumerationError("DFA must be initially connected (all states reachable)")

    symbols = _ordered_alphabet(tuple(dfa.input_alphabet), symbol_order=symbol_order)
    k = len(symbols)
    if k == 0:
        raise ICDFAEnumerationError("DFA must have a non-empty input alphabet")

    if len(dfa.initial_states) != 1:
        raise ICDFAEnumerationError("DFA must have exactly one initial state")
    initial = next(iter(dfa.initial_states))

    for state in dfa.states():
        for symbol in symbols:
            if len(dfa.delta(state, symbol)) != 1:
                raise ICDFAEnumerationError("DFA must be complete")

    state_to_index: dict[Hashable, int] = {}
    index_to_state: list[Hashable] = []
    queue: deque[Hashable] = deque([initial])
    state_to_index[initial] = 0
    index_to_state.append(initial)

    while queue:
        current = queue.popleft()
        for symbol in symbols:
            successors = dfa.delta(current, symbol)
            target = next(iter(successors))
            if target not in state_to_index:
                state_to_index[target] = len(index_to_state)
                index_to_state.append(target)
                queue.append(target)

    if set(index_to_state) != reachable:
        raise ICDFAEnumerationError("DFA must be initially connected")

    n = len(index_to_state)
    transitions: list[int] = []
    for index in range(n):
        state = index_to_state[index]
        for symbol in symbols:
            target = next(iter(dfa.delta(state, symbol)))
            transitions.append(state_to_index[target])

    validate_icdfa_empty_string(transitions, n=n, k=k)
    return ICDFAString(transitions=tuple(transitions), n=n, k=k)
