"""Enumeration of incomplete accessible DFA structures (IDFA∅).

Extends the complete ICDFA string representation of Almeida, Moreira, and Reis
(2007) with ``-1`` for missing transitions, following the accessible-DFA
generation and rank function ``B¹_{n,k}`` used by Johnson et al. (2010),
*Enumerating Finitary Processes* (arXiv:1011.0036).
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from functools import cache

from pensive.automata.icdfa import (
    _upper_bound_at,
    _validate_flags,
    next_flags,
)
from pensive.exceptions import PensiveValidationError

__all__ = [
    "MISSING_TRANSITION",
    "count_accessible_idfa",
    "count_idfa_strings_for_flags",
    "extended_flags",
    "first_idfa_string",
    "idfa_string_to_topological_graph",
    "iter_idfa_strings",
    "last_idfa_string",
    "next_idfa_string",
    "rank_idfa_string",
    "transition_count",
    "unrank_idfa_string",
    "validate_idfa_string",
]

MISSING_TRANSITION = -1


class IDFAEnumerationError(PensiveValidationError):
    """Raised when incomplete accessible DFA enumeration fails."""


def extended_flags(flags: Sequence[int], *, n: int, k: int) -> tuple[int, ...]:
    """Return ``(f_0, …, f_n)`` with ``f_0 = -1`` and ``f_n = nk``."""
    return (MISSING_TRANSITION,) + tuple(flags) + (n * k,)


def validate_idfa_string(
    transitions: Sequence[int],
    *,
    n: int,
    k: int,
) -> None:
    """Validate an incomplete accessible DFA string (rules R1 and R2)."""
    length = k * n
    if len(transitions) != length:
        raise IDFAEnumerationError(f"expected length {length}, got {len(transitions)}")
    for value in transitions:
        if value != MISSING_TRANSITION and not 0 <= value < n:
            raise IDFAEnumerationError(f"transition value {value!r} not in [-1, {n - 1}]")

    for index, value in enumerate(transitions):
        if value == MISSING_TRANSITION or value <= 1:
            continue
        if not any(transitions[j] == value - 1 for j in range(index) if transitions[j] != MISSING_TRANSITION):
            raise IDFAEnumerationError(f"state {value} at index {index} appears before state {value - 1}")

    for state in range(1, n):
        if state not in transitions[: k * state]:
            raise IDFAEnumerationError(f"state {state} does not appear in the first {k * state} symbols")


def transition_count(transitions: Sequence[int]) -> int:
    """Return the number of defined transitions in ``transitions``."""
    return sum(1 for value in transitions if value != MISSING_TRANSITION)


def string_from_flags(
    flags: Sequence[int],
    *,
    n: int,
    k: int,
    filler: int = MISSING_TRANSITION,
) -> tuple[int, ...]:
    """Build the first IDFA∅ string for a valid flag sequence."""
    _validate_flags(flags, n=n, k=k)
    transitions = [filler] * (k * n)
    for state, flag in enumerate(flags, start=1):
        transitions[flag] = state
    for index in range(flags[0]):
        transitions[index] = 0
    return tuple(transitions)


def first_idfa_string(*, n: int, k: int) -> tuple[int, ...]:
    """Return the first incomplete accessible DFA string in generation order."""
    if n == 1:
        return (MISSING_TRANSITION,) * k
    flags = tuple(k * state - 1 for state in range(1, n))
    return string_from_flags(flags, n=n, k=k)


def last_idfa_string(*, n: int, k: int) -> tuple[int, ...]:
    """Return the last incomplete accessible DFA string for the last flag sequence."""
    if n == 1:
        return (n - 1,) * k
    flags = list(range(n - 1))
    transitions = list(string_from_flags(flags, n=n, k=k))
    flag_set = set(flags)
    for index in range(k * n):
        if index in flag_set:
            continue
        transitions[index] = _upper_bound_at(index, transitions, flags)
    return tuple(transitions)


def _is_last_idfa_string(
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


def _advance_idfa_value(current: int, upper: int) -> tuple[int, bool]:
    """Advance one position; return ``(new_value, carry)``."""
    if current < upper:
        if current == MISSING_TRANSITION and upper < 0:
            return MISSING_TRANSITION, True
        if current == MISSING_TRANSITION:
            return 0, False
        return current + 1, False
    if current == upper:
        return MISSING_TRANSITION, True
    raise IDFAEnumerationError(f"invalid transition value {current!r} with upper bound {upper}")


def next_idfa_string(
    transitions: list[int],
    flags: Sequence[int],
    *,
    n: int,
    k: int,
) -> None:
    """Advance ``transitions`` in-place to the next string for fixed ``flags``."""
    if _is_last_idfa_string(transitions, flags, n=n):
        raise StopIteration

    flag_set = set(flags)

    def nextidfa(state: int, symbol: int) -> None:
        index = state * k + symbol
        while index in flag_set:
            for tail in range(index + 1, k * n):
                if tail not in flag_set:
                    transitions[tail] = MISSING_TRANSITION
            symbol -= 1
            if symbol < 0:
                if state == 0:
                    raise IDFAEnumerationError("cannot advance past first IDFA string for flags")
                nextidfa(state - 1, k - 1)
                return
            index -= 1

        upper = _upper_bound_at(index, transitions, flags)
        new_value, carry = _advance_idfa_value(transitions[index], upper)
        transitions[index] = new_value
        if carry:
            if symbol == 0:
                if state == 0:
                    raise IDFAEnumerationError("cannot advance past first IDFA string for flags")
                nextidfa(state - 1, k - 1)
            else:
                nextidfa(state, symbol - 1)

    nextidfa(n - 1, k - 1)


def count_idfa_strings_for_flags(flags: Sequence[int], *, n: int, k: int) -> int:
    """Return the number of IDFA∅ strings with the given flag sequence."""
    ext = extended_flags(flags, n=n, k=k)
    product = 1
    for segment_index in range(n):
        segment = ext[segment_index + 1] - ext[segment_index] - 1
        if segment == 0:
            continue
        if segment_index == 0:
            continue
        product *= (segment_index + 2) ** segment
    return product


def count_accessible_idfa(k: int, n: int) -> int:
    """Return ``B¹_{k,n}``, the number of incomplete accessible DFA∅ structures."""
    if n == 1:
        return 2**k

    total = 0

    def visit(prefix: list[int]) -> None:
        nonlocal total
        depth = len(prefix)
        if depth == n - 1:
            total += count_idfa_strings_for_flags(prefix, n=n, k=k)
            return

        lower = 0 if depth == 0 else prefix[-1] + 1
        upper = k * (depth + 1)
        for value in range(lower, upper):
            prefix.append(value)
            visit(prefix)
            prefix.pop()

    visit([])
    return total


@cache
def _enumeration_index(k: int, n: int) -> dict[tuple[int, ...], int]:
    return {candidate: rank for rank, candidate in enumerate(_iter_idfa_strings_impl(k, n))}


def rank_idfa_string(transitions: Sequence[int], *, n: int, k: int) -> int:
    """Return ``B¹_{n,k}(S)``, the rank of ``transitions`` in generation order."""
    validate_idfa_string(transitions, n=n, k=k)
    try:
        return _enumeration_index(k, n)[tuple(transitions)]
    except KeyError as exc:
        raise IDFAEnumerationError("string is not in the accessible DFA enumeration") from exc


def unrank_idfa_string(rank: int, *, n: int, k: int) -> tuple[int, ...]:
    """Return the IDFA∅ string with rank ``rank`` in ``[0, B¹_{n,k})``."""
    if rank < 0:
        raise IDFAEnumerationError("rank must be nonnegative")
    if n == 1:
        if rank >= 2**k:
            raise IDFAEnumerationError("rank out of range")
        transitions: list[int] = []
        for index in range(k):
            mask = 2 ** (k - 1 - index)
            transitions.append(0 if rank & mask else MISSING_TRANSITION)
        return tuple(transitions)
    for index, candidate in enumerate(_iter_idfa_strings_impl(k, n)):
        if index == rank:
            return candidate
    raise IDFAEnumerationError("rank out of range")


def _iter_idfa_strings_impl(k: int, n: int) -> Iterator[tuple[int, ...]]:
    if n == 1:
        for rank in range(2**k):
            yield unrank_idfa_string(rank, n=n, k=k)
        return

    flags = [k * state - 1 for state in range(1, n)]
    transitions = list(string_from_flags(flags, n=n, k=k))
    while True:
        yield tuple(transitions)
        try:
            next_idfa_string(transitions, flags, n=n, k=k)
        except StopIteration:
            try:
                next_flags(flags, k=k)
            except StopIteration:
                break
            transitions[:] = list(string_from_flags(flags, n=n, k=k))


def iter_idfa_strings(k: int, n: int) -> Iterator[tuple[int, ...]]:
    """Yield all incomplete accessible DFA∅ strings in generation order."""
    yield from _iter_idfa_strings_impl(k, n)


def idfa_string_to_topological_graph(
    transitions: Sequence[int],
    *,
    n: int,
    k: int,
    alphabet: Sequence[object] | None = None,
):
    """Decode an IDFA string into a :class:`~pensive.generators.synchronization.TopologicalUnifilarGraph`."""
    from pensive.generators.synchronization import TopologicalUnifilarGraph

    validate_idfa_string(transitions, n=n, k=k)
    if alphabet is None:
        symbols = tuple(range(k))
    else:
        if len(alphabet) != k:
            raise IDFAEnumerationError("alphabet length must equal k")
        symbols = tuple(alphabet)

    states = frozenset(range(n))
    edges: dict[tuple[int, object], int] = {}
    for index, target in enumerate(transitions):
        if target == MISSING_TRANSITION:
            continue
        source = index // k
        symbol = symbols[index % k]
        edges[(source, symbol)] = target
    return TopologicalUnifilarGraph(states=states, alphabet=frozenset(symbols), transitions=edges)


def _delta_table(transitions: Sequence[int], *, n: int, k: int) -> list[list[int | None]]:
    table: list[list[int | None]] = [[None] * k for _ in range(n)]
    for index, target in enumerate(transitions):
        if target == MISSING_TRANSITION:
            continue
        table[index // k][index % k] = target
    return table


def reroot_idfa_string(
    transitions: Sequence[int],
    *,
    new_root: int,
    n: int,
    k: int,
) -> tuple[int, ...] | None:
    """Relabel with ``new_root`` as state ``0``; return ``None`` if labeling fails."""
    validate_idfa_string(transitions, n=n, k=k)
    table = _delta_table(transitions, n=n, k=k)
    old_to_new: dict[int, int] = {new_root: 0}
    new_to_old: list[int] = [new_root]
    next_label = 1

    while True:
        progressed = False
        for new_source in range(len(new_to_old)):
            old_source = new_to_old[new_source]
            for symbol in range(k):
                target = table[old_source][symbol]
                if target is None or target in old_to_new:
                    continue
                old_to_new[target] = next_label
                new_to_old.append(target)
                next_label += 1
                progressed = True
        if not progressed:
            break

    if len(new_to_old) != n:
        return None

    rebuilt = [MISSING_TRANSITION] * (k * n)
    for new_source, old_source in enumerate(new_to_old):
        for symbol in range(k):
            target = table[old_source][symbol]
            if target is None:
                continue
            rebuilt[new_source * k + symbol] = old_to_new[target]
    try:
        validate_idfa_string(rebuilt, n=n, k=k)
    except IDFAEnumerationError:
        return None
    return tuple(rebuilt)
