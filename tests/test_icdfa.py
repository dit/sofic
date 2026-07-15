"""Tests for ICDFA enumeration and canonical strings."""

from __future__ import annotations

import itertools

import pytest

from sofic.automata.dfa import DFA
from sofic.automata.icdfa import (
    ICDFAEnumerationError,
    _upper_bound_at,
    count_flag_sequences,
    count_icdfa,
    count_icdfa_empty,
    dfa_to_icdfa_string,
    first_icdfa_empty_string,
    flags_from_string,
    icdfa_string_to_dfa,
    iter_icdfa,
    iter_icdfa_empty_strings,
    last_icdfa_empty_string,
    next_flags,
    next_icdfa_empty_string,
    string_from_flags,
    validate_icdfa_empty_string,
)

B2 = [1, 12, 216, 5248, 160675]


def test_count_flag_sequences() -> None:
    assert count_flag_sequences(2, 3) == 5
    assert count_flag_sequences(3, 4) == 55


def test_count_icdfa_empty() -> None:
    for n, expected in enumerate(B2, start=1):
        assert count_icdfa_empty(2, n) == expected


def test_count_icdfa() -> None:
    assert count_icdfa(2, 2) == 48
    assert count_icdfa(2, 3) == 1728


def test_enumerate_small() -> None:
    assert len(list(iter_icdfa_empty_strings(2, 2))) == 12
    assert len(list(iter_icdfa_empty_strings(2, 3))) == 216


@pytest.mark.slow
def test_enumerate_n4() -> None:
    assert len(list(iter_icdfa_empty_strings(2, 4))) == 5248


def test_first_last_boundaries() -> None:
    assert first_icdfa_empty_string(n=3, k=2) == (0, 1, 0, 2, 0, 0)
    assert last_icdfa_empty_string(n=3, k=2) == (1, 2, 2, 2, 2, 2)

    flags = [1, 3]
    current = list(first_icdfa_empty_string(n=3, k=2))
    transitions = list(string_from_flags(flags, n=3, k=2))
    last_for_flags = list(transitions)
    for index in range(6):
        if index in flags:
            continue
        last_for_flags[index] = _upper_bound_at(index, last_for_flags, flags)
    seen: list[tuple[int, ...]] = []
    current = list(transitions)
    while True:
        seen.append(tuple(current))
        if tuple(current) == tuple(last_for_flags):
            break
        next_icdfa_empty_string(current, flags, n=3, k=2)

    assert len(seen) == 18


def test_flag_iteration_matches_count() -> None:
    for k, n in ((2, 3), (2, 4), (3, 3)):
        flags = [k * state - 1 for state in range(1, n)]
        count = 0
        while True:
            count += 1
            try:
                next_flags(flags, k=k)
            except StopIteration:
                break
        assert count == count_flag_sequences(k, n)


def test_round_trip_codec() -> None:
    alphabet = ("a", "b")
    for n in (2, 3):
        for transitions in iter_icdfa_empty_strings(2, n):
            dfa = icdfa_string_to_dfa(transitions, alphabet)
            encoded = dfa_to_icdfa_string(dfa, symbol_order=alphabet)
            assert encoded.transitions == transitions


def test_dfa_round_trip() -> None:
    alphabet = ("a", "b")
    for transitions, finals in itertools.islice(iter_icdfa(2, 2), 0, 24):
        dfa = icdfa_string_to_dfa(transitions, alphabet, final_states=finals)
        encoded = dfa_to_icdfa_string(dfa, symbol_order=alphabet)
        assert encoded.transitions == transitions


def test_iter_icdfa_count() -> None:
    assert len(list(iter_icdfa(2, 2))) == count_icdfa(2, 2)


def test_invalid_strings() -> None:
    with pytest.raises(ICDFAEnumerationError):
        validate_icdfa_empty_string((0, 0, 0, 0), n=2, k=2)
    with pytest.raises(ICDFAEnumerationError):
        validate_icdfa_empty_string((0, 2, 0, 0), n=2, k=2)


def test_flags_from_string() -> None:
    transitions = first_icdfa_empty_string(n=3, k=2)
    assert flags_from_string(transitions, n=3) == (1, 3)
    rebuilt = string_from_flags((1, 3), n=3, k=2)
    assert rebuilt == transitions


def _manual_dfa() -> DFA:
    dfa = DFA(
        input_alphabet=frozenset({"a", "b"}),
        initial_states=frozenset({0}),
        accepting_states=frozenset({1}),
    )
    for state in (0, 1):
        dfa.graph.add_state(state)
    dfa.add_transition(0, 1, "a")
    dfa.add_transition(0, 0, "b")
    dfa.add_transition(1, 1, "a")
    dfa.add_transition(1, 0, "b")
    return dfa


def test_dfa_to_icdfa_manual() -> None:
    encoded = dfa_to_icdfa_string(_manual_dfa(), symbol_order=("a", "b"))
    assert encoded.transitions == (1, 0, 1, 0)
    restored = icdfa_string_to_dfa(encoded.transitions, ("a", "b"), final_states=frozenset({1}))
    assert restored.recognizes(("a",))
    assert not restored.recognizes(())
