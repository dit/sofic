"""Tests for canonical symbolic-shift examples."""

from __future__ import annotations

import pytest

from pensive.examples import (
    dyck_shift_order,
    motzkin_shift,
    sofic_dyck_fig1_shift,
    sofic_dyck_nondeterminizable_shift,
    sofic_dyck_zeta_example_shift,
)
from pensive.shifts.sofic_dyck import SoficDyckShift


@pytest.mark.parametrize(
    "constructor",
    [
        dyck_shift_order,
        motzkin_shift,
        sofic_dyck_fig1_shift,
        sofic_dyck_nondeterminizable_shift,
        sofic_dyck_zeta_example_shift,
    ],
)
def test_shift_examples_validate(constructor):
    shift = constructor()

    assert isinstance(shift, SoficDyckShift)
    shift.validate()


def test_dyck_shift_order_structure_and_matching():
    shift = dyck_shift_order(3)

    assert set(shift.states()) == {"1"}
    assert shift.call_alphabet == frozenset({"a1", "a2", "a3"})
    assert shift.return_alphabet == frozenset({"b1", "b2", "b3"})
    assert shift.internal_alphabet == frozenset()
    assert len(shift.matched_edges) == 3
    assert shift.is_admissible_word(("a1", "a2", "b2", "b1"))
    assert not shift.is_admissible_word(("a1", "b2"))


def test_dyck_shift_order_custom_symbols():
    shift = dyck_shift_order(2, call_symbols=("x", "y"), return_symbols=("X", "Y"))

    assert shift.call_alphabet == frozenset({"x", "y"})
    assert shift.return_alphabet == frozenset({"X", "Y"})
    assert shift.is_admissible_word(("x", "y", "Y", "X"))


def test_dyck_shift_order_rejects_bad_symbols():
    with pytest.raises(ValueError, match="length k"):
        dyck_shift_order(2, call_symbols=("a",), return_symbols=("b", "c"))
    with pytest.raises(ValueError, match="distinct"):
        dyck_shift_order(1, call_symbols=("a",), return_symbols=("a",))


def test_motzkin_shift_fig1_left_structure():
    shift = motzkin_shift()

    assert set(shift.states()) == {"1"}
    assert shift.call_alphabet == frozenset({"(", "["})
    assert shift.return_alphabet == frozenset({")", "]"})
    assert shift.internal_alphabet == frozenset({"i"})
    assert len(list(shift.transitions())) == 5
    assert len(shift.matched_edges) == 2
    assert set(shift.words_of_length(1)) == {("(",), ("[",), (")",), ("]",), ("i",)}


def test_sofic_dyck_fig1_right_structure_and_words():
    shift = sofic_dyck_fig1_shift()

    assert set(shift.states()) == {"1", "2"}
    assert len(list(shift.transitions())) == 6
    assert len(shift.matched_edges) == 2
    assert shift.is_admissible_word(("(", "(", "[", "i", "i", "]", "[", "]", ")"))
    assert shift.is_admissible_word((")", ")", ")", ")", ")"))
    assert not shift.is_admissible_word(("(", "[", "i", "]", "[", "]", ")"))
    assert not shift.is_admissible_word(("(", "]"))


def test_sofic_dyck_nondeterminizable_structure_and_words():
    shift = sofic_dyck_nondeterminizable_shift()

    assert set(shift.states()) == {"1", "2", "3"}
    assert shift.call_alphabet == frozenset({"a"})
    assert shift.return_alphabet == frozenset({"b"})
    assert shift.internal_alphabet == frozenset({"i", "j", "k"})
    assert len(list(shift.transitions())) == 7
    assert len(shift.matched_edges) == 1
    assert shift.is_admissible_word(("a", "a", "i", "b", "j", "i", "b", "j", "i", "b", "k"))
    assert not shift.is_admissible_word(("a", "i", "b", "k"))


def test_sofic_dyck_zeta_example_structure_and_words():
    shift = sofic_dyck_zeta_example_shift()

    assert set(shift.states()) == {"1", "2"}
    assert shift.call_alphabet == frozenset({"a", "a'"})
    assert shift.return_alphabet == frozenset({"b", "b'"})
    assert shift.internal_alphabet == frozenset({"i"})
    assert len(list(shift.transitions())) == 6
    assert len(shift.matched_edges) == 2
    assert shift.is_admissible_word(("a", "i", "i", "b"))
    assert shift.is_admissible_word(("a'", "b'"))
    assert not shift.is_admissible_word(("a", "b'"))
    assert not shift.is_admissible_word(("a", "i", "b"))
