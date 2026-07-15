"""Tests for SoficDyckShift."""

import pytest

from sofic.exceptions import SoficValidationError
from sofic.graph import KIND_CALL, KIND_INTERNAL, KIND_RETURN
from sofic.shifts.sofic_dyck import SoficDyckShift


def _dyck2() -> SoficDyckShift:
    shift = SoficDyckShift(
        call_alphabet=frozenset({"a", "b"}),
        return_alphabet=frozenset({"A", "B"}),
    )
    shift.graph.add_state("q")
    call_a = shift.add_call_transition("q", "q", "a")
    call_b = shift.add_call_transition("q", "q", "b")
    return_a = shift.add_return_transition("q", "q", "A")
    return_b = shift.add_return_transition("q", "q", "B")
    shift.add_matched_pair(call_a, return_a)
    shift.add_matched_pair(call_b, return_b)
    return shift


def test_one_state_dyck_shift_order_two():
    shift = _dyck2()

    shift.validate()
    assert shift.is_admissible_word(("a", "A"))
    assert shift.is_admissible_word(("a", "b", "B", "A"))
    assert not shift.is_admissible_word(("a", "B"))
    assert not shift.is_admissible_word(("a", "b", "A", "B"))


def test_pending_calls_and_returns_are_factors():
    shift = _dyck2()

    assert shift.is_admissible_word(("a",))
    assert shift.is_admissible_word(("A",))
    assert shift.is_admissible_word(("A", "a"))


def test_motzkin_style_internal_symbols():
    shift = SoficDyckShift(
        call_alphabet=frozenset({"c"}),
        return_alphabet=frozenset({"r"}),
        internal_alphabet=frozenset({"i"}),
    )
    shift.graph.add_state("q")
    call = shift.add_call_transition("q", "q", "c")
    ret = shift.add_return_transition("q", "q", "r")
    shift.add_internal_transition("q", "q", "i")
    shift.add_matched_pair(call, ret)

    shift.validate()
    assert shift.is_admissible_word(("i", "c", "i", "r", "i"))
    assert set(shift.words_of_length(1)) == {("c",), ("r",), ("i",)}


def test_factor_language_uses_dyck_matching():
    shift = _dyck2()
    words = set(shift.factor_language(2))

    assert ("a", "A") in words
    assert ("b", "B") in words
    assert ("A", "a") in words
    assert ("a", "B") not in words


def test_validate_rejects_missing_matched_edge():
    shift = _dyck2()
    call_ref = next(iter(call for call, _return in shift.matched_edges))
    shift.matched_edges = frozenset({(call_ref, ("missing", "missing", 0))})

    with pytest.raises(SoficValidationError, match="missing"):
        shift.validate()


def test_validate_rejects_matched_internal_edge():
    shift = SoficDyckShift(
        call_alphabet=frozenset({"c"}),
        return_alphabet=frozenset({"r"}),
        internal_alphabet=frozenset({"i"}),
    )
    shift.graph.add_state("q")
    call = shift.add_call_transition("q", "q", "c")
    internal = shift.add_internal_transition("q", "q", "i")
    shift.add_return_transition("q", "q", "r")
    shift.add_matched_pair(call, internal)

    with pytest.raises(SoficValidationError, match="not a return"):
        shift.validate()


def test_validate_rejects_bad_role_alphabet():
    shift = SoficDyckShift(call_alphabet=frozenset({"x"}), return_alphabet=frozenset({"x"}))

    with pytest.raises(SoficValidationError, match="disjoint"):
        shift.validate()


def test_symbols_carry_expected_edge_kinds():
    shift = _dyck2()
    kinds = {transition.data["symbol"]: transition.data["kind"] for transition in shift.transitions()}

    assert kinds["a"] == KIND_CALL
    assert kinds["A"] == KIND_RETURN
    assert KIND_INTERNAL not in kinds.values()
