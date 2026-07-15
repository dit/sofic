"""Tests for nested word automata."""

import pytest

from sofic.automata.nwa import NestedWord, NestedWordAutomaton
from sofic.automata.vpa import VisiblyPushdownAutomaton
from sofic.exceptions import SoficValidationError
from sofic.graph import (
    ATTR_KIND,
    ATTR_STACK_SYMBOL,
    ATTR_SYMBOL,
    KIND_CALL,
    KIND_INTERNAL,
    KIND_RETURN,
)


def _balanced_nwa() -> NestedWordAutomaton:
    nwa = NestedWordAutomaton(
        call_alphabet=frozenset({"("}),
        return_alphabet=frozenset({")"}),
        internal_alphabet=frozenset({"i"}),
        hier_alphabet=frozenset({"S"}),
        initial_state="q",
        accepting_states=frozenset({"q"}),
    )
    nwa.graph.add_state("q")
    nwa.add_call_transition("q", "q", "(", "S")
    nwa.add_return_transition("q", "q", ")", "S")
    nwa.add_internal_transition("q", "q", "i")
    return nwa


def _visible(symbols: tuple[str, ...]) -> NestedWord:
    return NestedWord.from_visible_word(
        symbols,
        call_alphabet={"("},
        return_alphabet={")"},
        internal_alphabet={"i"},
    )


def test_nested_word_rejects_crossing_matches():
    with pytest.raises(SoficValidationError):
        NestedWord(
            symbols=("a", "b", "c", "d"),
            kinds=(KIND_CALL, KIND_CALL, KIND_RETURN, KIND_RETURN),
            matching=(2, 3, 0, 1),
        )


def test_nested_word_rejects_bad_call_return_pair():
    with pytest.raises(SoficValidationError):
        NestedWord(
            symbols=("a", "b"),
            kinds=(KIND_CALL, KIND_INTERNAL),
            matching=(1, 0),
        )


def test_recognizes_balanced_nested_word():
    nwa = _balanced_nwa()

    assert nwa.recognizes(_visible(()))
    assert nwa.recognizes(_visible(("(", ")")))
    assert nwa.recognizes(_visible(("(", "(", ")", ")")))
    assert nwa.recognizes(_visible(("i", "(", ")", "i")))


def test_recognizes_pending_call_by_final_state():
    nwa = _balanced_nwa()

    assert nwa.recognizes(_visible(("(",)))
    assert nwa.recognizes(_visible(("(", "i")))


def test_rejects_pending_return_without_bottom_hier_state():
    nwa = _balanced_nwa()

    assert not nwa.recognizes(_visible((")",)))


def test_pending_return_can_use_bottom_hier_state():
    nwa = NestedWordAutomaton(
        return_alphabet=frozenset({"r"}),
        hier_alphabet=frozenset({"BOTTOM"}),
        bottom_hier_state="BOTTOM",
        initial_state="q0",
        accepting_states=frozenset({"q1"}),
    )
    nwa.graph.add_state("q0")
    nwa.graph.add_state("q1")
    nwa.add_return_transition("q0", "q1", "r", "BOTTOM")

    word = NestedWord(symbols=("r",), kinds=(KIND_RETURN,), matching=(None,))

    nwa.validate()
    assert nwa.recognizes(word)


def test_overlapping_role_alphabets_are_disambiguated_by_nested_word():
    nwa = NestedWordAutomaton(
        call_alphabet=frozenset({"x"}),
        return_alphabet=frozenset({"x"}),
        internal_alphabet=frozenset({"x"}),
        hier_alphabet=frozenset({"S"}),
        initial_state="q0",
        accepting_states=frozenset({"q1", "q2"}),
    )
    for state in ("q0", "q1", "q2"):
        nwa.graph.add_state(state)
    nwa.add_call_transition("q0", "q0", "x", "S")
    nwa.add_return_transition("q0", "q1", "x", "S")
    nwa.add_internal_transition("q0", "q2", "x")

    internal = NestedWord(symbols=("x",), kinds=(KIND_INTERNAL,), matching=(None,))
    call_return = NestedWord(symbols=("x", "x"), kinds=(KIND_CALL, KIND_RETURN), matching=(1, 0))

    nwa.validate()
    assert nwa.recognizes(internal)
    assert nwa.recognizes(call_return)


def _vpa() -> VisiblyPushdownAutomaton:
    vpa = VisiblyPushdownAutomaton(
        call_alphabet=frozenset({"("}),
        return_alphabet=frozenset({")"}),
        internal_alphabet=frozenset({"i"}),
        stack_alphabet=frozenset({"S"}),
        initial_state="q",
        accepting_states=frozenset({"q"}),
    )
    vpa.graph.add_state("q")
    vpa.graph.add_transition(
        "q",
        "q",
        **{ATTR_KIND: KIND_CALL, ATTR_SYMBOL: "(", ATTR_STACK_SYMBOL: "S"},
    )
    vpa.graph.add_transition(
        "q",
        "q",
        **{ATTR_KIND: KIND_RETURN, ATTR_SYMBOL: ")"},
    )
    vpa.graph.add_transition(
        "q",
        "q",
        **{ATTR_KIND: KIND_INTERNAL, ATTR_SYMBOL: "i"},
    )
    return vpa


def test_from_vpa_agrees_on_visible_words():
    vpa = _vpa()
    nwa = NestedWordAutomaton.from_vpa(vpa)

    for word in [
        (),
        ("(", ")"),
        ("(", "(", ")", ")"),
        ("i", "(", ")", "i"),
        ("(",),
        (")",),
    ]:
        assert nwa.recognizes_visible(word) == vpa.recognizes(word)


def test_to_vpa_tags_overlapping_role_symbols():
    nwa = NestedWordAutomaton(
        call_alphabet=frozenset({"x"}),
        return_alphabet=frozenset({"x"}),
        internal_alphabet=frozenset({"x"}),
        hier_alphabet=frozenset({"S"}),
        initial_state="q0",
        accepting_states=frozenset({"q1", "q2"}),
    )
    for state in ("q0", "q1", "q2"):
        nwa.graph.add_state(state)
    nwa.add_call_transition("q0", "q0", "x", "S")
    nwa.add_return_transition("q0", "q1", "x", "S")
    nwa.add_internal_transition("q0", "q2", "x")

    vpa = nwa.to_vpa()

    vpa.validate()
    assert vpa.recognizes(((KIND_CALL, "x"), (KIND_RETURN, "x")))
    assert vpa.recognizes(((KIND_INTERNAL, "x"),))
