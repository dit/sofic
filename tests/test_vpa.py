"""Tests for visibly pushdown automata."""

from pensive.automata.vpa import VisiblyPushdownAutomaton
from pensive.graph import ATTR_KIND, ATTR_STACK_SYMBOL, ATTR_SYMBOL, KIND_CALL, KIND_INTERNAL, KIND_RETURN


def _vpa() -> VisiblyPushdownAutomaton:
    vpa = VisiblyPushdownAutomaton(
        call_alphabet=frozenset({"("}),
        return_alphabet=frozenset({")"}),
        internal_alphabet=frozenset({"i"}),
        stack_alphabet=frozenset({"Z"}),
        initial_state="q0",
        accepting_states=frozenset({"q0"}),
    )
    vpa.graph.add_state("q0")
    vpa.graph.add_transition(
        "q0",
        "q0",
        **{ATTR_KIND: KIND_CALL, ATTR_SYMBOL: "(", ATTR_STACK_SYMBOL: "Z"},
    )
    vpa.graph.add_transition(
        "q0",
        "q0",
        **{ATTR_KIND: KIND_RETURN, ATTR_SYMBOL: ")"},
    )
    vpa.graph.add_transition(
        "q0",
        "q0",
        **{ATTR_KIND: KIND_INTERNAL, ATTR_SYMBOL: "i"},
    )
    return vpa


def test_validate_partition():
    _vpa().validate()


def test_recognizes_balanced_calls():
    vpa = _vpa()
    assert vpa.recognizes(())
    assert vpa.recognizes(("(", ")"))
    assert vpa.recognizes(("(", "(", ")", ")"))
    assert vpa.recognizes(("i", "(", ")", "i"))


def test_recognizes_rejects_underflow():
    vpa = _vpa()
    assert not vpa.recognizes((")",))


def test_recognizes_internal_only():
    vpa = _vpa()
    assert vpa.recognizes(("i", "i", "i"))
