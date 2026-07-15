"""Tests for visibly pushdown automata."""

import pytest

from sofic.automata.dfa import DFA
from sofic.automata.vpa import (
    CallDrivenAutomaton,
    CanonicalVisiblyPushdownAutomaton,
    DeterministicVisiblyPushdownAutomaton,
    MultipleEntryVisiblyPushdownAutomaton,
    SingleEntryVisiblyPushdownAutomaton,
    VisiblyPushdownAutomaton,
)
from sofic.exceptions import NonDeterministicError, SoficValidationError
from sofic.graph import ATTR_KIND, ATTR_STACK_SYMBOL, ATTR_SYMBOL, KIND_CALL, KIND_INTERNAL, KIND_RETURN


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


def _signature(vpa: VisiblyPushdownAutomaton) -> tuple[object, ...]:
    return (
        frozenset(vpa.states()),
        vpa.initial_state,
        vpa.accepting_states,
        frozenset(
            (
                transition.source,
                transition.target,
                transition.data.get(ATTR_KIND),
                transition.data.get(ATTR_SYMBOL),
                transition.data.get(ATTR_STACK_SYMBOL),
            )
            for transition in vpa.transitions()
        ),
    )


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


def test_guarded_returns_match_top_stack_symbol():
    vpa = VisiblyPushdownAutomaton(
        call_alphabet=frozenset({"a", "b"}),
        return_alphabet=frozenset({"r"}),
        stack_alphabet=frozenset({"A", "B"}),
        initial_state="q",
        accepting_states=frozenset({"q"}),
    )
    vpa.graph.add_state("q")
    vpa.add_call_transition("q", "q", "a", "A")
    vpa.add_call_transition("q", "q", "b", "B")
    vpa.add_return_transition("q", "q", "r", "A")

    assert vpa.recognizes(("a", "r"))
    assert not vpa.recognizes(("b", "r"))


def test_bottom_stack_symbol_allows_guarded_bottom_return():
    vpa = VisiblyPushdownAutomaton(
        return_alphabet=frozenset({"r"}),
        stack_alphabet=frozenset({"BOTTOM"}),
        bottom_stack_symbol="BOTTOM",
        initial_state="q0",
        accepting_states=frozenset({"q1"}),
    )
    vpa.graph.add_state("q0")
    vpa.graph.add_state("q1")
    vpa.add_return_transition("q0", "q1", "r", "BOTTOM")

    vpa.validate()
    assert vpa.recognizes(("r",))


def test_deterministic_validation_rejects_conflicting_calls():
    vpa = DeterministicVisiblyPushdownAutomaton(
        call_alphabet=frozenset({"c"}),
        stack_alphabet=frozenset({"S", "T"}),
        initial_state="q0",
        accepting_states=frozenset({"q0"}),
    )
    for state in ("q0", "q1", "q2"):
        vpa.graph.add_state(state)
    vpa.graph.add_transition("q0", "q1", **{ATTR_KIND: KIND_CALL, ATTR_SYMBOL: "c", ATTR_STACK_SYMBOL: "S"})
    vpa.graph.add_transition("q0", "q2", **{ATTR_KIND: KIND_CALL, ATTR_SYMBOL: "c", ATTR_STACK_SYMBOL: "T"})

    with pytest.raises(NonDeterministicError):
        vpa.validate()


def _cda() -> CallDrivenAutomaton:
    vpa = CallDrivenAutomaton(
        call_alphabet=frozenset({"c", "d"}),
        return_alphabet=frozenset({"r"}),
        stack_alphabet=frozenset({"m"}),
        initial_state="m",
        accepting_states=frozenset({"m"}),
        modules={"main": {"m"}, "proc": {"e0", "e1"}},
        base_module="main",
        call_partition={"c": "proc", "d": "proc"},
        call_entries={"c": "e0", "d": "e1"},
    )
    for state in ("m", "e0", "e1"):
        vpa.graph.add_state(state)
    vpa.add_call_transition("m", "e0", "c", "m")
    vpa.add_call_transition("m", "e1", "d", "m")
    vpa.add_return_transition("e0", "m", "r", "m")
    vpa.add_return_transition("e1", "m", "r", "m")
    return vpa


def test_cda_validate():
    _cda().validate()


def test_mevpa_validate():
    cda = _cda()
    mevpa = MultipleEntryVisiblyPushdownAutomaton(
        call_alphabet=cda.call_alphabet,
        return_alphabet=cda.return_alphabet,
        stack_alphabet=cda.stack_alphabet,
        initial_state=cda.initial_state,
        accepting_states=cda.accepting_states,
        modules=cda.modules,
        base_module=cda.base_module,
        call_partition=cda.call_partition,
        call_entries=cda.call_entries,
        entry_states={"proc": {"e0", "e1"}, "main": {"m"}},
        graph=cda.graph.copy(),
    )

    mevpa.validate()


def test_sevpa_validate_and_reject_bad_stack_symbol():
    sevpa = SingleEntryVisiblyPushdownAutomaton(
        call_alphabet=frozenset({"c"}),
        return_alphabet=frozenset({"r"}),
        stack_alphabet=frozenset({("m", "c")}),
        initial_state="m",
        accepting_states=frozenset({"m"}),
        modules={"main": {"m"}, "proc": {"e"}},
        base_module="main",
        call_partition={"c": "proc"},
        call_entries={"c": "e"},
        entry_states={"proc": "e"},
    )
    for state in ("m", "e"):
        sevpa.graph.add_state(state)
    sevpa.add_call_transition("m", "e", "c", ("m", "c"))
    sevpa.add_return_transition("e", "m", "r", ("m", "c"))
    sevpa.validate()

    bad = sevpa.copy()
    bad.stack_alphabet = frozenset({("wrong", "c"), ("m", "c")})
    bad.graph.add_transition("m", "e", **{ATTR_KIND: KIND_CALL, ATTR_SYMBOL: "c", ATTR_STACK_SYMBOL: ("wrong", "c")})
    with pytest.raises((SoficValidationError, NonDeterministicError)):
        bad.validate()


def _redundant_sevpa() -> SingleEntryVisiblyPushdownAutomaton:
    sevpa = SingleEntryVisiblyPushdownAutomaton(
        call_alphabet=frozenset({"c"}),
        return_alphabet=frozenset({"r"}),
        stack_alphabet=frozenset({("m0", "c"), ("m1", "c")}),
        initial_state="m0",
        accepting_states=frozenset({"m0", "m1"}),
        modules={"main": {"m0", "m1"}, "proc": {"p"}},
        base_module="main",
        call_partition={"c": "proc"},
        call_entries={"c": "p"},
        entry_states={"proc": "p"},
    )
    for state in ("m0", "m1", "p"):
        sevpa.graph.add_state(state)
    for state in ("m0", "m1"):
        sevpa.add_call_transition(state, "p", "c", (state, "c"))
    sevpa.add_return_transition("p", "m0", "r", ("m0", "c"))
    sevpa.add_return_transition("p", "m1", "r", ("m1", "c"))
    return sevpa


def test_sevpa_minimize_merges_redundant_states():
    sevpa = _redundant_sevpa()
    minimized = SingleEntryVisiblyPushdownAutomaton.minimize(sevpa, call_partition=sevpa.call_partition)

    minimized.validate()
    assert len(list(minimized.states())) == 2
    assert minimized.recognizes(())
    assert minimized.recognizes(("c", "r"))


def _internal_vpa(state0: str = "q0", state1: str = "q1") -> DeterministicVisiblyPushdownAutomaton:
    vpa = DeterministicVisiblyPushdownAutomaton(
        internal_alphabet=frozenset({"a", "b"}),
        initial_state=state0,
        accepting_states=frozenset({state1}),
    )
    for state in (state0, state1):
        vpa.graph.add_state(state)
    vpa.add_internal_transition(state0, state1, "a")
    vpa.add_internal_transition(state0, state0, "b")
    vpa.add_internal_transition(state1, state1, "a")
    vpa.add_internal_transition(state1, state0, "b")
    return vpa


def _call_return_vpa() -> VisiblyPushdownAutomaton:
    vpa = VisiblyPushdownAutomaton(
        call_alphabet=frozenset({"c"}),
        return_alphabet=frozenset({"r"}),
        stack_alphabet=frozenset({"S"}),
        initial_state="q0",
        accepting_states=frozenset({"q2"}),
    )
    for state in ("q0", "q1", "q2"):
        vpa.graph.add_state(state)
    vpa.add_call_transition("q0", "q1", "c", "S")
    vpa.add_return_transition("q1", "q2", "r", "S")
    return vpa


def _internal_symbol_vpa() -> VisiblyPushdownAutomaton:
    vpa = VisiblyPushdownAutomaton(
        internal_alphabet=frozenset({"i"}),
        initial_state="p0",
        accepting_states=frozenset({"p1"}),
    )
    for state in ("p0", "p1"):
        vpa.graph.add_state(state)
    vpa.add_internal_transition("p0", "p1", "i")
    return vpa


def _pending_call_vpa() -> VisiblyPushdownAutomaton:
    vpa = VisiblyPushdownAutomaton(
        call_alphabet=frozenset({"c"}),
        stack_alphabet=frozenset({"S"}),
        initial_state="p0",
        accepting_states=frozenset({"p1"}),
    )
    for state in ("p0", "p1"):
        vpa.graph.add_state(state)
    vpa.add_call_transition("p0", "p1", "c", "S")
    return vpa


def test_vpa_standard_boolean_and_structural_operations():
    call_return = _call_return_vpa()
    internal = _internal_symbol_vpa()

    union = call_return.union(internal)
    intersection = union.intersection(call_return)
    difference = union.difference(call_return)
    complement = call_return.complement()
    concat = call_return.concat(internal)
    star = call_return.kleene_star()

    union.validate()
    assert union.recognizes(("c", "r"))
    assert union.recognizes(("i",))
    assert not union.recognizes(("c",))

    assert intersection.recognizes(("c", "r"))
    assert not intersection.recognizes(("i",))
    assert difference.recognizes(("i",))
    assert not difference.recognizes(("c", "r"))

    assert complement.recognizes(())
    assert not complement.recognizes(("c", "r"))
    assert not complement.recognizes(("not-in-alphabet",))

    assert concat.recognizes(("c", "r", "i"))
    assert not concat.recognizes(("c", "r"))
    assert star.recognizes(())
    assert star.recognizes(("c", "r"))
    assert star.recognizes(("c", "r", "c", "r"))
    assert not star.recognizes(("c",))


def test_vpa_concat_does_not_reuse_left_pending_stack():
    pending_call = _pending_call_vpa()
    call_return = _call_return_vpa()
    concat = pending_call.concat(call_return)

    assert pending_call.recognizes(("c",))
    assert not call_return.recognizes(("r",))
    assert not concat.recognizes(("c", "r"))
    assert concat.recognizes(("c", "c", "r"))


def test_canonical_vpa_is_stable_under_state_renaming():
    first = CanonicalVisiblyPushdownAutomaton.from_vpa(_internal_vpa("q0", "q1"))
    second = CanonicalVisiblyPushdownAutomaton.from_vpa(_internal_vpa("left", "right"))

    assert _signature(first) == _signature(second)


def test_no_call_canonical_vpa_matches_dfa_minimization():
    vpa = _internal_vpa()
    vpa.graph.add_state("q2")
    vpa.add_internal_transition("q2", "q2", "a")
    vpa.add_internal_transition("q2", "q0", "b")
    vpa.accepting_states = frozenset({"q1", "q2"})

    dfa = DFA(
        input_alphabet=vpa.internal_alphabet,
        initial_states=frozenset({vpa.initial_state}),
        accepting_states=vpa.accepting_states,
        graph=vpa.graph.copy(),
    )
    minimized_dfa = dfa.minimize(alphabet=vpa.internal_alphabet)
    canonical = CanonicalVisiblyPushdownAutomaton.from_vpa(vpa)

    assert len(list(canonical.states())) == len(list(minimized_dfa.states()))
    for word in [(), ("a",), ("b",), ("a", "a"), ("a", "b"), ("b", "a")]:
        assert canonical.recognizes(word) == minimized_dfa.recognizes(word)
