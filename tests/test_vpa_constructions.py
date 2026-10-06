"""Concrete VPA constructions checked against brute-force reference semantics."""

from itertools import product

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from sofic.automata.nwa import NestedWordAutomaton
from sofic.automata.vpa import (
    CanonicalVisiblyPushdownAutomaton,
    DeterministicVisiblyPushdownAutomaton,
    MultipleEntryVisiblyPushdownAutomaton,
    SingleEntryVisiblyPushdownAutomaton,
    VisiblyPushdownAutomaton,
)
from sofic.automata.vpa.operations import to_multiple_entry, to_single_entry
from sofic.exceptions import NonWellMatchedLanguageError

CALLS, RETURNS, INTERNALS = frozenset({"c"}), frozenset({"r"}), frozenset({"i"})
SYMBOLS = ("c", "r", "i")
MAX_LENGTH = 6
WORDS = [word for n in range(MAX_LENGTH + 1) for word in product(SYMBOLS, repeat=n)]


@st.composite
def vpas(draw, max_states: int = 3):
    """Small, possibly nondeterministic VPAs over one call, return, and internal symbol."""
    n = draw(st.integers(1, max_states))
    states = list(range(n))
    with_bottom = draw(st.booleans())
    vpa = VisiblyPushdownAutomaton(
        call_alphabet=CALLS,
        return_alphabet=RETURNS,
        internal_alphabet=INTERNALS,
        stack_alphabet=frozenset({"A", "B"} | ({"Z"} if with_bottom else set())),
        bottom_stack_symbol="Z" if with_bottom else None,
        initial_state=0,
        accepting_states=frozenset(draw(st.sets(st.sampled_from(states)))),
    )
    for state in states:
        vpa.graph.add_state(state)
    state = st.sampled_from(states)
    for source, target in draw(st.lists(st.tuples(state, state), max_size=4)):
        vpa.add_internal_transition(source, target, "i")
    for source, target, push in draw(st.lists(st.tuples(state, state, st.sampled_from("AB")), max_size=3)):
        vpa.add_call_transition(source, target, "c", push)
    guards = ["A", "B", None] + (["Z"] if with_bottom else [])
    for source, target, guard in draw(st.lists(st.tuples(state, state, st.sampled_from(guards)), max_size=3)):
        vpa.add_return_transition(source, target, "r", guard)
    return vpa


def _language(recognize) -> frozenset[tuple[str, ...]]:
    return frozenset(word for word in WORDS if recognize(word))


def _concat_reference(left, right):
    return lambda w: any(left(w[:k]) and right(w[k:]) for k in range(len(w) + 1))


def _star_reference(inner):
    def recognize(word):
        accepted = [True] + [False] * len(word)
        for end in range(1, len(word) + 1):
            accepted[end] = any(accepted[start] and inner(word[start:end]) for start in range(end))
        return accepted[-1]

    return recognize


SETTINGS = settings(max_examples=40, deadline=None, suppress_health_check=[HealthCheck.too_slow])


@SETTINGS
@given(vpas(), vpas())
def test_boolean_operations_match_reference(left, right):
    a, b = _language(left.recognizes), _language(right.recognizes)
    assert _language(left.union(right).recognizes) == a | b
    assert _language(left.intersection(right).recognizes) == a & b
    assert _language(left.difference(right).recognizes) == a - b
    assert _language(left.complement().recognizes) == frozenset(WORDS) - a


@SETTINGS
@given(vpas(), vpas())
def test_concatenation_matches_reference(left, right):
    reference = _concat_reference(left.recognizes, right.recognizes)
    assert _language(left.concat(right).recognizes) == _language(reference)


@SETTINGS
@given(vpas(max_states=2))
def test_kleene_star_matches_reference(vpa):
    assert _language(vpa.kleene_star().recognizes) == _language(_star_reference(vpa.recognizes))


@SETTINGS
@given(vpas())
def test_determinization_preserves_language_and_is_deterministic(vpa):
    deterministic = vpa.determinize()
    assert isinstance(deterministic, DeterministicVisiblyPushdownAutomaton)
    deterministic.validate()
    assert _language(deterministic.recognizes) == _language(vpa.recognizes)
    assert DeterministicVisiblyPushdownAutomaton.from_vpa(vpa).equivalent(vpa)


@SETTINGS
@given(vpas())
def test_double_complement_and_decision_procedures(vpa):
    assert vpa.complement().complement().equivalent(vpa)
    assert vpa.equivalent(vpa)
    assert vpa.includes(vpa.intersection(vpa.complement()))
    sample_accepts = any(vpa.recognizes(word) for word in WORDS)
    witness = vpa.accepted_word()
    assert vpa.is_empty() == (witness is None)
    if witness is not None:
        assert vpa.recognizes(witness)
    if sample_accepts:
        assert not vpa.is_empty()
    assert vpa.union(vpa.complement()).is_universal()


@SETTINGS
@given(vpas())
def test_has_unmatched_word_matches_reference(vpa):
    def unmatched(word):
        depth = 0
        for symbol in word:
            if symbol == "c":
                depth += 1
            elif symbol == "r":
                if depth == 0:
                    return True
                depth -= 1
        return depth > 0

    if any(vpa.recognizes(w) and unmatched(w) for w in WORDS):
        assert vpa.has_unmatched_word()
    if not vpa.has_unmatched_word():
        assert not any(vpa.recognizes(w) and unmatched(w) for w in WORDS)


@SETTINGS
@given(vpas())
def test_modular_conversions_preserve_well_matched_languages(vpa):
    if vpa.has_unmatched_word():
        with pytest.raises(NonWellMatchedLanguageError):
            to_single_entry(vpa)
        return
    reference = _language(vpa.recognizes)
    single = to_single_entry(vpa)
    multiple = to_multiple_entry(vpa)
    assert _language(single.recognizes) == reference
    assert _language(multiple.recognizes) == reference
    assert _language(SingleEntryVisiblyPushdownAutomaton.minimize(vpa).recognizes) == reference
    assert _language(MultipleEntryVisiblyPushdownAutomaton.minimize(vpa).recognizes) == reference
    canonical = CanonicalVisiblyPushdownAutomaton.from_vpa(vpa)
    assert _language(canonical.recognizes) == reference


def _pending_call_vpa() -> VisiblyPushdownAutomaton:
    vpa = VisiblyPushdownAutomaton(
        call_alphabet=CALLS,
        stack_alphabet=frozenset({"S"}),
        initial_state=0,
        accepting_states=frozenset({1}),
    )
    vpa.graph.add_state(0)
    vpa.graph.add_state(1)
    vpa.add_call_transition(0, 1, "c", "S")
    return vpa


def test_canonical_vpa_rejects_languages_with_pending_calls():
    vpa = _pending_call_vpa()
    assert vpa.recognizes(("c",))
    with pytest.raises(NonWellMatchedLanguageError):
        CanonicalVisiblyPushdownAutomaton.from_vpa(vpa)


def test_nondeterministic_from_vpa_determinizes():
    vpa = VisiblyPushdownAutomaton(internal_alphabet=INTERNALS, initial_state=0, accepting_states=frozenset({1}))
    vpa.graph.add_state(0)
    vpa.graph.add_state(1)
    vpa.add_internal_transition(0, 0, "i")
    vpa.add_internal_transition(0, 1, "i")
    deterministic = DeterministicVisiblyPushdownAutomaton.from_vpa(vpa)
    deterministic.validate()
    assert deterministic.recognizes(("i", "i"))
    assert not deterministic.recognizes(())


def test_nwa_operations_delegate_to_vpas():
    nwa = NestedWordAutomaton(
        call_alphabet=frozenset({"x"}),
        return_alphabet=frozenset({"x"}),
        internal_alphabet=frozenset(),
        hier_alphabet=frozenset({"H"}),
        initial_state=0,
        accepting_states=frozenset({0}),
    )
    nwa.graph.add_state(0)
    nwa.add_call_transition(0, 0, "x", "H")
    nwa.add_return_transition(0, 0, "x", "H")
    complement = nwa.complement()
    assert not nwa.is_empty()
    assert nwa.union(complement).is_universal()
    assert nwa.intersection(complement).is_empty()
    assert nwa.kleene_star().equivalent(nwa)
