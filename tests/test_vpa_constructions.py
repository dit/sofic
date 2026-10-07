"""Concrete VPA constructions checked against brute-force reference semantics."""

from itertools import product

import pytest
from hypothesis import HealthCheck, given, settings

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
from sofic.testing.strategies import vpas

CALLS, RETURNS, INTERNALS = frozenset({"c"}), frozenset({"r"}), frozenset({"i"})
SYMBOLS = ("c", "r", "i")
MAX_LENGTH = 6
WORDS = [word for n in range(MAX_LENGTH + 1) for word in product(SYMBOLS, repeat=n)]


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


def _well_matched_tracker() -> VisiblyPushdownAutomaton:
    tracker = VisiblyPushdownAutomaton(
        call_alphabet=CALLS,
        return_alphabet=RETURNS,
        internal_alphabet=INTERNALS,
        stack_alphabet=frozenset({"E", "N"}),
        initial_state="E",
        accepting_states=frozenset({"E"}),
    )
    for state in ("E", "N"):
        tracker.graph.add_state(state)
    for state in ("E", "N"):
        tracker.add_call_transition(state, "N", "c", state)
        tracker.add_internal_transition(state, state, "i")
        tracker.add_return_transition(state, "E", "r", "E")
        tracker.add_return_transition(state, "N", "r", "N")
    return tracker


def _brute_force_accepts(vpa, word) -> bool:
    """Configuration-set simulation for a VPA without a bottom stack symbol."""
    configs = {(vpa.initial_state, ())}
    for symbol in word:
        following = set()
        for state, stack in configs:
            for transition in vpa.graph.out_transitions(state):
                if transition.data.get("symbol") != symbol:
                    continue
                pushed = transition.data.get("stack_symbol")
                if symbol == "i":
                    following.add((transition.target, stack))
                elif symbol == "c":
                    following.add((transition.target, (*stack, pushed)))
                elif stack and pushed in (None, stack[-1]):
                    following.add((transition.target, stack[:-1]))
        configs = following
    return any(state in vpa.accepting_states for state, _stack in configs)


def _random_well_matched_vpa(rng):
    n = rng.randint(1, 2)
    vpa = VisiblyPushdownAutomaton(
        call_alphabet=CALLS,
        return_alphabet=RETURNS,
        internal_alphabet=INTERNALS,
        stack_alphabet=frozenset({"A", "B"}),
        initial_state=0,
        accepting_states=frozenset(s for s in range(n) if rng.random() < 0.5),
    )
    for state in range(n):
        vpa.graph.add_state(state)
    for source, target in product(range(n), repeat=2):
        if rng.random() < 0.35:
            vpa.add_call_transition(source, target, "c", rng.choice("AB"))
        if rng.random() < 0.35:
            vpa.add_internal_transition(source, target, "i")
        if rng.random() < 0.35:
            vpa.add_return_transition(source, target, "r", rng.choice(["A", "B", None]))
    return vpa.intersection(_well_matched_tracker())


def test_modular_minimize_splits_callers_with_distinct_return_targets():
    vpa = VisiblyPushdownAutomaton(
        call_alphabet=CALLS,
        return_alphabet=RETURNS,
        internal_alphabet=INTERNALS,
        stack_alphabet=frozenset({"A", "B"}),
        initial_state=0,
        accepting_states=frozenset({0, 1}),
    )
    vpa.graph.add_state(0)
    vpa.graph.add_state(1)
    vpa.add_call_transition(0, 0, "c", "B")
    vpa.add_call_transition(0, 1, "c", "B")
    vpa.add_return_transition(0, 1, "r", "B")
    vpa.add_call_transition(1, 1, "c", "A")
    matched = vpa.intersection(_well_matched_tracker())
    reference = _language(lambda word: _brute_force_accepts(matched, word))
    single = SingleEntryVisiblyPushdownAutomaton.minimize(to_single_entry(matched))
    assert _language(single.recognizes) == reference
    assert _language(SingleEntryVisiblyPushdownAutomaton.minimize(matched).recognizes) == reference
    assert _language(MultipleEntryVisiblyPushdownAutomaton.minimize(matched).recognizes) == reference


def test_modular_minimize_matches_brute_force_on_random_well_matched_vpas():
    import random

    rng = random.Random(7)
    for _trial in range(60):
        vpa = _random_well_matched_vpa(rng)
        reference = _language(lambda word, vpa=vpa: _brute_force_accepts(vpa, word))
        assert _language(vpa.recognizes) == reference
        for cls in (SingleEntryVisiblyPushdownAutomaton, MultipleEntryVisiblyPushdownAutomaton):
            minimized = cls.minimize(vpa)
            minimized.validate()
            assert _language(minimized.recognizes) == reference
        assert _language(SingleEntryVisiblyPushdownAutomaton.minimize(to_single_entry(vpa)).recognizes) == reference


def _slow_equivalence_vpa() -> VisiblyPushdownAutomaton:
    vpa = VisiblyPushdownAutomaton(
        call_alphabet=CALLS,
        return_alphabet=RETURNS,
        internal_alphabet=INTERNALS,
        stack_alphabet=frozenset({"A", "B"}),
        initial_state=0,
        accepting_states=frozenset({0}),
    )
    for state in range(3):
        vpa.graph.add_state(state)
    vpa.add_internal_transition(0, 0, "i")
    vpa.add_call_transition(0, 0, "c", "A")
    vpa.add_return_transition(0, 0, "r")
    vpa.add_internal_transition(0, 2, "i")
    vpa.add_call_transition(1, 2, "c", "B")
    vpa.add_call_transition(2, 0, "c", "A")
    vpa.add_internal_transition(2, 1, "i")
    vpa.add_call_transition(2, 1, "c", "B")
    vpa.add_call_transition(2, 2, "c", "B")
    vpa.add_return_transition(2, 2, "r", "B")
    return vpa


def test_equivalence_with_own_determinization_is_fast():
    import time

    vpa = _slow_equivalence_vpa()
    started = time.perf_counter()
    deterministic = vpa.determinize()
    assert vpa.equivalent(deterministic)
    assert deterministic.complement().complement().equivalent(deterministic)
    # Took over 15 s before worklist saturation and the deterministic complement shortcut.
    assert time.perf_counter() - started < 10.0
    for word in (w for w in WORDS if len(w) <= 4):
        assert deterministic.recognizes(word) == _brute_force_accepts(vpa, word)


def test_deterministic_transition_maps_track_mutation():
    deterministic = DeterministicVisiblyPushdownAutomaton(
        call_alphabet=CALLS,
        return_alphabet=RETURNS,
        internal_alphabet=INTERNALS,
        stack_alphabet=frozenset({"A"}),
        initial_state=0,
        accepting_states=frozenset({1}),
    )
    deterministic.graph.add_state(0)
    deterministic.graph.add_state(1)
    deterministic.add_call_transition(0, 1, "c", "A")
    assert deterministic.return_successor(1, "r", "A") is None
    deterministic.add_return_transition(1, 0, "r", "A")
    assert deterministic.return_successor(1, "r", "A") == 0
    deterministic.graph.add_transition(0, 0, kind="internal", symbol="i")
    assert deterministic.internal_successor(0, "i") == 0
