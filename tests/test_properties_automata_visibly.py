"""Property-based tests for visibly pushdown automata, nested word automata, and PAPNI encoding."""

from __future__ import annotations

from typing import Any

import pytest
from hypothesis import assume, given
from hypothesis import strategies as st

from sofic.automata import (
    DyckAlphabet,
    NestedWordAutomaton,
    VisiblyPushdownAutomaton,
    encode_dyck_samples,
    encode_dyck_word,
    is_well_matched,
)
from sofic.automata.vpa import to_multiple_entry, to_single_entry
from sofic.exceptions import NonWellMatchedLanguageError
from sofic.testing import nwas, vpas
from tests import oracles

VISIBLE = ("c", "r", "i")
N = 5
WORDS = oracles.words(VISIBLE, N)
EVERYTHING = frozenset(WORDS)

Word = tuple[Any, ...]


def vpa_truth(vpa: VisiblyPushdownAutomaton) -> frozenset[Word]:
    return frozenset(w for w in WORDS if oracles.vpa_accepts(vpa, w))


def vpa_lang(vpa: VisiblyPushdownAutomaton) -> frozenset[Word]:
    return frozenset(w for w in WORDS if vpa.recognizes(w))


def nwa_truth(nwa: NestedWordAutomaton) -> frozenset[Word]:
    return frozenset(w for w in WORDS if oracles.nwa_accepts(nwa, w))


def nwa_lang(nwa: NestedWordAutomaton) -> frozenset[Word]:
    return frozenset(w for w in WORDS if nwa.recognizes_visible(w))


def concat_of(left: frozenset[Word], right: frozenset[Word]) -> frozenset[Word]:
    return frozenset(w for w in WORDS if any(w[:i] in left and w[i:] in right for i in range(len(w) + 1)))


def star_of(language: frozenset[Word]) -> frozenset[Word]:
    def member(w: Word) -> bool:
        reach = [True] + [False] * len(w)
        for end in range(1, len(w) + 1):
            reach[end] = any(reach[start] and w[start:end] in language for start in range(end))
        return reach[-1]

    return frozenset(w for w in WORDS if member(w))


def well_matched(w: Word) -> bool:
    depth = 0
    for symbol in w:
        depth += {"c": 1, "r": -1}.get(symbol, 0)
        if depth < 0:
            return False
    return depth == 0


def well_matched_vpa() -> VisiblyPushdownAutomaton:
    """VPA of all well-matched words: state ``top`` means the stack is empty."""
    vpa = VisiblyPushdownAutomaton(
        call_alphabet=frozenset({"c"}),
        return_alphabet=frozenset({"r"}),
        internal_alphabet=frozenset({"i"}),
        stack_alphabet=frozenset({"d0", "d1"}),
        initial_state="top",
        accepting_states=frozenset({"top"}),
    )
    for state in ("top", "deep"):
        vpa.graph.add_state(state)
        vpa.add_internal_transition(state, state, "i")
    vpa.add_call_transition("top", "deep", "c", "d0")
    vpa.add_call_transition("deep", "deep", "c", "d1")
    vpa.add_return_transition("deep", "deep", "r", "d1")
    vpa.add_return_transition("deep", "top", "r", "d0")
    return vpa


# --------------------------------------------------------------------------- VPA


def test_well_matched_vpa_fixture() -> None:
    assert vpa_truth(well_matched_vpa()) == frozenset(w for w in WORDS if well_matched(w))


@given(vpas())
def test_vpa_recognizes_matches_oracle(a: VisiblyPushdownAutomaton) -> None:
    assert vpa_lang(a) == vpa_truth(a)


@given(vpas(), vpas())
def test_vpa_boolean_operations_match_brute_force(a: VisiblyPushdownAutomaton, b: VisiblyPushdownAutomaton) -> None:
    la, lb = vpa_truth(a), vpa_truth(b)
    assert vpa_lang(a.union(b)) == la | lb
    assert vpa_lang(a.intersection(b)) == la & lb
    assert vpa_lang(a.complement()) == EVERYTHING - la
    assert vpa_lang(a.difference(b)) == la - lb


@given(vpas(max_states=2), vpas(max_states=2))
def test_vpa_concat_matches_brute_force(a: VisiblyPushdownAutomaton, b: VisiblyPushdownAutomaton) -> None:
    assert vpa_lang(a.concat(b)) == concat_of(vpa_truth(a), vpa_truth(b))


@given(vpas(max_states=2))
def test_vpa_kleene_star_matches_brute_force(a: VisiblyPushdownAutomaton) -> None:
    assert vpa_lang(a.kleene_star()) == star_of(vpa_truth(a))


@given(vpas())
def test_vpa_determinize_preserves_language(a: VisiblyPushdownAutomaton) -> None:
    det = a.determinize()
    det.validate()
    assert vpa_lang(det) == vpa_truth(a)
    assert det.equivalent(a)


@given(vpas())
def test_vpa_emptiness_and_witness(a: VisiblyPushdownAutomaton) -> None:
    accepted = vpa_truth(a)
    witness = a.accepted_word()
    assert (witness is None) == a.is_empty()
    if witness is not None:
        assert oracles.vpa_accepts(a, witness)
    else:
        assert not accepted
    if accepted:
        assert not a.is_empty()


@given(vpas())
def test_vpa_universality(a: VisiblyPushdownAutomaton) -> None:
    accepted = vpa_truth(a)
    if a.is_universal():
        assert accepted == EVERYTHING
    if accepted != EVERYTHING:
        assert not a.is_universal()


@given(vpas(), vpas())
def test_vpa_inclusion_and_equivalence(a: VisiblyPushdownAutomaton, b: VisiblyPushdownAutomaton) -> None:
    la, lb = vpa_truth(a), vpa_truth(b)
    if a.includes(b):
        assert lb <= la
    if not lb <= la:
        assert not a.includes(b)
    if a.equivalent(b):
        assert la == lb
    if la != lb:
        assert not a.equivalent(b)
    assert a.includes(a.intersection(b))
    assert a.union(b).includes(b)


@given(vpas())
def test_vpa_unmatched_word_detection(a: VisiblyPushdownAutomaton) -> None:
    accepted = vpa_truth(a)
    if any(not well_matched(w) for w in accepted):
        assert a.has_unmatched_word()
    if not a.has_unmatched_word():
        assert all(well_matched(w) for w in accepted)


@given(vpas())
def test_modular_conversions_on_well_matched_languages(a: VisiblyPushdownAutomaton) -> None:
    restricted = a.intersection(well_matched_vpa())
    assert not restricted.has_unmatched_word()
    expected = vpa_truth(a) & vpa_truth(well_matched_vpa())
    single = to_single_entry(restricted)
    multiple = to_multiple_entry(restricted)
    assert vpa_lang(single) == expected
    assert vpa_lang(multiple) == expected


@given(vpas())
def test_modular_conversion_rejects_unmatched_languages(a: VisiblyPushdownAutomaton) -> None:
    assume(a.has_unmatched_word())
    with pytest.raises(NonWellMatchedLanguageError):
        to_single_entry(a)
    with pytest.raises(NonWellMatchedLanguageError):
        to_multiple_entry(a)


# --------------------------------------------------------------------------- NWA


@given(nwas())
def test_nwa_recognizes_matches_oracle(a: NestedWordAutomaton) -> None:
    assert nwa_lang(a) == nwa_truth(a)


@given(nwas())
def test_nwa_vpa_round_trips(a: NestedWordAutomaton) -> None:
    expected = nwa_truth(a)
    assert vpa_lang(a.to_vpa(tag_symbols=False)) == expected
    assert nwa_lang(NestedWordAutomaton.from_vpa(a.to_vpa(tag_symbols=False))) == expected


@given(vpas())
def test_nwa_from_vpa_preserves_language(a: VisiblyPushdownAutomaton) -> None:
    assert nwa_lang(NestedWordAutomaton.from_vpa(a)) == vpa_truth(a)


@given(nwas(), nwas())
def test_nwa_boolean_operations_match_brute_force(a: NestedWordAutomaton, b: NestedWordAutomaton) -> None:
    la, lb = nwa_truth(a), nwa_truth(b)
    assert nwa_lang(a.union(b)) == la | lb
    assert nwa_lang(a.intersection(b)) == la & lb
    assert nwa_lang(a.complement()) == EVERYTHING - la
    assert nwa_lang(a.difference(b)) == la - lb


@given(nwas(max_states=2), nwas(max_states=2))
def test_nwa_concat_and_star_match_brute_force(a: NestedWordAutomaton, b: NestedWordAutomaton) -> None:
    la, lb = nwa_truth(a), nwa_truth(b)
    assert nwa_lang(a.concat(b)) == concat_of(la, lb)
    assert nwa_lang(a.kleene_star()) == star_of(la)


@given(nwas(), nwas())
def test_nwa_decision_procedures(a: NestedWordAutomaton, b: NestedWordAutomaton) -> None:
    la, lb = nwa_truth(a), nwa_truth(b)
    if la:
        assert not a.is_empty()
    if a.is_empty():
        assert not la
    if la != EVERYTHING:
        assert not a.is_universal()
    if a.is_universal():
        assert la == EVERYTHING
    if not lb <= la:
        assert not a.includes(b)
    if la != lb:
        assert not a.equivalent(b)
    assert a.equivalent(a.union(a.intersection(b)))
    assert a.union(b).includes(a)


# --------------------------------------------------------------------------- PAPNI encoding

DYCK = DyckAlphabet(
    call_alphabet=frozenset({"(", "["}),
    return_alphabet=frozenset({")", "]"}),
    internal_alphabet=frozenset({"i"}),
)
DYCK_SYMBOLS = ("(", "[", ")", "]", "i")


@st.composite
def dyck_words(draw: Any) -> Word:
    tokens = draw(st.lists(st.sampled_from(("open", "close", "internal")), max_size=8))
    word: list[str] = []
    depth = 0
    for token in tokens:
        if token == "open":
            word.append(draw(st.sampled_from(("(", "["))))
            depth += 1
        elif token == "close" and depth:
            word.append(draw(st.sampled_from((")", "]"))))
            depth -= 1
        elif token == "internal":
            word.append("i")
    word.extend(draw(st.sampled_from((")", "]"))) for _ in range(depth))
    return tuple(word)


def counter_well_matched(w: Word) -> bool:
    depth = 0
    for symbol in w:
        if symbol in DYCK.call_alphabet:
            depth += 1
        elif symbol in DYCK.return_alphabet:
            depth -= 1
            if depth < 0:
                return False
        elif symbol not in DYCK.internal_alphabet:
            return False
    return depth == 0


any_word = st.lists(st.sampled_from(DYCK_SYMBOLS), max_size=6).map(tuple)


@given(any_word | dyck_words())
def test_is_well_matched_matches_counter(w: Word) -> None:
    assert is_well_matched(w, DYCK) == counter_well_matched(w)


@given(dyck_words())
def test_encode_dyck_word_pairs_returns_with_matching_calls(w: Word) -> None:
    assert is_well_matched(w, DYCK)
    encoded = encode_dyck_word(w, DYCK)
    assert len(encoded) == len(w)
    stack: list[str] = []
    for symbol, code in zip(w, encoded, strict=True):
        if symbol in DYCK.return_alphabet:
            assert code == (symbol, stack.pop())
        else:
            assert code == symbol
            if symbol in DYCK.call_alphabet:
                stack.append(symbol)
    decoded = tuple(code[0] if isinstance(code, tuple) else code for code in encoded)
    assert decoded == w


@given(st.lists(any_word | dyck_words(), max_size=6))
def test_encode_dyck_samples_filters_and_encodes(samples: list[Word]) -> None:
    kept = [encode_dyck_word(w, DYCK) for w in samples if counter_well_matched(w)]
    assert encode_dyck_samples(samples, DYCK) == kept
    if all(counter_well_matched(w) for w in samples):
        assert encode_dyck_samples(samples, DYCK, drop_non_well_matched=False) == kept
    else:
        with pytest.raises(ValueError):
            encode_dyck_samples(samples, DYCK, drop_non_well_matched=False)
        with pytest.raises(ValueError):
            encode_dyck_word(next(w for w in samples if not counter_well_matched(w)), DYCK)
