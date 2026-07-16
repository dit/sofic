"""Tests for SAT-based exact minimal DFA identification."""

from __future__ import annotations

import itertools

import pytest

pytest.importorskip("pysat")

from sofic.automata import learn_dfa_edsm, learn_dfa_sat  # noqa: E402

ALPHABET = ("a", "b")


def _labeled(predicate, max_len=5):
    positive, negative = [], []
    for length in range(max_len + 1):
        for word in itertools.product(ALPHABET, repeat=length):
            (positive if predicate(word) else negative).append(word)
    return positive, negative


def _even_a(word):
    return word.count("a") % 2 == 0


def _a_before_b(word):
    return "".join(word).find("ba") == -1


def test_sat_recovers_minimal_parity_dfa():
    positive, negative = _labeled(_even_a)
    dfa = learn_dfa_sat(positive, negative)
    dfa.validate()
    assert len(list(dfa.states())) == 2
    for length in range(8):
        for word in itertools.product(ALPHABET, repeat=length):
            assert dfa.recognizes(word) == _even_a(word)


def test_sat_recovers_a_before_b():
    positive, negative = _labeled(_a_before_b)
    dfa = learn_dfa_sat(positive, negative)
    dfa.validate()
    assert len(list(dfa.states())) == 3


@pytest.mark.parametrize("predicate", [_even_a, _a_before_b])
def test_sat_no_larger_than_edsm(predicate):
    positive, negative = _labeled(predicate)
    sat = learn_dfa_sat(positive, negative)
    edsm = learn_dfa_edsm(positive, negative)
    assert len(list(sat.states())) <= len(list(edsm.states()))


def test_sat_is_consistent_with_sample():
    positive = [("a",), ("a", "b", "a"), ("a", "b", "a", "b", "a")]
    negative = [(), ("b",), ("a", "b"), ("b", "a")]
    dfa = learn_dfa_sat(positive, negative)
    dfa.validate()
    assert all(dfa.recognizes(word) for word in positive)
    assert all(not dfa.recognizes(word) for word in negative)


def test_sat_only_positive_collapses_to_one_state():
    dfa = learn_dfa_sat([("a",), ("a", "a"), ("a", "a", "a")], [])
    assert len(list(dfa.states())) == 1
    assert dfa.recognizes(("a", "a"))


def test_sat_upper_bound_too_small_raises():
    positive, negative = _labeled(_even_a)
    with pytest.raises(RuntimeError):
        learn_dfa_sat(positive, negative, upper_bound=1)


def test_sat_requires_samples():
    with pytest.raises(ValueError):
        learn_dfa_sat([], [])


def test_sat_contradictory_labels():
    with pytest.raises(ValueError):
        learn_dfa_sat([("a",)], [("a",)])
