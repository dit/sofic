import itertools

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from sofic.automata import learn_dfa_k_testable
from sofic.automata.learning import learn_dfa_k_testable as learning_export


def _words(alphabet, max_length):
    for length in range(max_length + 1):
        yield from itertools.product(alphabet, repeat=length)


def _k_tss_member(word, k, prefixes, suffixes, factors, short):
    window = k - 1
    if len(word) < window:
        return word in short
    return (
        word[:window] in prefixes
        and word[len(word) - window :] in suffixes
        and all(word[index : index + k] in factors for index in range(len(word) - k + 1))
    )


def _definition(samples, k):
    words = [tuple(word) for word in samples]
    window = k - 1
    prefixes = {word[:window] for word in words if len(word) >= window}
    suffixes = {word[len(word) - window :] for word in words if len(word) >= window}
    factors = {word[index : index + k] for word in words for index in range(len(word) - k + 1)}
    short = {word for word in words if len(word) < window}
    return prefixes, suffixes, factors, short


_samples = st.lists(st.lists(st.sampled_from("ab"), max_size=6).map(tuple), max_size=6)


def test_exported_from_learning_package():
    assert learn_dfa_k_testable is learning_export


@settings(max_examples=100, deadline=None)
@given(_samples, st.integers(1, 4))
def test_agrees_with_definition(samples, k):
    dfa = learn_dfa_k_testable(samples, k, alphabet="ab")
    sets = _definition(samples, k)
    for word in _words("ab", 7):
        assert dfa.recognizes(word) == _k_tss_member(word, k, *sets), word


@settings(max_examples=100, deadline=None)
@given(_samples, st.integers(1, 4))
def test_contains_samples(samples, k):
    dfa = learn_dfa_k_testable(samples, k)
    assert all(dfa.recognizes(word) for word in samples)


@settings(max_examples=100, deadline=None)
@given(_samples, st.integers(1, 3))
def test_monotone_in_k(samples, k):
    coarse = learn_dfa_k_testable(samples, k, alphabet="ab")
    fine = learn_dfa_k_testable(samples, k + 1, alphabet="ab")
    for word in _words("ab", 7):
        assert not fine.recognizes(word) or coarse.recognizes(word), word


@settings(max_examples=60, deadline=None)
@given(
    st.integers(2, 3).flatmap(
        lambda k: st.tuples(
            st.just(k),
            st.frozensets(st.tuples(*[st.sampled_from("ab")] * (k - 1))),
            st.frozensets(st.tuples(*[st.sampled_from("ab")] * (k - 1))),
            st.frozensets(st.tuples(*[st.sampled_from("ab")] * k)),
            st.frozensets(st.lists(st.sampled_from("ab"), max_size=k - 2).map(tuple)),
        )
    )
)
def test_identifies_target_in_the_limit(target):
    k, prefixes, suffixes, factors, short = target
    language = [word for word in _words("ab", 11) if _k_tss_member(word, k, prefixes, suffixes, factors, short)]
    dfa = learn_dfa_k_testable(language, k, alphabet="ab")
    for word in _words("ab", 8):
        assert dfa.recognizes(word) == _k_tss_member(word, k, prefixes, suffixes, factors, short), word


def test_characteristic_sample_for_ab_star():
    dfa = learn_dfa_k_testable(["", "ab", "abab"], 2)
    for word in _words("ab", 8):
        assert dfa.recognizes(word) == ("".join(word) in {"ab" * n for n in range(5)})


def test_k_equal_one_is_star_of_occurring_symbols():
    dfa = learn_dfa_k_testable(["a", "aa"], 1, alphabet="ab")
    assert dfa.recognizes(())
    assert dfa.recognizes(("a",) * 5)
    assert not dfa.recognizes(("b",))
    assert len(list(dfa.states())) == 1


def test_k_equal_one_empty_word_only():
    dfa = learn_dfa_k_testable([""], 1, alphabet="ab")
    assert dfa.recognizes(())
    assert not dfa.recognizes(("a",))


@pytest.mark.parametrize("k", [1, 2, 3])
def test_empty_sample_gives_empty_language(k):
    dfa = learn_dfa_k_testable([], k, alphabet="ab")
    assert not any(dfa.recognizes(word) for word in _words("ab", 5))


def test_short_words_are_kept_exactly():
    dfa = learn_dfa_k_testable(["a", "abba"], 3)
    assert dfa.recognizes(("a",))
    assert not dfa.recognizes(())
    assert not dfa.recognizes(("b",))
    assert dfa.recognizes(tuple("abba"))


def test_invalid_arguments():
    with pytest.raises(ValueError, match="at least 1"):
        learn_dfa_k_testable(["a"], 0)
    with pytest.raises(ValueError, match="not in the alphabet"):
        learn_dfa_k_testable(["ac"], 2, alphabet="ab")
