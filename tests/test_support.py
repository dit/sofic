"""Stationary support languages and the decisions built on them."""

from __future__ import annotations

import math
from itertools import product

import pytest
from hypothesis import given, settings

from sofic.examples import bernoulli, binary_markov_chain, even_process, golden_mean
from sofic.generators.mealy import MealyHMM
from sofic.generators.relative_entropy_rate import relative_entropy_rate
from sofic.generators.support import is_absolutely_continuous, support_equal, support_includes, support_nfa
from sofic.testing.strategies import mealy_hmms

POSITIVE = 1e-12


def _transient_prefix() -> MealyHMM:
    """A transient state emits one ``1`` and then the chain emits ``0`` forever."""
    hmm = MealyHMM(observation_alphabet=frozenset({"0", "1"}), initial_distribution={"T": 1.0})
    hmm.add_transition("T", "A", "1", 1.0)
    hmm.add_transition("A", "A", "0", 1.0)
    return hmm


def _period_two() -> MealyHMM:
    hmm = MealyHMM(observation_alphabet=frozenset({"0", "1"}), initial_distribution={"A": 0.5, "B": 0.5})
    hmm.add_transition("A", "B", "0", 1.0)
    hmm.add_transition("B", "A", "1", 1.0)
    return hmm


MODELS = {
    "iid": bernoulli(0.3),
    "golden": golden_mean(0.4),
    "golden_parry": golden_mean(2 / (1 + math.sqrt(5))),
    "even": even_process(0.6),
    "markov": binary_markov_chain(0.2, 0.7),
    "period_two": _period_two(),
    "transient": _transient_prefix(),
}


def _positive_words(model, max_length: int) -> set[tuple[str, ...]]:
    pi = model.to_mealy().stationary_distribution()
    alphabet = sorted(model.observation_alphabet)
    return {
        word
        for length in range(max_length + 1)
        for word in product(alphabet, repeat=length)
        if model.word_probability(word, start=pi) > POSITIVE
    }


def _brute_includes(p, q, max_length: int) -> bool:
    return _positive_words(p, max_length) <= _positive_words(q, max_length)


@pytest.mark.parametrize("name", sorted(MODELS))
def test_support_nfa_accepts_exactly_the_positive_words(name):
    model = MODELS[name]
    nfa = support_nfa(model)
    positive = _positive_words(model, 6)
    for length in range(7):
        for word in product(("0", "1"), repeat=length):
            assert nfa.recognizes(word) == (word in positive)


def test_support_nfa_drops_words_only_transient_states_emit():
    model = _transient_prefix()
    assert model.to_support_nfa().recognizes(("1",))
    assert not support_nfa(model).recognizes(("1",))
    assert support_nfa(model).recognizes(("0", "0", "0"))


@pytest.mark.parametrize("p", sorted(MODELS))
@pytest.mark.parametrize("q", sorted(MODELS))
def test_support_includes_matches_word_enumeration(p, q):
    expected = _brute_includes(MODELS[p], MODELS[q], 8)
    assert support_includes(MODELS[p], MODELS[q]) == expected
    assert is_absolutely_continuous(MODELS[p], MODELS[q]) == expected


@pytest.mark.filterwarnings("ignore:stationary distribution is not unique")
@settings(max_examples=40, deadline=None)
@given(mealy_hmms(max_states=2), mealy_hmms(max_states=2))
def test_support_includes_matches_enumeration_on_random_hmms(p, q):
    # A shortest word in L(p) \ L(q) has length below |p states| * 2^|q states| = 8.
    assert support_includes(p, q) == _brute_includes(p, q, 8)


def test_golden_mean_is_below_the_full_shift():
    assert support_includes(golden_mean(0.5), bernoulli())
    assert not support_includes(bernoulli(), golden_mean(0.5))
    assert golden_mean(0.5).is_absolutely_continuous(bernoulli())
    assert not bernoulli().support_includes(golden_mean(0.5))


def test_support_equal_ignores_probabilities():
    assert support_equal(golden_mean(0.3), golden_mean(0.8))
    assert support_equal(bernoulli(0.1), binary_markov_chain(0.2, 0.7))
    assert not support_equal(golden_mean(0.5), bernoulli())
    assert golden_mean(0.3).support_equal(golden_mean(0.6))


@pytest.mark.parametrize("name", sorted(MODELS))
def test_support_includes_is_reflexive(name):
    assert support_includes(MODELS[name], MODELS[name])
    assert support_equal(MODELS[name], MODELS[name])


def test_support_includes_is_transitive():
    names = sorted(MODELS)
    included = {(a, b): support_includes(MODELS[a], MODELS[b]) for a in names for b in names}
    for a, b, c in product(names, repeat=3):
        if included[a, b] and included[b, c]:
            assert included[a, c]


@pytest.mark.parametrize("p", ["iid", "golden", "even", "markov", "period_two"])
@pytest.mark.parametrize("q", ["iid", "golden", "golden_parry", "markov"])
def test_finite_relative_entropy_rate_iff_support_included(p, q):
    assert math.isfinite(relative_entropy_rate(MODELS[p], MODELS[q])) == support_includes(MODELS[p], MODELS[q])
