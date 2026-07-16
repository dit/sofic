"""Tests for state-merging automata learners: EDSM and ALERGIA."""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from sofic.automata import learn_dfa_edsm, learn_pfa_alergia
from sofic.examples import bernoulli, even_process, golden_mean


def _labeled(predicate, alphabet=("a", "b"), max_len=5):
    positive, negative = [], []
    for length in range(max_len + 1):
        for word in itertools.product(alphabet, repeat=length):
            (positive if predicate(word) else negative).append(word)
    return positive, negative


# --------------------------------------------------------------------------- EDSM


def test_edsm_parity_recovers_minimal_dfa():
    def even_number_of_a(word):
        return word.count("a") % 2 == 0

    positive, negative = _labeled(even_number_of_a)
    dfa = learn_dfa_edsm(positive, negative)
    dfa.validate()

    assert len(list(dfa.states())) == 2
    for length in range(8):
        for word in itertools.product("ab", repeat=length):
            assert dfa.recognizes(word) == even_number_of_a(word)


def test_edsm_a_before_b():
    def a_before_b(word):
        return "".join(word).find("ba") == -1

    positive, negative = _labeled(a_before_b)
    dfa = learn_dfa_edsm(positive, negative)
    dfa.validate()

    for length in range(7):
        for word in itertools.product("ab", repeat=length):
            assert dfa.recognizes(word) == a_before_b(word)


def test_edsm_consistent_with_sample():
    positive = [("a",), ("a", "b", "a"), ("a", "b", "a", "b", "a")]
    negative = [(), ("b",), ("a", "b"), ("b", "a")]
    dfa = learn_dfa_edsm(positive, negative)
    dfa.validate()
    for word in positive:
        assert dfa.recognizes(word)
    for word in negative:
        assert not dfa.recognizes(word)


def test_edsm_no_more_states_than_rpni_on_parity():
    from sofic.automata import learn_dfa_rpni

    def even_number_of_a(word):
        return word.count("a") % 2 == 0

    positive, negative = _labeled(even_number_of_a)
    edsm = learn_dfa_edsm(positive, negative)
    rpni = learn_dfa_rpni(positive, negative)
    assert len(list(edsm.states())) <= len(list(rpni.states()))


def test_edsm_requires_samples():
    with pytest.raises(ValueError):
        learn_dfa_edsm([], [])


def test_edsm_contradictory_labels():
    with pytest.raises(ValueError):
        learn_dfa_edsm([("a",)], [("a",)])


# ------------------------------------------------------------------------ ALERGIA


def _variable_samples(model, n_samples, rng, min_len=1, max_len=14):
    samples = []
    for _ in range(n_samples):
        length = int(rng.integers(min_len, max_len + 1))
        observations, _states = model.sample(length, rng=rng)
        samples.append(observations)
    return samples


def test_alergia_returns_valid_pfa():
    rng = np.random.default_rng(0)
    samples = _variable_samples(golden_mean(0.3), 3000, rng)
    pfa = learn_pfa_alergia(samples, alpha=0.05)
    pfa.validate()
    assert len(list(pfa.states())) >= 1


def test_alergia_iid_collapses():
    rng = np.random.default_rng(1)
    samples = _variable_samples(bernoulli(0.5), 4000, rng)
    pfa = learn_pfa_alergia(samples, alpha=0.001)
    pfa.validate()
    assert len(list(pfa.states())) <= 3


def test_alergia_alpha_monotone_in_state_count():
    rng = np.random.default_rng(2)
    samples = _variable_samples(golden_mean(0.3), 5000, rng)
    small = len(list(learn_pfa_alergia(samples, alpha=0.001).states()))
    large = len(list(learn_pfa_alergia(samples, alpha=0.2).states()))
    assert small <= large


def test_alergia_even_process_merges():
    rng = np.random.default_rng(3)
    samples = _variable_samples(even_process(0.5), 5000, rng)
    pfa = learn_pfa_alergia(samples, alpha=0.001)
    pfa.validate()
    # Far fewer states than the raw prefix tree of the samples.
    assert len(list(pfa.states())) <= 10


def test_alergia_requires_samples():
    with pytest.raises(ValueError):
        learn_pfa_alergia([])


def test_alergia_rejects_out_of_range_alpha():
    with pytest.raises(ValueError):
        learn_pfa_alergia([("0",)], alpha=0.0)
    with pytest.raises(ValueError):
        learn_pfa_alergia([("0",)], alpha=1.0)
