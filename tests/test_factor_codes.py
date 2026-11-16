"""Sliding block codes on processes: image processes and higher block presentations.

Word distributions are checked against brute-force pushforwards of the
path-enumeration oracle in :mod:`tests.oracles`.
"""

from __future__ import annotations

import math
from collections import defaultdict
from itertools import product
from typing import Any

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from sofic.examples import golden_mean
from sofic.exceptions import MixedStateExplosionError
from sofic.generators.epsilon_machine import EpsilonMachine
from sofic.generators.factor_codes import higher_block, image_process
from sofic.generators.measures import entropy_rate_bounds
from sofic.graph import ATTR_EMISSION, ATTR_PROB
from sofic.shifts.sliding_block_code import SlidingBlockCode
from sofic.testing.strategies import epsilon_machines, mealy_hmms
from tests import oracles

ALPHABET = ("0", "1")
OUTPUTS = ("a", "b", "c")
TOL = 1e-9

small_hmms = mealy_hmms(alphabet=ALPHABET, max_states=3)
small_eps = epsilon_machines(alphabet=ALPHABET, max_states=3)
reversible_eps = small_eps.filter(lambda m: m.reverse_is_finite())
heavy = settings(max_examples=max(10, settings.default.max_examples // 4))


@st.composite
def codes(draw: Any, max_window: int = 3) -> SlidingBlockCode:
    window = draw(st.integers(1, max_window))
    memory = draw(st.integers(0, window - 1))
    blocks = list(product(ALPHABET, repeat=window))
    outputs = draw(st.lists(st.sampled_from(OUTPUTS), min_size=len(blocks), max_size=len(blocks)))
    return SlidingBlockCode(dict(zip(blocks, outputs, strict=True)), memory=memory, anticipation=window - 1 - memory)


def pushforward(hmm: Any, code: SlidingBlockCode, n: int) -> dict[tuple[Any, ...], float]:
    """Law of ``Phi(X_{0:n+w-1})`` by brute-force enumeration of source words."""
    image: dict[tuple[Any, ...], float] = defaultdict(float)
    for word, prob in oracles.word_distribution(hmm, n + code.window - 1, ALPHABET).items():
        image[code.apply_word(word)] += prob
    return image


def entropy(distribution: dict[Any, float]) -> float:
    return -sum(p * math.log2(p) for p in distribution.values() if p > 0.0)


def state_and_context_entropy(machine: EpsilonMachine, length: int) -> float:
    """``H[S_0, X_{-length:0}]`` for a stationary unifilar machine."""
    frontier = {(state, ()): float(mass) for state, mass in machine.initial_distribution.items()}
    for _ in range(length):
        advanced: dict[tuple[Any, tuple[Any, ...]], float] = defaultdict(float)
        for (state, word), mass in frontier.items():
            for edge in machine.graph.out_transitions(state):
                advanced[(edge.target, (*word, edge.data[ATTR_EMISSION]))] += mass * float(edge.data[ATTR_PROB])
        frontier = advanced
    return entropy(frontier)


# --------------------------------------------------------------------------- image processes


@given(small_hmms, codes(), st.integers(0, 2))
def test_image_word_distribution_is_pushforward(hmm, code, n):
    image = code.apply_to_process(hmm)
    expected = pushforward(hmm, code, n)
    for word in product(OUTPUTS, repeat=n):
        assert image.word_probability(word) == pytest.approx(expected.get(word, 0.0), abs=TOL)


@given(small_hmms, codes(), st.integers(1, 3))
def test_image_block_entropy_bounded_by_source_window(hmm, code, n):
    image = code.apply_to_process(hmm)
    source = oracles.word_distribution(hmm, n + code.window - 1, ALPHABET)
    assert oracles.block_entropy(image.words_of_length(n)) <= oracles.block_entropy(source) + TOL


@heavy
@given(small_eps, codes(max_window=2))
def test_factor_code_does_not_increase_entropy_rate(machine, code):
    image = code.apply_to_process(machine)
    h_source = float(machine.entropy_rate())
    lower, _upper = entropy_rate_bounds(image, 8)
    assert lower <= h_source + TOL
    try:
        h_image = float(EpsilonMachine.from_hmm(image, max_states=500).entropy_rate())
    except MixedStateExplosionError:
        return
    assert h_image <= h_source + TOL


def test_image_matches_apply_word_on_golden_mean():
    xor = SlidingBlockCode({(a, b): str(int(a) ^ int(b)) for a, b in product(ALPHABET, repeat=2)}, memory=1)
    image = xor.apply_to_process(golden_mean(0.5))
    assert image.words_of_length(2) == pytest.approx(
        {("0", "0"): 1 / 6, ("0", "1"): 1 / 6, ("1", "0"): 1 / 6, ("1", "1"): 1 / 2}
    )


def test_constant_code_collapses_entropy_rate():
    machine = golden_mean(0.5)
    constant = SlidingBlockCode({(a,): "*" for a in ALPHABET})
    image = constant.apply_to_process(machine)
    assert image.observation_alphabet == frozenset({"*"})
    assert float(image.entropy_rate()) == pytest.approx(0.0, abs=TOL)


def test_image_is_stationary_for_stationary_source():
    code = SlidingBlockCode({word: word.count("1") for word in product(ALPHABET, repeat=3)}, memory=1, anticipation=1)
    image = image_process(golden_mean(0.5), code)
    assert image.is_stationary()


def test_missing_block_raises():
    partial = SlidingBlockCode({("0",): "a"}, input_alphabet=ALPHABET)
    with pytest.raises(ValueError, match="not in block_map"):
        partial.apply_to_process(golden_mean(0.5))


# --------------------------------------------------------------------------- higher block presentations


@given(small_hmms, st.integers(1, 3), st.integers(0, 2))
def test_higher_block_words_are_recoded_source_words(hmm, k, n):
    recoded = hmm.higher_block(k)
    source = oracles.word_distribution(hmm, n + k - 1, ALPHABET) if n else {(): 1.0}
    expected = {tuple(word[t : t + k] for t in range(n)): prob for word, prob in source.items() if prob > 0.0}
    observed = recoded.words_of_length(n)
    assert set(observed) == set(expected)
    for word, prob in expected.items():
        assert observed[word] == pytest.approx(prob, abs=TOL)


@given(small_hmms, st.integers(1, 3))
def test_one_block_code_inverts_higher_block(hmm, k):
    recoded = higher_block(hmm, k)
    first = SlidingBlockCode(
        {(block,): block[0] for block in recoded.observation_alphabet}, output_alphabet=hmm.observation_alphabet
    )
    assert first.apply_to_process(recoded).is_equal_process(hmm)


@given(small_eps, st.integers(1, 3))
def test_higher_block_preserves_unifilarity_and_entropy_rate(machine, k):
    recoded = machine.higher_block(k)
    assert recoded.is_unifilar()
    assert float(recoded.entropy_rate()) == pytest.approx(float(machine.entropy_rate()), abs=1e-9)


@heavy
@given(reversible_eps, st.integers(1, 3))
def test_higher_block_shifts_excess_entropy_by_k_minus_one_entropy_rates(machine, k):
    recoded = EpsilonMachine.from_hmm(machine.higher_block(k))
    h = float(machine.entropy_rate())
    assert float(recoded.excess_entropy()) == pytest.approx(float(machine.excess_entropy()) + (k - 1) * h, abs=1e-6)


@given(small_eps, st.integers(1, 3))
def test_higher_block_statistical_complexity_is_state_and_context_entropy(machine, k):
    recoded = EpsilonMachine.from_hmm(machine.higher_block(k))
    expected = state_and_context_entropy(machine, k - 1)
    assert float(recoded.statistical_complexity()) == pytest.approx(expected, abs=1e-6)
    assert float(recoded.statistical_complexity()) >= float(machine.statistical_complexity()) - 1e-9


def test_higher_block_rejects_nonpositive_k():
    with pytest.raises(ValueError, match="at least 1"):
        golden_mean(0.5).higher_block(0)
