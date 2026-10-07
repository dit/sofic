"""Finite-word distributions for stochastic generators."""

from __future__ import annotations

from collections.abc import Callable, Hashable, Iterator, Mapping, Sequence
from typing import Any

import numpy as np

from sofic.generators.base import HiddenMarkovModel, QuasiStochasticModel
from sofic.generators.markov import MarkovChain
from sofic.generators.matrices import emission_tensors, start_vector, symbol_matrices
from sofic.generators.pfa import ProbabilisticFiniteAutomaton
from sofic.graph import ATTR_PROB

_TOL = 1e-15


def _enumerate_words(
    alphabet: Sequence[Any],
    length: int,
    start: Any = None,
    step: Callable[[Any, Any], Any] | None = None,
) -> Iterator[tuple[tuple[Any, ...], Any]]:
    """Yield ``(word, mass)`` for every word of ``length`` over ``alphabet``.

    Words come in lexicographic (``itertools.product``) order of ``alphabet``.
    ``mass`` starts at ``start`` and is advanced by ``step(mass, symbol)`` once per
    symbol, so prefixes are propagated once rather than per word; a step returning
    ``None`` prunes every extension of that prefix. With ``step=None`` the mass is
    carried unchanged and every word is yielded.
    """

    def walk(prefix: tuple[Any, ...], mass: Any, remaining: int) -> Iterator[tuple[tuple[Any, ...], Any]]:
        if remaining == 0:
            yield prefix, mass
            return
        for symbol in alphabet:
            if step is None:
                yield from walk(prefix + (symbol,), mass, remaining - 1)
                continue
            nxt = step(mass, symbol)
            if nxt is not None:
                yield from walk(prefix + (symbol,), nxt, remaining - 1)

    yield from walk((), start, length)


def _matrix_step(matrices: Mapping[Any, np.ndarray], n: int, *, prune: bool = True) -> Callable[[np.ndarray, Any], Any]:
    """Return a :func:`_enumerate_words` step multiplying a row mass by ``matrices[symbol]``.

    Symbols without a matrix act as the zero matrix. With ``prune`` a numeric
    prefix whose mass vector is identically zero is dropped.
    """
    zero = np.zeros((n, n), dtype=float)

    def step(mass: np.ndarray, symbol: Any) -> np.ndarray | None:
        nxt = mass @ matrices.get(symbol, zero)
        if prune and nxt.dtype != object and not np.any(nxt):
            return None
        return nxt

    return step


def _hmm_words_of_length(hmm: HiddenMarkovModel, length: int) -> dict[tuple[Any, ...], float]:
    """Return observed words of ``length`` and their probabilities."""
    if length < 0:
        raise ValueError("length must be nonnegative")
    pi, joint = emission_tensors(hmm)
    alphabet = sorted(hmm.observation_alphabet, key=repr)
    if length == 0:
        return {(): float(pi.sum())} if pi.sum() > _TOL else {}
    if not alphabet:
        return {}

    terminal = np.ones(len(pi), dtype=float)
    distribution: dict[tuple[Any, ...], float] = {}
    for word, mass in _enumerate_words(alphabet, length, pi.copy(), _matrix_step(joint, len(pi))):
        probability = float(mass @ terminal)
        if abs(probability) > _TOL:
            distribution[word] = probability
    return distribution


def _hmm_word_probability(
    hmm: HiddenMarkovModel,
    word: Sequence[Any],
    *,
    start: Hashable | Mapping[Hashable, float] | Sequence[float] | np.ndarray | None = None,
) -> float:
    """Return the probability of an observed ``word`` from ``start``.

    ``start`` may be ``None`` (use the model's initial distribution), a state,
    a state-probability mapping, or a dense vector in the model's state order.
    """
    mealy = hmm.to_mealy()
    joint = symbol_matrices(mealy)
    mass = np.asarray(start_vector(mealy, start), dtype=float)
    if len(word) == 0:
        return float(mass.sum())
    n = len(mass)
    zero = np.zeros((n, n), dtype=float)
    for symbol in word:
        matrix = joint.get(symbol, zero)
        mass = mass @ matrix
        if not np.any(np.abs(mass) > _TOL):
            return 0.0
    return float(mass.sum())


def _hmm_log_word_probability(
    hmm: HiddenMarkovModel,
    word: Sequence[Any],
    *,
    start: Hashable | Mapping[Hashable, float] | Sequence[float] | np.ndarray | None = None,
) -> float:
    """Return ``log2(P(word))`` or ``-inf`` for forbidden words."""
    probability = _hmm_word_probability(hmm, word, start=start)
    if probability <= 0.0:
        return float("-inf")
    return float(np.log2(probability))


def _hmm_word_probabilities(
    hmm: HiddenMarkovModel,
    lengths: int | Sequence[int],
    *,
    start: Hashable | Mapping[Hashable, float] | Sequence[float] | np.ndarray | None = None,
    sparse: bool = True,
) -> dict[tuple[Any, ...], float]:
    """Return probabilities for all observed words at the requested lengths."""
    requested = (lengths,) if isinstance(lengths, int) else tuple(lengths)
    if any(length < 0 for length in requested):
        raise ValueError("lengths must be nonnegative")

    mealy = hmm.to_mealy()
    alphabet = sorted(mealy.observation_alphabet, key=repr)
    distribution: dict[tuple[Any, ...], float] = {}
    for length in requested:
        if length == 0:
            probability = _hmm_word_probability(mealy, (), start=start)
            if not sparse or abs(probability) > _TOL:
                distribution[()] = probability
            continue
        for word, _ in _enumerate_words(alphabet, length):
            probability = _hmm_word_probability(mealy, word, start=start)
            if not sparse or abs(probability) > _TOL:
                distribution[word] = probability
    return distribution


def _hmm_conditional_word_probability(
    hmm: HiddenMarkovModel,
    word: Sequence[Any],
    condition: Sequence[Any],
    *,
    start: Hashable | Mapping[Hashable, float] | Sequence[float] | np.ndarray | None = None,
) -> float:
    """Return ``P(word | condition)`` from the requested start distribution."""
    condition_probability = _hmm_word_probability(hmm, condition, start=start)
    if condition_probability <= _TOL:
        raise ZeroDivisionError("condition has zero probability")
    joint_word = tuple(condition) + tuple(word)
    return _hmm_word_probability(hmm, joint_word, start=start) / condition_probability


def _pfa_words_of_length(pfa: ProbabilisticFiniteAutomaton, length: int) -> dict[tuple[Any, ...], float]:
    """Return output words of ``length`` and their probabilities."""
    if length < 0:
        raise ValueError("length must be nonnegative")
    alphabet = sorted(pfa.output_alphabet, key=repr)
    if length == 0:
        probability = pfa.string_probability(())
        return {(): probability} if probability > _TOL else {}
    if not alphabet:
        return {}
    distribution: dict[tuple[Any, ...], float] = {}
    for word, _ in _enumerate_words(alphabet, length):
        probability = pfa.string_probability(word)
        if probability > _TOL:
            distribution[word] = probability
    return distribution


def _quasi_words_of_length(model: QuasiStochasticModel, length: int) -> dict[tuple[Any, ...], float]:
    """Return words of ``length`` and their signed quasiprobabilities."""
    if length < 0:
        raise ValueError("length must be nonnegative")
    alphabet = _quasi_alphabet(model)
    if length == 0:
        probability = float(model.word_probability(()))
        return {(): probability} if abs(probability) > _TOL else {}
    if not alphabet:
        return {}
    distribution: dict[tuple[Any, ...], float] = {}
    for word, _ in _enumerate_words(sorted(alphabet, key=repr), length):
        probability = float(model.word_probability(word))
        if abs(probability) > _TOL:
            distribution[word] = probability
    return distribution


def _markov_words_of_length(chain: MarkovChain, length: int) -> dict[tuple[Hashable, ...], float]:
    """Return visible state paths of ``length`` and their probabilities."""
    if length < 0:
        raise ValueError("length must be nonnegative")
    states = tuple(chain.states())
    start = _markov_start(chain)
    if length == 0:
        total = float(sum(start.values()))
        return {(): total} if total > _TOL else {}
    distribution: dict[tuple[Hashable, ...], float] = {}
    for word, _ in _enumerate_words(states, length):
        probability = _markov_path_probability(chain, word, start)
        if probability > _TOL:
            distribution[word] = probability
    return distribution


def _quasi_alphabet(model: QuasiStochasticModel) -> tuple[Any, ...]:
    for name in ("observation_alphabet", "output_alphabet"):
        alphabet = getattr(model, name, None)
        if alphabet:
            return tuple(alphabet)
    return tuple(model.symbol_matrices())


def _markov_start(chain: MarkovChain) -> dict[Hashable, float]:
    """Initial law of ``chain``, or its stationary law when none is given."""
    if chain.initial_distribution:
        return {state: float(mass) for state, mass in chain.initial_distribution.items()}
    idx = chain.reindex()
    pi = chain.stationary_distribution()
    return {idx.state(i): float(mass) for i, mass in enumerate(pi)}


def _markov_path_probability(chain: MarkovChain, path: tuple[Hashable, ...], start: Mapping[Hashable, float]) -> float:
    if not path:
        return float(sum(start.values()))
    probability = float(start.get(path[0], 0.0))
    for source, target in zip(path, path[1:], strict=False):
        edge_probability = 0.0
        for transition in chain.graph.out_transitions(source):
            if transition.target == target:
                edge_probability += float(transition.data.get(ATTR_PROB, 0.0))
        probability *= edge_probability
        if probability <= _TOL:
            return 0.0
    return probability
