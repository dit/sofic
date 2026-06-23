"""Finite-word distributions for stochastic generators."""

from __future__ import annotations

from collections.abc import Hashable, Mapping, Sequence
from itertools import product
from typing import Any

import numpy as np

from pensive.generators.base import HiddenMarkovModel, QuasiStochasticModel
from pensive.generators.hmm_inference import _emission_transition_tensors
from pensive.generators.markov import MarkovChain
from pensive.generators.pfa import ProbabilisticFiniteAutomaton
from pensive.graph import ATTR_PROB

_TOL = 1e-15


def hmm_words_of_length(hmm: HiddenMarkovModel, length: int) -> dict[tuple[Any, ...], float]:
    """Return observed words of ``length`` and their probabilities."""
    if length < 0:
        raise ValueError("length must be nonnegative")
    pi, joint = _emission_transition_tensors(hmm)
    alphabet = sorted(hmm.observation_alphabet, key=repr)
    if length == 0:
        return {(): float(pi.sum())} if pi.sum() > _TOL else {}
    if not alphabet:
        return {}

    terminal = np.ones(len(pi), dtype=float)
    zero = np.zeros((len(pi), len(pi)), dtype=float)
    distribution: dict[tuple[Any, ...], float] = {}
    for word in product(alphabet, repeat=length):
        mass = pi.copy()
        for symbol in word:
            mass = mass @ joint.get(symbol, zero)
        probability = float(mass @ terminal)
        if abs(probability) > _TOL:
            distribution[word] = probability
    return distribution


def hmm_word_probability(
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
    pi, joint = _emission_transition_tensors(mealy)
    mass = _start_vector(mealy, pi, start)
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


def hmm_log_word_probability(
    hmm: HiddenMarkovModel,
    word: Sequence[Any],
    *,
    start: Hashable | Mapping[Hashable, float] | Sequence[float] | np.ndarray | None = None,
) -> float:
    """Return ``log2(P(word))`` or ``-inf`` for forbidden words."""
    probability = hmm_word_probability(hmm, word, start=start)
    if probability <= 0.0:
        return float("-inf")
    return float(np.log2(probability))


def hmm_word_probabilities(
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
            probability = hmm_word_probability(mealy, (), start=start)
            if not sparse or abs(probability) > _TOL:
                distribution[()] = probability
            continue
        for word in product(alphabet, repeat=length):
            probability = hmm_word_probability(mealy, word, start=start)
            if not sparse or abs(probability) > _TOL:
                distribution[word] = probability
    return distribution


def hmm_conditional_word_probability(
    hmm: HiddenMarkovModel,
    word: Sequence[Any],
    condition: Sequence[Any],
    *,
    start: Hashable | Mapping[Hashable, float] | Sequence[float] | np.ndarray | None = None,
) -> float:
    """Return ``P(word | condition)`` from the requested start distribution."""
    condition_probability = hmm_word_probability(hmm, condition, start=start)
    if condition_probability <= _TOL:
        raise ZeroDivisionError("condition has zero probability")
    joint_word = tuple(condition) + tuple(word)
    return hmm_word_probability(hmm, joint_word, start=start) / condition_probability


def pfa_words_of_length(pfa: ProbabilisticFiniteAutomaton, length: int) -> dict[tuple[Any, ...], float]:
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
    for word in product(alphabet, repeat=length):
        probability = pfa.string_probability(word)
        if probability > _TOL:
            distribution[word] = probability
    return distribution


def quasi_words_of_length(model: QuasiStochasticModel, length: int) -> dict[tuple[Any, ...], float]:
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
    for word in product(sorted(alphabet, key=repr), repeat=length):
        probability = float(model.word_probability(word))
        if abs(probability) > _TOL:
            distribution[word] = probability
    return distribution


def markov_words_of_length(chain: MarkovChain, length: int) -> dict[tuple[Hashable, ...], float]:
    """Return visible state paths of ``length`` and their probabilities."""
    if length < 0:
        raise ValueError("length must be nonnegative")
    states = tuple(chain.states())
    if length == 0:
        return {(): 1.0}
    distribution: dict[tuple[Hashable, ...], float] = {}
    for word in product(states, repeat=length):
        probability = _markov_path_probability(chain, word)
        if probability > _TOL:
            distribution[word] = probability
    return distribution


def _quasi_alphabet(model: QuasiStochasticModel) -> tuple[Any, ...]:
    for name in ("observation_alphabet", "output_alphabet"):
        alphabet = getattr(model, name, None)
        if alphabet:
            return tuple(alphabet)
    return tuple(model.transition_matrices())


def _start_vector(
    hmm: HiddenMarkovModel,
    default: np.ndarray,
    start: Hashable | Mapping[Hashable, float] | Sequence[float] | np.ndarray | None,
) -> np.ndarray:
    if start is None:
        return np.array(default, dtype=float)

    idx = hmm.reindex()
    n = len(idx)
    if isinstance(start, Mapping):
        vector = np.zeros(n, dtype=float)
        for state, mass in start.items():
            if not hmm.graph.has_state(state):
                raise ValueError(f"unknown start state {state!r}")
            vector[idx.index(state)] = float(mass)
        return vector

    if hmm.graph.has_state(start):
        vector = np.zeros(n, dtype=float)
        vector[idx.index(start)] = 1.0
        return vector

    vector = np.asarray(start, dtype=float)
    if vector.shape != (n,):
        raise ValueError(f"start vector must have shape {(n,)}, got {vector.shape}")
    return vector.copy()


def _markov_path_probability(chain: MarkovChain, path: tuple[Hashable, ...]) -> float:
    if not path:
        return 1.0
    probability = float(chain.initial_distribution.get(path[0], 0.0))
    for source, target in zip(path, path[1:], strict=False):
        edge_probability = 0.0
        for transition in chain.graph.out_transitions(source):
            if transition.target == target:
                edge_probability += float(transition.data.get(ATTR_PROB, 0.0))
        probability *= edge_probability
        if probability <= _TOL:
            return 0.0
    return probability
