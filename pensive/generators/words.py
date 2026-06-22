"""Finite-word distributions for stochastic generators."""

from __future__ import annotations

from collections.abc import Hashable
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
