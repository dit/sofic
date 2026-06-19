"""Forward/backward/Viterbi and sampling for hidden Markov models."""

from __future__ import annotations

from collections.abc import Hashable, Sequence
from typing import Any

import numpy as np

from pensive.generators.base import HiddenMarkovModel
from pensive.graph import ATTR_EMISSION, ATTR_PROB


def _as_mealy_hmm(hmm: HiddenMarkovModel) -> Any:
    """Return a Mealy-style representation through the HMM representation hook."""
    return hmm.to_mealy()


def _emission_transition_tensors_from_mealy(
    hmm: Any,
) -> tuple[np.ndarray, dict[Any, np.ndarray]]:
    """Return initial vector ``pi`` and symbol -> joint transition matrices."""
    idx = hmm.reindex()
    n = len(idx)
    pi = np.zeros(n, dtype=float)
    for state, mass in hmm.initial_distribution.items():
        pi[idx.index(state)] = float(mass)

    symbols: set[Any] = set(hmm.observation_alphabet)
    joint: dict[Any, np.ndarray] = {symbol: np.zeros((n, n), dtype=float) for symbol in symbols}

    for transition in hmm.transitions():
        emission = transition.data.get(ATTR_EMISSION)
        if emission is None:
            continue
        i = idx.index(transition.source)
        j = idx.index(transition.target)
        joint[emission][i, j] += float(transition.data.get(ATTR_PROB, 0.0))
    return pi, joint


def _emission_transition_tensors(
    hmm: HiddenMarkovModel,
) -> tuple[np.ndarray, dict[Any, np.ndarray]]:
    """Return initial vector ``pi`` and symbol -> joint transition matrices."""
    return _emission_transition_tensors_from_mealy(_as_mealy_hmm(hmm))


def forward(hmm: HiddenMarkovModel, observations: Sequence[Any]) -> np.ndarray:
    """Return forward messages ``alpha[t, s]`` for ``len(observations)+1`` rows."""
    pi, joint = _emission_transition_tensors(hmm)
    n = len(pi)
    obs = list(observations)
    alpha = np.zeros((len(obs) + 1, n), dtype=float)
    alpha[0] = pi
    for t, symbol in enumerate(obs):
        matrix = joint.get(symbol)
        if matrix is None:
            alpha[t + 1] = 0.0
        else:
            alpha[t + 1] = alpha[t] @ matrix
    return alpha


def backward(hmm: HiddenMarkovModel, observations: Sequence[Any]) -> np.ndarray:
    """Return backward messages ``beta[t, s]`` for ``len(observations)+1`` rows."""
    _, joint = _emission_transition_tensors(hmm)
    n = next(iter(joint.values())).shape[0] if joint else len(_as_mealy_hmm(hmm).reindex())
    obs = list(observations)
    beta = np.zeros((len(obs) + 1, n), dtype=float)
    beta[len(obs)] = 1.0
    for t in range(len(obs) - 1, -1, -1):
        matrix = joint.get(obs[t])
        if matrix is None:
            beta[t] = 0.0
        else:
            beta[t] = matrix @ beta[t + 1]
    return beta


def log_likelihood(hmm: HiddenMarkovModel, observations: Sequence[Any]) -> float:
    alpha = forward(hmm, observations)
    total = alpha[-1].sum()
    if total <= 0.0:
        return float("-inf")
    return float(np.log(total))


def _log_probabilities(values: np.ndarray) -> np.ndarray:
    log_values = np.full(values.shape, -np.inf, dtype=float)
    positive = values > 0.0
    log_values[positive] = np.log(values[positive])
    return log_values


def viterbi(hmm: HiddenMarkovModel, observations: Sequence[Any]) -> list[Hashable]:
    mealy = _as_mealy_hmm(hmm)
    idx = mealy.reindex()
    pi, joint = _emission_transition_tensors_from_mealy(mealy)
    n = len(idx)
    obs = list(observations)
    if n == 0:
        return []
    if not obs:
        if not np.any(pi > 0.0):
            return []
        return [idx.state(int(np.argmax(pi)))]

    log_pi = _log_probabilities(pi)
    viterbi_log = np.full((len(obs), n), -np.inf, dtype=float)
    backpointer = np.full((len(obs), n), -1, dtype=int)

    matrix0 = joint.get(obs[0])
    if matrix0 is not None:
        log_matrix0 = _log_probabilities(matrix0)
        for j in range(n):
            best = log_pi + log_matrix0[:, j]
            viterbi_log[0, j] = np.max(best)
            backpointer[0, j] = int(np.argmax(best))

    for t in range(1, len(obs)):
        matrix = joint.get(obs[t])
        if matrix is None:
            continue
        log_matrix = _log_probabilities(matrix)
        for j in range(n):
            scores = viterbi_log[t - 1] + log_matrix[:, j]
            viterbi_log[t, j] = np.max(scores)
            backpointer[t, j] = int(np.argmax(scores))

    if not np.any(np.isfinite(viterbi_log[-1])):
        return []

    path = [0] * len(obs)
    path[-1] = int(np.argmax(viterbi_log[-1]))
    for t in range(len(obs) - 2, -1, -1):
        path[t] = backpointer[t + 1, path[t + 1]]
    return [idx.state(i) for i in path]


def sample(
    hmm: HiddenMarkovModel,
    n: int,
    rng: np.random.Generator | None = None,
) -> tuple[list[Any], list[Hashable]]:
    generator = rng if rng is not None else np.random.default_rng()
    mealy = _as_mealy_hmm(hmm)
    idx = mealy.reindex()
    pi, joint = _emission_transition_tensors_from_mealy(mealy)
    state = int(generator.choice(len(idx), p=pi / pi.sum()))

    observations: list[Any] = []
    states: list[Hashable] = []
    for _ in range(n):
        states.append(idx.state(state))
        row_sum = sum(matrix[state].sum() for matrix in joint.values())
        if row_sum <= 0.0:
            break
        symbol_probs = np.array([joint[sym][state].sum() for sym in joint], dtype=float)
        symbol_probs /= symbol_probs.sum()
        symbol_index = int(generator.choice(len(joint), p=symbol_probs))
        symbol = list(joint.keys())[symbol_index]
        observations.append(symbol)
        matrix = joint[symbol]
        row = matrix[state]
        if row.sum() <= 0.0:
            break
        state = int(generator.choice(len(idx), p=row / row.sum()))
    return observations, states
