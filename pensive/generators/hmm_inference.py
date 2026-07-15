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
    from pensive.generators.prob import as_prob, has_symbolic, zeros

    idx = hmm.reindex()
    n = len(idx)
    edge_probs = [transition.data.get(ATTR_PROB, 0.0) for transition in hmm.transitions()]
    init_probs = list(hmm.initial_distribution.values())
    symbolic = has_symbolic(edge_probs) or has_symbolic(init_probs)

    pi = zeros((n,), symbolic=symbolic)
    for state, mass in hmm.initial_distribution.items():
        pi[idx.index(state)] = as_prob(mass)

    symbols: set[Any] = set(hmm.observation_alphabet)
    joint: dict[Any, np.ndarray] = {symbol: zeros((n, n), symbolic=symbolic) for symbol in symbols}

    for transition in hmm.transitions():
        emission = transition.data.get(ATTR_EMISSION)
        if emission is None:
            continue
        i = idx.index(transition.source)
        j = idx.index(transition.target)
        joint[emission][i, j] = as_prob(joint[emission][i, j]) + as_prob(transition.data.get(ATTR_PROB, 0.0))
    return pi, joint


def _emission_transition_tensors(
    hmm: HiddenMarkovModel,
) -> tuple[np.ndarray, dict[Any, np.ndarray]]:
    """Return initial vector ``pi`` and symbol -> joint transition matrices."""
    return _emission_transition_tensors_from_mealy(_as_mealy_hmm(hmm))


def _stationary_emission_tensors(
    hmm: HiddenMarkovModel,
) -> tuple[np.ndarray, dict[Any, np.ndarray]]:
    """Return the stationary state law and symbol -> joint transition matrices.

    Block/word statistics of a *stationary* process must weight the initial state
    by the stationary distribution, not by the model's (possibly transient)
    ``initial_distribution``. The stationary vector is recovered directly from the
    summed emission-transition matrices so it stays aligned with ``joint``'s state
    indexing; it falls back to the initial vector only when no stationary law can be
    found (e.g. a degenerate generator).
    """
    from pensive.generators.prob import zeros
    from pensive.generators.stationary import stationary_distribution_from_transition

    pi_initial, joint = _emission_transition_tensors(hmm)
    n = len(pi_initial)
    if n == 0:
        return pi_initial, joint
    symbolic = pi_initial.dtype == object or any(matrix.dtype == object for matrix in joint.values())
    transition = zeros((n, n), symbolic=symbolic)
    for matrix in joint.values():
        transition = transition + matrix
    try:
        pi = stationary_distribution_from_transition(transition)
    except Exception:
        pi = pi_initial
    return pi, joint


def _forward_scaled(
    pi: np.ndarray,
    joint: dict[Any, np.ndarray],
    obs: list[Any],
) -> tuple[np.ndarray, np.ndarray]:
    """Return per-step-normalized forward messages and log scaling factors.

    ``alpha_hat[t]`` sums to one; ``log P(obs) = log_scales.sum()``. A ``-inf``
    entry in ``log_scales`` marks an impossible step. Normalizing each step avoids
    the underflow that makes the raw forward product vanish for long sequences.
    """
    n = len(pi)
    alpha_hat = np.zeros((len(obs) + 1, n), dtype=float)
    log_scales = np.zeros(len(obs) + 1, dtype=float)
    total0 = float(pi.sum())
    if total0 <= 0.0:
        log_scales[0] = -np.inf
        return alpha_hat, log_scales
    alpha_hat[0] = pi / total0
    log_scales[0] = float(np.log(total0))
    for t, symbol in enumerate(obs):
        matrix = joint.get(symbol)
        if matrix is None:
            log_scales[t + 1] = -np.inf
            continue
        row = alpha_hat[t] @ matrix
        scale = float(row.sum())
        if scale <= 0.0:
            log_scales[t + 1] = -np.inf
            continue
        alpha_hat[t + 1] = row / scale
        log_scales[t + 1] = float(np.log(scale))
    return alpha_hat, log_scales


def forward(hmm: HiddenMarkovModel, observations: Sequence[Any], *, scaled: bool = False) -> np.ndarray:
    """Return forward messages ``alpha[t, s]`` for ``len(observations)+1`` rows.

    With ``scaled=True`` each row is normalized to sum to one (the numerically
    stable message used for posteriors); otherwise the raw messages are returned.
    """
    pi, joint = _emission_transition_tensors(hmm)
    obs = list(observations)
    if scaled:
        alpha_hat, _log_scales = _forward_scaled(pi, joint, obs)
        return alpha_hat
    n = len(pi)
    alpha = np.zeros((len(obs) + 1, n), dtype=float)
    alpha[0] = pi
    for t, symbol in enumerate(obs):
        matrix = joint.get(symbol)
        if matrix is None:
            alpha[t + 1] = 0.0
        else:
            alpha[t + 1] = alpha[t] @ matrix
    return alpha


def backward(hmm: HiddenMarkovModel, observations: Sequence[Any], *, scaled: bool = False) -> np.ndarray:
    """Return backward messages ``beta[t, s]`` for ``len(observations)+1`` rows.

    With ``scaled=True`` each row is normalized to sum to one. The smoothed
    posterior is then ``normalize(alpha_hat[t] * beta_hat[t])`` (the per-row
    scaling constants cancel on renormalization).
    """
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
        if scaled:
            total = float(beta[t].sum())
            if total > 0.0:
                beta[t] = beta[t] / total
    return beta


def log_likelihood(hmm: HiddenMarkovModel, observations: Sequence[Any]) -> float:
    """Natural-log likelihood ``log P(observations)``.

    Uses the per-step-scaled forward recursion so the result stays finite for long
    sequences instead of underflowing to ``-inf``.
    """
    pi, joint = _emission_transition_tensors(hmm)
    _alpha_hat, log_scales = _forward_scaled(pi, joint, list(observations))
    if not np.all(np.isfinite(log_scales)):
        return float("-inf")
    return float(log_scales.sum())


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
