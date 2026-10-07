"""Forward/backward filtering, smoothing, and Viterbi decoding for hidden Markov models.

Fixed-interval smoothing (one- and two-slice marginals) follows Cappe, Moulines &
Ryden (2005, Section 3.2).
"""

from __future__ import annotations

from collections.abc import Hashable, Sequence
from typing import Any

import numpy as np

from sofic.generators.base import HiddenMarkovModel
from sofic.generators.matrices import emission_tensors, symbol_matrices


def _as_mealy_hmm(hmm: HiddenMarkovModel) -> Any:
    """Return a Mealy-style representation through the HMM representation hook."""
    return hmm.to_mealy()


def _forward_scaled(
    pi: np.ndarray,
    joint: dict[Any, np.ndarray],
    obs: list[Any],
) -> tuple[np.ndarray, np.ndarray]:
    """Return per-step-normalized forward messages and log scaling factors.

    ``alpha_hat[t]`` sums to one; ``log2 P(obs) = log_scales.sum()``. A ``-inf``
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
    log_scales[0] = float(np.log2(total0))
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
        log_scales[t + 1] = float(np.log2(scale))
    return alpha_hat, log_scales


def forward(hmm: HiddenMarkovModel, observations: Sequence[Any], *, normalize: bool = False) -> np.ndarray:
    """Return forward messages ``alpha[t, s]`` for ``len(observations)+1`` rows.

    With ``normalize=True`` each row is normalized to sum to one (the numerically
    stable message used for posteriors); otherwise the raw messages are returned.
    """
    pi, joint = emission_tensors(hmm)
    obs = list(observations)
    if normalize:
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


def backward(hmm: HiddenMarkovModel, observations: Sequence[Any], *, normalize: bool = False) -> np.ndarray:
    """Return backward messages ``beta[t, s]`` for ``len(observations)+1`` rows.

    With ``normalize=True`` each row is normalized to sum to one. The smoothed
    posterior is then ``normalize(alpha_hat[t] * beta_hat[t])`` (the per-row
    scaling constants cancel on renormalization).
    """
    joint = symbol_matrices(_as_mealy_hmm(hmm))
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
        if normalize:
            total = float(beta[t].sum())
            if total > 0.0:
                beta[t] = beta[t] / total
    return beta


def _backward_scaled(joint: dict[Any, np.ndarray], obs: list[Any], n_states: int) -> np.ndarray:
    """Per-row-normalized backward messages from precomputed transition tensors.

    ``beta_hat[t]`` sums to one; the per-row scaling constants cancel against the
    forward scaling when the smoothed posterior is renormalized. Shares tensors
    with the forward pass so smoothing and EM avoid recomputing them.
    """
    beta = np.zeros((len(obs) + 1, n_states), dtype=float)
    beta[len(obs)] = 1.0
    for t in range(len(obs) - 1, -1, -1):
        matrix = joint.get(obs[t])
        row = beta[t + 1] if matrix is None else matrix @ beta[t + 1]
        beta[t] = 0.0 if matrix is None else row
        total = float(beta[t].sum())
        if total > 0.0:
            beta[t] = beta[t] / total
    return beta


def log_likelihood(hmm: HiddenMarkovModel, observations: Sequence[Any]) -> float:
    """Log-likelihood ``log2 P(observations)`` in bits.

    Uses the per-step-scaled forward recursion so the result stays finite for long
    sequences instead of underflowing to ``-inf``.
    """
    pi, joint = emission_tensors(hmm)
    _alpha_hat, log_scales = _forward_scaled(pi, joint, list(observations))
    if not np.all(np.isfinite(log_scales)):
        return float("-inf")
    return float(log_scales.sum())


def smooth(hmm: HiddenMarkovModel, observations: Sequence[Any]) -> np.ndarray:
    r"""Return fixed-interval smoothed marginals ``gamma[t, s]``.

    ``gamma[t, s] = P(X_t = s \mid Y_{0:n-1})`` for ``t = 0, ..., n`` (there are
    ``n + 1`` hidden states behind ``n`` edge emissions). Computed as the
    per-row-renormalized product of the scaled forward and backward messages, the
    forward-backward smoother of Cappe, Moulines & Ryden (2005, Section 3.2).
    Rows for observation sequences of zero probability are returned as zeros.
    """
    pi, joint = emission_tensors(hmm)
    obs = list(observations)
    n_states = len(pi)
    alpha_hat, log_scales = _forward_scaled(pi, joint, obs)
    if not np.all(np.isfinite(log_scales)):
        return np.zeros((len(obs) + 1, n_states), dtype=float)
    beta_hat = _backward_scaled(joint, obs, n_states)
    gamma = alpha_hat * beta_hat
    row_sums = gamma.sum(axis=1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        gamma = np.where(row_sums > 0.0, gamma / row_sums, 0.0)
    return gamma


def two_slice_marginals(hmm: HiddenMarkovModel, observations: Sequence[Any]) -> np.ndarray:
    r"""Return two-slice smoothed marginals ``xi[t, i, j]``.

    ``xi[t, i, j] = P(X_t = i, X_{t+1} = j \mid Y_{0:n-1})`` for ``t = 0, ..., n-1``,
    where the transition at index ``t`` emits ``Y_t`` (Cappe, Moulines & Ryden,
    2005, Section 3.2). Marginalizing over ``j`` recovers ``gamma[t]`` for
    ``t < n``. Returns an all-zero tensor for zero-probability sequences.
    """
    pi, joint = emission_tensors(hmm)
    obs = list(observations)
    n_states = len(pi)
    xi = np.zeros((len(obs), n_states, n_states), dtype=float)
    alpha_hat, log_scales = _forward_scaled(pi, joint, obs)
    if not np.all(np.isfinite(log_scales)):
        return xi
    beta_hat = _backward_scaled(joint, obs, n_states)
    for t, symbol in enumerate(obs):
        matrix = joint.get(symbol)
        if matrix is None:
            continue
        block = alpha_hat[t][:, None] * matrix * beta_hat[t + 1][None, :]
        total = float(block.sum())
        if total > 0.0:
            xi[t] = block / total
    return xi


def _log_probabilities(values: np.ndarray) -> np.ndarray:
    log_values = np.full(values.shape, -np.inf, dtype=float)
    positive = values > 0.0
    log_values[positive] = np.log(values[positive])
    return log_values


def viterbi(hmm: HiddenMarkovModel, observations: Sequence[Any]) -> list[Hashable]:
    mealy = _as_mealy_hmm(hmm)
    idx = mealy.reindex()
    pi, joint = emission_tensors(mealy)
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
