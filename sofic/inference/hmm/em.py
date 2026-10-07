"""Baum-Welch (EM) parameter re-estimation for hidden Markov models.

Follows Baum, Petrie, Soules & Weiss and Cappe, Moulines & Ryden (2005, Chapter 10).
"""

from __future__ import annotations

import warnings
from collections import defaultdict
from collections.abc import Iterable
from typing import Any

import numpy as np

from sofic.generators.base import HiddenMarkovModel
from sofic.generators.matrices import emission_tensors
from sofic.inference.hmm.filtering import _backward_scaled, _forward_scaled


def _expected_edge_counts(
    pi: np.ndarray,
    joint: dict[Any, np.ndarray],
    obs: list[Any],
) -> tuple[dict[tuple[int, Any, int], float], np.ndarray, np.ndarray, float]:
    r"""Expected sufficient statistics for one observation sequence.

    Returns ``(edge_counts, source_totals, gamma0, loglik)`` where

    - ``edge_counts[(i, symbol, j)]`` is
      :math:`\sum_t P(X_t = i, Y_t = symbol, X_{t+1} = j \mid Y)`, the expected
      number of uses of edge ``i --symbol--> j``;
    - ``source_totals[i] = \sum_{t=0}^{n-1} P(X_t = i \mid Y)`` is the expected
      number of transitions out of state ``i`` (the Baum-Welch denominator);
    - ``gamma0`` is the smoothed marginal of the initial state ``X_0``;
    - ``loglik`` is the log-likelihood of the sequence in bits.

    Only edges present in ``joint`` (structural support) receive mass, so the
    statistics preserve the model topology.
    """
    n_states = len(pi)
    alpha_hat, log_scales = _forward_scaled(pi, joint, obs)
    if not np.all(np.isfinite(log_scales)):
        return {}, np.zeros(n_states), np.zeros(n_states), float("-inf")
    beta_hat = _backward_scaled(joint, obs, n_states)
    edge_counts: dict[tuple[int, Any, int], float] = {}
    source_totals = np.zeros(n_states, dtype=float)
    for t, symbol in enumerate(obs):
        matrix = joint.get(symbol)
        if matrix is None:
            continue
        block = alpha_hat[t][:, None] * matrix * beta_hat[t + 1][None, :]
        total = float(block.sum())
        if total <= 0.0:
            continue
        block = block / total
        source_totals += block.sum(axis=1)
        for i, j in np.argwhere(block > 0.0):
            key = (int(i), symbol, int(j))
            edge_counts[key] = edge_counts.get(key, 0.0) + float(block[i, j])
    g0 = alpha_hat[0] * beta_hat[0]
    s0 = float(g0.sum())
    gamma0 = g0 / s0 if s0 > 0.0 else np.zeros(n_states)
    return edge_counts, source_totals, gamma0, float(log_scales.sum())


def _as_sequence_list(sequences: Iterable[Any]) -> list[list[Any]]:
    """Normalize ``sequences`` to a list of observation sequences.

    Accepts either a single flat observation sequence (e.g. ``[0, 1, 0]``) or an
    iterable of sequences (e.g. ``[[0, 1], [1, 0]]``). A single sequence is
    detected when its first element is not itself a non-string sequence.
    """
    seqs = list(sequences)
    if not seqs:
        return []
    first = seqs[0]
    if isinstance(first, (list, tuple)) and not isinstance(first, (str, bytes)):
        return [list(seq) for seq in seqs]
    return [seqs]


def baum_welch(
    hmm: HiddenMarkovModel,
    sequences: Iterable[Any],
    *,
    max_iter: int = 100,
    tol: float = 1e-6,
    estimate_initial: bool = True,
    n_restarts: int = 1,
    rng: np.random.Generator | int | None = None,
    return_restarts: bool = False,
) -> tuple[Any, list[float]] | tuple[Any, list[float], list[float]]:
    r"""Fit HMM parameters by Baum-Welch (EM) expectation-maximization.

    Re-estimates the Mealy joint edge law
    :math:`A_o[i, j] = P(X_{t+1} = j, O = o \mid X_t = i)` and (optionally) the
    initial distribution from data, holding the transition-graph topology fixed:
    structurally absent edges receive zero expected count and stay absent, so the
    fitted model generates the same sofic shift as ``hmm``. This is the EM
    algorithm for probabilistic functions of finite Markov chains of Baum, Petrie,
    Soules & Weiss and Cappe, Moulines & Ryden (2005, Chapter 10); see also
    Rabiner (1989).

    ``sequences`` may be a single observation sequence or an iterable of
    sequences (several sequences are needed to identify the initial distribution;
    Cappe, Moulines & Ryden, 2005, Section 10.1). Unifilarity is *not* preserved,
    so the fit is returned as a plain :class:`~sofic.generators.mealy.MealyHMM`.

    Returns ``(fitted_model, loglik_trace)`` where ``loglik_trace`` is the
    non-decreasing sequence of total log-likelihoods (bits) observed before each
    parameter update.

    EM converges to a local maximum of the likelihood. With ``n_restarts > 1`` the
    first run starts from ``hmm``'s parameters and each further run from edge laws
    drawn uniformly (Dirichlet(1)) over each state's structurally allowed edges;
    the fit with the highest final log-likelihood is returned. Pass
    ``return_restarts=True`` to also get every run's final log-likelihood, which
    shows whether near-equal optima exist.
    """
    from sofic.generators.mealy import MealyHMM

    mealy = hmm.to_mealy()
    idx = mealy.reindex()
    n_states = len(idx)
    states = [idx.state(i) for i in range(n_states)]
    alphabet = frozenset(mealy.observation_alphabet)
    seqs = _as_sequence_list(sequences)

    pi, joint = emission_tensors(mealy)
    support = {
        (i, symbol, j)
        for symbol, matrix in joint.items()
        for i in range(n_states)
        for j in range(n_states)
        if matrix[i, j] > 0.0
    }

    if n_restarts < 1:
        raise ValueError("n_restarts must be at least 1")
    generator = rng if isinstance(rng, np.random.Generator) else np.random.default_rng(rng)
    runs = []
    for restart in range(n_restarts):
        start_joint = joint if restart == 0 else _random_edge_law(joint, support, n_states, generator)
        runs.append(
            _baum_welch_run(pi, start_joint, seqs, max_iter=max_iter, tol=tol, estimate_initial=estimate_initial)
        )
    finals = [trace[-1] if trace else float("-inf") for _pi, _joint, trace in runs]
    pi, joint, loglik_trace = runs[int(np.argmax(finals))]

    fitted = MealyHMM(
        initial_distribution={states[i]: float(pi[i]) for i in range(n_states) if pi[i] > 0.0},
        observation_alphabet=alphabet,
    )
    for state in states:
        fitted.graph.add_state(state)
    for i, symbol, j in sorted(support, key=lambda edge: (edge[0], str(edge[1]), edge[2])):
        prob = float(joint[symbol][i, j])
        if prob > 0.0:
            fitted.add_transition(states[i], states[j], symbol, prob)
    fitted.validate()
    if return_restarts:
        return fitted, loglik_trace, finals
    return fitted, loglik_trace


def _random_edge_law(
    joint: dict[Any, np.ndarray],
    support: set[tuple[int, Any, int]],
    n_states: int,
    rng: np.random.Generator,
) -> dict[Any, np.ndarray]:
    """Edge laws drawn uniformly over each state's allowed ``(symbol, target)`` edges."""
    new_joint = {symbol: np.zeros((n_states, n_states), dtype=float) for symbol in joint}
    for i in range(n_states):
        edges = sorted(((symbol, j) for (source, symbol, j) in support if source == i), key=lambda e: (str(e[0]), e[1]))
        if not edges:
            continue
        weights = rng.dirichlet(np.ones(len(edges)))
        for (symbol, j), weight in zip(edges, weights, strict=True):
            new_joint[symbol][i, j] = weight
    return new_joint


def _baum_welch_run(
    pi: np.ndarray,
    joint: dict[Any, np.ndarray],
    seqs: list[Any],
    *,
    max_iter: int,
    tol: float,
    estimate_initial: bool,
) -> tuple[np.ndarray, dict[Any, np.ndarray], list[float]]:
    """One EM run from ``(pi, joint)``; returns the final parameters and trace."""
    n_states = len(pi)
    loglik_trace: list[float] = []
    prev_ll: float | None = None
    for _iteration in range(max_iter):
        total_edge_counts: dict[tuple[int, Any, int], float] = defaultdict(float)
        total_source = np.zeros(n_states, dtype=float)
        gamma0_sum = np.zeros(n_states, dtype=float)
        total_ll = 0.0
        skipped = 0
        for obs in seqs:
            edge_counts, source_totals, gamma0, loglik = _expected_edge_counts(pi, joint, obs)
            if not np.isfinite(loglik):
                skipped += 1
                continue
            for key, value in edge_counts.items():
                total_edge_counts[key] += value
            total_source += source_totals
            gamma0_sum += gamma0
            total_ll += loglik
        if seqs and skipped == len(seqs):
            raise ValueError("every observation sequence has zero probability under the model")
        if skipped and _iteration == 0:
            warnings.warn(
                f"{skipped} of {len(seqs)} sequences have zero probability under the model and are ignored",
                RuntimeWarning,
                stacklevel=3,
            )
        loglik_trace.append(total_ll)
        if prev_ll is not None and abs(total_ll - prev_ll) < tol:
            break
        prev_ll = total_ll

        new_joint = {symbol: np.zeros((n_states, n_states), dtype=float) for symbol in joint}
        for (i, symbol, j), count in total_edge_counts.items():
            if total_source[i] > 0.0:
                new_joint[symbol][i, j] = count / total_source[i]
        for i in range(n_states):
            if total_source[i] <= 0.0:
                for symbol in joint:
                    new_joint[symbol][i, :] = joint[symbol][i, :]
        joint = new_joint
        if estimate_initial:
            mass = float(gamma0_sum.sum())
            if mass > 0.0:
                pi = gamma0_sum / mass
    return pi, joint, loglik_trace
