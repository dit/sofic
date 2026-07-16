"""Bayesian nonparametric HMM inference: the (sticky) HDP-HMM.

The hierarchical Dirichlet process HMM :cite:`Teh2006` and its sticky variant
:cite:`Fox2011` place a nonparametric prior over the number of hidden states, so
the state count is *inferred* rather than fixed. This module implements the
**weak-limit blocked Gibbs sampler** :cite:`Fox2011`: the countably-infinite HDP
prior is approximated by a symmetric Dirichlet over ``max_states`` components,
and each sweep

1. samples the whole hidden-state sequence with forward-filter/backward-sample
   (FFBS) given the current parameters,
2. draws conjugate Dirichlet transition, initial, and (categorical /
   Dirichlet-multinomial) emission rows given the state sequence, and
3. updates the shared top-level weights ``beta`` from Antoniak table counts,
   with the sticky self-transition override of :cite:`Fox2011`.

Emissions are **categorical (discrete)** throughout -- the Gaussian-emission
variant is deliberately excluded. Retained posterior samples are returned as
:class:`~sofic.generators.moore.MooreHMM` generators restricted to the states
occupied in that sweep, together with a posterior over the number of occupied
states.
"""

from __future__ import annotations

from collections.abc import Hashable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from sofic.generators.moore import MooreHMM
from sofic.inference.bayesian.counts import BayesianInferenceError

__all__ = ["HDPHMMPosterior", "infer_hdp_hmm"]


@dataclass
class HDPHMMPosterior:
    """Posterior samples from the weak-limit (sticky) HDP-HMM sampler.

    Attributes
    ----------
    samples
        Retained posterior draws as :class:`~sofic.generators.moore.MooreHMM`
        generators, each restricted to the states occupied in its sweep.
    state_counts
        Number of occupied states in each retained draw (``len == len(samples)``).
    log_likelihoods
        Data log-likelihood (natural log) of each retained draw.
    alphabet
        Sorted observation alphabet used by the sampler.
    """

    samples: list[MooreHMM] = field(default_factory=list)
    state_counts: list[int] = field(default_factory=list)
    log_likelihoods: list[float] = field(default_factory=list)
    alphabet: tuple[Any, ...] = ()

    def state_count_posterior(self) -> dict[int, float]:
        """Return the posterior distribution over the number of occupied states."""
        if not self.state_counts:
            return {}
        counts = np.asarray(self.state_counts)
        values, freqs = np.unique(counts, return_counts=True)
        total = float(freqs.sum())
        return {int(value): float(freq) / total for value, freq in zip(values, freqs, strict=True)}

    def map_state_count(self) -> int:
        """Return the posterior modal number of occupied states."""
        posterior = self.state_count_posterior()
        if not posterior:
            raise BayesianInferenceError("no retained samples")
        return max(posterior, key=posterior.get)

    def best_sample(self) -> MooreHMM:
        """Return the retained draw with the highest data log-likelihood."""
        if not self.samples:
            raise BayesianInferenceError("no retained samples")
        return self.samples[int(np.argmax(self.log_likelihoods))]


def _normalize_sequences(sequences: Sequence[Any]) -> list[list[Any]]:
    if sequences is None:
        raise BayesianInferenceError("sequences is required")
    if isinstance(sequences, (str, bytes)):
        raise BayesianInferenceError("pass a sequence of observations, not a string")
    first = next(iter(sequences), None)
    if first is None:
        raise BayesianInferenceError("at least one non-empty sequence is required")
    if not isinstance(first, (list, tuple)):
        sequences = [sequences]  # a single flat observation sequence
    out = [list(seq) for seq in sequences if len(seq) > 0]
    if not out:
        raise BayesianInferenceError("at least one non-empty sequence is required")
    return out


def _collect_alphabet(sequences: Sequence[Sequence[Any]]) -> tuple[Any, ...]:
    symbols: set[Any] = set()
    for seq in sequences:
        symbols.update(seq)
    return tuple(sorted(symbols, key=repr))


def _ffbs(
    obs_idx: np.ndarray,
    log_pi0: np.ndarray,
    trans: np.ndarray,
    emit: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, float]:
    """Forward-filter backward-sample one sequence; return states and log-likelihood."""
    n_states = trans.shape[0]
    length = obs_idx.shape[0]
    alpha = np.empty((length, n_states))
    loglik = 0.0

    weights = np.exp(log_pi0) * emit[:, obs_idx[0]]
    scale = weights.sum()
    if scale <= 0.0:
        weights = emit[:, obs_idx[0]].copy()
        scale = weights.sum()
    alpha[0] = weights / scale
    loglik += np.log(scale)

    for t in range(1, length):
        predicted = alpha[t - 1] @ trans
        weights = predicted * emit[:, obs_idx[t]]
        scale = weights.sum()
        if scale <= 0.0:
            weights = emit[:, obs_idx[t]].copy()
            scale = weights.sum()
        alpha[t] = weights / scale
        loglik += np.log(scale)

    states = np.empty(length, dtype=int)
    states[length - 1] = rng.choice(n_states, p=alpha[length - 1])
    for t in range(length - 2, -1, -1):
        probs = alpha[t] * trans[:, states[t + 1]]
        total = probs.sum()
        probs = probs / total if total > 0 else np.full(n_states, 1.0 / n_states)
        states[t] = rng.choice(n_states, p=probs)
    return states, float(loglik)


def _sample_dirichlet_rows(alpha_rows: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Draw one Dirichlet vector per row (rows with zero mass fall back to uniform)."""
    out = np.empty_like(alpha_rows, dtype=float)
    for i, row in enumerate(alpha_rows):
        if row.sum() <= 0:
            out[i] = np.full(row.shape[0], 1.0 / row.shape[0])
        else:
            out[i] = rng.dirichlet(row)
    return out


def _antoniak_tables(customers: int, mass: float, rng: np.random.Generator) -> int:
    """Sample the number of occupied CRP tables for ``customers`` at concentration ``mass``."""
    if customers <= 0 or mass <= 0.0:
        return 0
    indices = np.arange(customers)
    probs = mass / (mass + indices)
    return int((rng.random(customers) < probs).sum())


def infer_hdp_hmm(
    sequences: Sequence[Any],
    *,
    max_states: int = 20,
    alpha: float = 1.0,
    gamma: float = 1.0,
    kappa: float = 0.0,
    eta: float = 1.0,
    iterations: int = 300,
    burn_in: int = 150,
    thin: int = 5,
    rng: np.random.Generator | int | None = None,
) -> HDPHMMPosterior:
    """Infer a (sticky) HDP-HMM from discrete sequences by weak-limit blocked Gibbs.

    Parameters
    ----------
    sequences
        Either a single observation sequence (list/tuple of symbols) or a
        collection of such sequences.
    max_states
        Weak-limit truncation ``L`` -- an upper bound on the number of states.
        The *occupied* count is inferred and is typically far smaller.
    alpha
        Second-level (per-state transition) DP concentration.
    gamma
        Top-level DP concentration governing the shared weights ``beta``.
    kappa
        Sticky self-transition mass :cite:`Fox2011`. ``0`` recovers the plain
        HDP-HMM; positive values bias toward state persistence.
    eta
        Symmetric Dirichlet concentration of the categorical emission prior.
    iterations, burn_in, thin
        Total Gibbs sweeps, discarded warm-up sweeps, and retention stride.
    rng
        ``numpy`` generator or seed.

    Returns
    -------
    HDPHMMPosterior
        Retained :class:`~sofic.generators.moore.MooreHMM` draws and a posterior
        over the number of occupied states.
    """
    generator = rng if isinstance(rng, np.random.Generator) else np.random.default_rng(rng)

    seqs = _normalize_sequences(sequences)
    alphabet = _collect_alphabet(seqs)
    if not alphabet:
        raise BayesianInferenceError("sequences contain no observations")
    if max_states < 1:
        raise BayesianInferenceError("max_states must be >= 1")
    for name, value in (("alpha", alpha), ("gamma", gamma), ("eta", eta)):
        if value <= 0.0:
            raise BayesianInferenceError(f"{name} must be positive")
    if kappa < 0.0:
        raise BayesianInferenceError("kappa must be non-negative")
    if burn_in >= iterations:
        raise BayesianInferenceError("burn_in must be smaller than iterations")

    symbol_index = {symbol: i for i, symbol in enumerate(alphabet)}
    obs = [np.array([symbol_index[o] for o in seq], dtype=int) for seq in seqs]
    n_obs = len(alphabet)
    length = max_states

    # Warm start: label each observation by its symbol so state k initially emits
    # symbol k. Random (near-uniform) initialization mixes very poorly for
    # discrete emissions -- the symmetric prior leaves the state labels
    # unidentified for many sweeps -- whereas a symbol-based labeling immediately
    # breaks that symmetry with near-deterministic emissions.
    beta = np.full(length, 1.0 / length)
    rho = kappa / (alpha + kappa) if (alpha + kappa) > 0 else 0.0
    init_trans = np.zeros((length, length))
    init_emit = np.zeros((length, n_obs))
    init_first = np.zeros(length)
    for obs_idx in obs:
        warm = obs_idx % length
        init_first[warm[0]] += 1
        for o_i, s in zip(obs_idx, warm, strict=True):
            init_emit[s, o_i] += 1
        for a, b in zip(warm[:-1], warm[1:], strict=True):
            init_trans[a, b] += 1
    trans = _sample_dirichlet_rows(alpha * beta[None, :] + kappa * np.eye(length) + init_trans, generator)
    emit = _sample_dirichlet_rows(eta + init_emit, generator)
    log_pi0 = np.log(np.clip(_sample_dirichlet_rows((alpha * beta + init_first)[None, :], generator)[0], 1e-300, None))

    posterior = HDPHMMPosterior(alphabet=alphabet)

    for sweep in range(iterations):
        trans_counts = np.zeros((length, length))
        init_counts = np.zeros(length)
        emit_counts = np.zeros((length, n_obs))
        sweep_states: list[np.ndarray] = []
        total_loglik = 0.0

        for obs_idx in obs:
            states, loglik = _ffbs(obs_idx, log_pi0, trans, emit, generator)
            sweep_states.append(states)
            total_loglik += loglik
            init_counts[states[0]] += 1
            for o_i, s in zip(obs_idx, states, strict=True):
                emit_counts[s, o_i] += 1
            for a, b in zip(states[:-1], states[1:], strict=True):
                trans_counts[a, b] += 1

        sticky = kappa * np.eye(length)
        trans = _sample_dirichlet_rows(alpha * beta[None, :] + sticky + trans_counts, generator)
        emit = _sample_dirichlet_rows(eta + emit_counts, generator)
        pi0 = _sample_dirichlet_rows((alpha * beta + init_counts)[None, :], generator)[0]
        log_pi0 = np.log(np.clip(pi0, 1e-300, None))

        # Antoniak table counts -> top-level weights beta (with sticky override).
        bar_m = np.zeros(length)
        rows = list(trans_counts) + [init_counts]
        for j, row in enumerate(rows):
            for k in range(length):
                mass = alpha * beta[k] + (kappa if j == k else 0.0)
                tables = _antoniak_tables(int(row[k]), mass, generator)
                if j == k and kappa > 0.0 and tables > 0:
                    denom = rho + beta[k] * (1.0 - rho)
                    override_p = rho / denom if denom > 0 else 0.0
                    overrides = int(generator.binomial(tables, min(max(override_p, 0.0), 1.0)))
                    tables -= overrides
                bar_m[k] += tables
        beta = generator.dirichlet(gamma / length + bar_m)

        if sweep >= burn_in and (sweep - burn_in) % thin == 0:
            machine = _build_moore(sweep_states, trans, emit, pi0, alphabet)
            posterior.samples.append(machine)
            posterior.state_counts.append(len(list(machine.states())))
            posterior.log_likelihoods.append(total_loglik)

    if not posterior.samples:
        raise BayesianInferenceError("no samples retained; increase iterations or lower burn_in/thin")
    return posterior


def _build_moore(
    sweep_states: Sequence[np.ndarray],
    trans: np.ndarray,
    emit: np.ndarray,
    pi0: np.ndarray,
    alphabet: Sequence[Any],
) -> MooreHMM:
    """Restrict the sampled parameters to occupied states and build a MooreHMM."""
    occupied = sorted({int(s) for states in sweep_states for s in states})
    index = {state: position for position, state in enumerate(occupied)}
    names: dict[int, Hashable] = {state: f"q{position}" for position, state in enumerate(occupied)}

    sub_trans = trans[np.ix_(occupied, occupied)]
    row_sums = sub_trans.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    sub_trans = sub_trans / row_sums

    init = pi0[occupied]
    init_total = init.sum()
    init = init / init_total if init_total > 0 else np.full(len(occupied), 1.0 / len(occupied))

    machine = MooreHMM(
        observation_alphabet=frozenset(alphabet),
        initial_distribution={names[state]: float(init[index[state]]) for state in occupied},
    )
    for state in occupied:
        machine.graph.add_state(names[state])
    for state in occupied:
        machine.set_emission_distribution(
            names[state],
            {symbol: float(emit[state, j]) for j, symbol in enumerate(alphabet)},
        )
        for target in occupied:
            prob = float(sub_trans[index[state], index[target]])
            if prob > 0.0:
                machine.add_transition(names[state], names[target], prob)
    machine.validate()
    return machine
