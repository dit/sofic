"""Inference for hidden Markov models.

Forward/backward/Viterbi decoding and sampling, plus the Cappe, Moulines &
Ryden (2005) toolbox: fixed-interval smoothing (one- and two-slice marginals),
Baum-Welch EM parameter re-estimation, and the score / observed information via
the Fisher and Louis identities.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Hashable, Iterable, Sequence
from typing import Any

import numpy as np

from sofic.generators.base import HiddenMarkovModel
from sofic.graph import ATTR_EMISSION, ATTR_PROB


def _as_mealy_hmm(hmm: HiddenMarkovModel) -> Any:
    """Return a Mealy-style representation through the HMM representation hook."""
    return hmm.to_mealy()


def _emission_transition_tensors_from_mealy(
    hmm: Any,
) -> tuple[np.ndarray, dict[Any, np.ndarray]]:
    """Return initial vector ``pi`` and symbol -> joint transition matrices."""
    from sofic.generators.prob import as_prob, has_symbolic, zeros

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


def _limit_distribution_from_initial(pi_initial: np.ndarray, transition: np.ndarray) -> np.ndarray | None:
    """Return the limiting occupation law of ``pi_initial`` under ``transition``.

    On reducible chains the left-eigenvector stationary law is not unique; the
    process measure is the limit reached from the model's initial distribution.
    """
    pi = np.asarray(pi_initial, dtype=float).copy()
    total = float(pi.sum())
    if total <= 0.0:
        return None
    pi /= total
    matrix = np.asarray(transition, dtype=float)
    n = len(pi)
    for _ in range(max(100, 20 * n)):
        nxt = pi @ matrix
        mass = float(nxt.sum())
        if mass <= 0.0:
            return None
        nxt /= mass
        if np.allclose(nxt, pi, rtol=1e-12, atol=1e-14):
            pi = nxt
            break
        pi = nxt
    pi[np.isclose(pi, 0.0, atol=1e-15)] = 0.0
    mass = float(pi.sum())
    if mass <= 0.0:
        return None
    return pi / mass


def _stationary_emission_tensors(
    hmm: HiddenMarkovModel,
) -> tuple[np.ndarray, dict[Any, np.ndarray]]:
    """Return the stationary state law and symbol -> joint transition matrices.

    Block/word statistics of a *stationary* process must weight the initial state
    by the stationary distribution, not by the model's (possibly transient)
    ``initial_distribution``. The stationary vector is recovered directly from the
    summed emission-transition matrices so it stays aligned with ``joint``'s state
    indexing.

    When the chain is reducible (multiple absorbing classes), the eigenvector
    stationary law is not unique — prefer the limiting occupation reached from
    ``initial_distribution``. Fall back to the eigenvector solution, then to the
    initial vector, only when the limit cannot be formed.
    """
    from sofic.generators.prob import zeros
    from sofic.generators.stationary import stationary_distribution_from_transition

    pi_initial, joint = _emission_transition_tensors(hmm)
    n = len(pi_initial)
    if n == 0:
        return pi_initial, joint
    symbolic = pi_initial.dtype == object or any(matrix.dtype == object for matrix in joint.values())
    transition = zeros((n, n), symbolic=symbolic)
    for matrix in joint.values():
        transition = transition + matrix
    if not symbolic:
        limited = _limit_distribution_from_initial(pi_initial, transition)
        if limited is not None and np.allclose(limited @ transition, limited, rtol=1e-8, atol=1e-10):
            return limited, joint
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
    """Natural-log likelihood ``log P(observations)``.

    Uses the per-step-scaled forward recursion so the result stays finite for long
    sequences instead of underflowing to ``-inf``.
    """
    pi, joint = _emission_transition_tensors(hmm)
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
    pi, joint = _emission_transition_tensors(hmm)
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
    pi, joint = _emission_transition_tensors(hmm)
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
    - ``loglik`` is the natural-log likelihood of the sequence.

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
) -> tuple[Any, list[float]]:
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
    non-decreasing sequence of total natural-log likelihoods observed before each
    parameter update.
    """
    from sofic.generators.mealy import MealyHMM

    mealy = hmm.to_mealy()
    idx = mealy.reindex()
    n_states = len(idx)
    states = [idx.state(i) for i in range(n_states)]
    alphabet = frozenset(mealy.observation_alphabet)
    seqs = _as_sequence_list(sequences)

    pi, joint = _emission_transition_tensors_from_mealy(mealy)
    support = {
        (i, symbol, j)
        for symbol, matrix in joint.items()
        for i in range(n_states)
        for j in range(n_states)
        if matrix[i, j] > 0.0
    }

    loglik_trace: list[float] = []
    prev_ll: float | None = None
    for _iteration in range(max_iter):
        total_edge_counts: dict[tuple[int, Any, int], float] = defaultdict(float)
        total_source = np.zeros(n_states, dtype=float)
        gamma0_sum = np.zeros(n_states, dtype=float)
        total_ll = 0.0
        for obs in seqs:
            edge_counts, source_totals, gamma0, loglik = _expected_edge_counts(pi, joint, obs)
            if not np.isfinite(loglik):
                continue
            for key, value in edge_counts.items():
                total_edge_counts[key] += value
            total_source += source_totals
            gamma0_sum += gamma0
            total_ll += loglik
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
    return fitted, loglik_trace


def score(hmm: HiddenMarkovModel, observations: Sequence[Any]) -> dict[tuple[Hashable, Any, Hashable], float]:
    r"""Return the score (gradient of the log-likelihood) via the Fisher identity.

    For each edge ``i --o--> j``, returns
    :math:`\partial \log P(Y) / \partial A_o[i, j] = E[N_{i,o,j} \mid Y] / A_o[i, j]`,
    where ``N`` is the (unobserved) edge-use count. This is Fisher's identity,
    ``\nabla \log L(\theta) = E[\nabla \log f(X, Y; \theta) \mid Y]`` (Cappe,
    Moulines & Ryden, 2005, Section 10.2.3), evaluated in the raw (unconstrained)
    joint-edge parameters. Keys are ``(source, symbol, target)`` state labels.
    """
    mealy = hmm.to_mealy()
    idx = mealy.reindex()
    pi, joint = _emission_transition_tensors_from_mealy(mealy)
    edge_counts, _source_totals, _gamma0, loglik = _expected_edge_counts(pi, joint, list(observations))
    if not np.isfinite(loglik):
        raise ValueError("observations have zero probability under the model; score is undefined")
    result: dict[tuple[Hashable, Any, Hashable], float] = {}
    n_states = len(pi)
    for symbol, matrix in joint.items():
        for i in range(n_states):
            for j in range(n_states):
                prob = float(matrix[i, j])
                if prob > 0.0:
                    count = edge_counts.get((i, symbol, j), 0.0)
                    result[(idx.state(i), symbol, idx.state(j))] = count / prob
    return result


def _free_parameterization(
    joint: dict[Any, np.ndarray],
    n_states: int,
) -> tuple[list[tuple[int, Any, int]], list[tuple[int, Any, int]], list[int]]:
    """Build the free multinomial parameterization of the joint edge law.

    Each source state whose outgoing edges number ``k >= 2`` contributes ``k - 1``
    free parameters (its last edge in canonical order is the reference). Returns
    ``(free_edges, reference_by_param, source_by_param)``: the edge for each free
    parameter, the reference edge of its source block, and the source-state index.
    """
    free_edges: list[tuple[int, Any, int]] = []
    reference_by_param: list[tuple[int, Any, int]] = []
    source_by_param: list[int] = []
    for i in range(n_states):
        out_edges = sorted(
            ((i, symbol, j) for symbol, matrix in joint.items() for j in range(n_states) if matrix[i, j] > 0.0),
            key=lambda edge: (str(edge[1]), edge[2]),
        )
        if len(out_edges) < 2:
            continue
        reference = out_edges[-1]
        for edge in out_edges[:-1]:
            free_edges.append(edge)
            reference_by_param.append(reference)
            source_by_param.append(i)
    return free_edges, reference_by_param, source_by_param


def observed_information(hmm: HiddenMarkovModel, observations: Sequence[Any]) -> np.ndarray:
    r"""Return the observed information matrix via Louis' identity.

    The observed information ``J = -\partial^2 \log L / \partial\theta^2`` for the
    free multinomial parameters of the joint edge law is obtained from Louis'
    (1982) identity,

    .. math:: J = E[-\partial^2 \ell_c \mid Y] - \operatorname{Cov}(\partial \ell_c \mid Y),

    where :math:`\ell_c` is the complete-data log-likelihood (Cappe, Moulines &
    Ryden, 2005, Section 10.2.3). The complete-data information ``B`` follows from
    the expected edge counts; the conditional covariance of the complete-data
    score is computed exactly by a forward smoothing recursion for the first and
    second moments of the additive score functional. The matrix is ordered by
    :func:`free_parameter_labels`; an empty ``(0, 0)`` matrix is returned when the
    model has no free parameters.
    """
    mealy = hmm.to_mealy()
    pi, joint = _emission_transition_tensors_from_mealy(mealy)
    obs = list(observations)
    n_states = len(pi)

    free_edges, reference_by_param, source_by_param = _free_parameterization(joint, n_states)
    d = len(free_edges)
    if d == 0:
        return np.zeros((0, 0), dtype=float)

    edge_counts, _source_totals, _gamma0, loglik = _expected_edge_counts(pi, joint, obs)
    if not np.isfinite(loglik):
        raise ValueError("observations have zero probability under the model; information is undefined")

    prob_of = {edge: float(joint[edge[1]][edge[0], edge[2]]) for edge in set(free_edges) | set(reference_by_param)}

    # Complete-data information B = E[-d^2 l_c | Y], block-diagonal by source state.
    complete_information = np.zeros((d, d), dtype=float)
    for p in range(d):
        ref_p = reference_by_param[p]
        count_ref = edge_counts.get(ref_p, 0.0)
        ref_term = count_ref / prob_of[ref_p] ** 2
        for q in range(d):
            if source_by_param[p] != source_by_param[q]:
                continue
            value = ref_term
            if p == q:
                edge_p = free_edges[p]
                value += edge_counts.get(edge_p, 0.0) / prob_of[edge_p] ** 2
            complete_information[p, q] = value

    # Per-transition score contribution s(edge) as a d-vector (sparse per source block).
    edge_score: dict[tuple[int, Any, int], np.ndarray] = {}
    for p, edge in enumerate(free_edges):
        edge_score.setdefault(edge, np.zeros(d))[p] += 1.0 / prob_of[edge]
    for p, ref in enumerate(reference_by_param):
        edge_score.setdefault(ref, np.zeros(d))[p] += -1.0 / prob_of[ref]
    zero_d = np.zeros(d)

    # Forward smoothing recursion for E[S | Y] and E[S S^T | Y] of the additive
    # complete-data score functional S = sum_t s(edge_t).
    alpha_hat, _log_scales = _forward_scaled(pi, joint, obs)
    first = np.zeros((n_states, d), dtype=float)
    second = np.zeros((n_states, d, d), dtype=float)
    for t, symbol in enumerate(obs):
        matrix = joint.get(symbol)
        if matrix is None:
            continue
        weight = alpha_hat[t][:, None] * matrix  # weight[i, k] = P(X_t=i, X_{t+1}=k, Y_t | Y_{0:t-1})
        denom = weight.sum(axis=0)
        new_first = np.zeros((n_states, d), dtype=float)
        new_second = np.zeros((n_states, d, d), dtype=float)
        for k in range(n_states):
            if denom[k] <= 0.0:
                continue
            for i in range(n_states):
                if weight[i, k] <= 0.0:
                    continue
                retro = weight[i, k] / denom[k]  # P(X_t=i | X_{t+1}=k, Y_{0:t})
                s_vec = edge_score.get((i, symbol, k), zero_d)
                first_i = first[i]
                combined = first_i + s_vec
                new_first[k] += retro * combined
                cross = np.outer(first_i, s_vec)
                new_second[k] += retro * (second[i] + cross + cross.T + np.outer(s_vec, s_vec))
        first, second = new_first, new_second

    phi_final = alpha_hat[len(obs)]
    expected_score = phi_final @ first
    expected_outer = np.einsum("k,kpq->pq", phi_final, second)
    score_covariance = expected_outer - np.outer(expected_score, expected_score)
    return complete_information - score_covariance


def free_parameter_labels(hmm: HiddenMarkovModel) -> list[tuple[Hashable, Any, Hashable]]:
    """Return the ``(source, symbol, target)`` label for each free parameter.

    The order matches the rows and columns of :func:`observed_information` and the
    entries of :func:`standard_errors`.
    """
    mealy = hmm.to_mealy()
    idx = mealy.reindex()
    _pi, joint = _emission_transition_tensors_from_mealy(mealy)
    free_edges, _reference, _source = _free_parameterization(joint, len(idx))
    return [(idx.state(i), symbol, idx.state(j)) for i, symbol, j in free_edges]


def standard_errors(
    hmm: HiddenMarkovModel,
    observations: Sequence[Any],
) -> dict[tuple[Hashable, Any, Hashable], float]:
    r"""Return asymptotic standard errors of the free edge parameters.

    Standard errors are ``sqrt(diag(J^{-1}))`` where ``J`` is the
    :func:`observed_information` matrix (Cappe, Moulines & Ryden, 2005,
    Section 10.2.3). Uses the Moore-Penrose pseudoinverse when ``J`` is singular;
    a non-positive variance estimate (numerically unidentified parameter) yields
    ``nan``. Keyed by the labels from :func:`free_parameter_labels`.
    """
    labels = free_parameter_labels(hmm)
    information = observed_information(hmm, observations)
    if information.shape[0] == 0:
        return {}
    try:
        covariance = np.linalg.inv(information)
    except np.linalg.LinAlgError:
        covariance = np.linalg.pinv(information)
    variances = np.diag(covariance)
    with np.errstate(invalid="ignore"):
        errors = np.where(variances > 0.0, np.sqrt(variances), np.nan)
    return dict(zip(labels, (float(value) for value in errors), strict=True))


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
