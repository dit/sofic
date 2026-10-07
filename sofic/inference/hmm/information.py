"""Score and observed information of hidden Markov models.

The score follows from the Fisher identity and the observed information from
Louis' identity (Cappe, Moulines & Ryden, 2005, Section 10.2.3).
"""

from __future__ import annotations

from collections.abc import Hashable, Sequence
from typing import Any

import numpy as np

from sofic.generators.base import HiddenMarkovModel
from sofic.generators.matrices import emission_tensors, symbol_matrices
from sofic.inference.hmm.em import _expected_edge_counts
from sofic.inference.hmm.filtering import _forward_scaled


def score(hmm: HiddenMarkovModel, observations: Sequence[Any]) -> dict[tuple[Hashable, Any, Hashable], float]:
    r"""Return the score (gradient of the log-likelihood) via the Fisher identity.

    For each edge ``i --o--> j``, returns
    :math:`\partial \log P(Y) / \partial A_o[i, j] = E[N_{i,o,j} \mid Y] / A_o[i, j]`,
    where ``N`` is the (unobserved) edge-use count. This is Fisher's identity,
    ``\nabla \log L(\theta) = E[\nabla \log f(X, Y; \theta) \mid Y]`` (Cappe,
    Moulines & Ryden, 2005, Section 10.2.3), evaluated in the raw (unconstrained)
    joint-edge parameters. Keys are ``(source, symbol, target)`` state labels.

    The score is the gradient of the *natural* log-likelihood
    :math:`\ln P(Y) = \ln 2 \cdot` :func:`log_likelihood`, the convention under
    which the observed information and standard errors are standard.
    """
    mealy = hmm.to_mealy()
    idx = mealy.reindex()
    pi, joint = emission_tensors(mealy)
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
    pi, joint = emission_tensors(mealy)
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
    joint = symbol_matrices(mealy)
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
