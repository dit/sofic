"""Predictive rate-distortion (causal information bottleneck) for finite ε-machines.

Lossy predictive features :math:`R` compress the causal state :math:`S` while
keeping information about the length-:math:`L` future :math:`X_{0:L}`. They
solve the information bottleneck (Tishby, Pereira & Bialek
:cite:`tishby2000information`)

.. math:: \\min_{q(r \\mid s)} \\; I[S; R] - \\beta \\, I[R; X_{0:L}],

with the Markov chain :math:`R - S - X_{0:L}`. Because the causal state is a
sufficient statistic of the past, compressing :math:`S` is equivalent to
compressing the whole past (optimal causal inference, Still, Crutchfield &
Ellison :cite:`still2010optimal`; predictive rate-distortion, Marzen &
Crutchfield :cite:`Marzen2016`). The self-consistent equations are iterated
Blahut--Arimoto style (Cover & Thomas :cite:`Cover2006`, §10.8).
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any

import numpy as np

from sofic.generators.correlations import _mutual_information
from sofic.generators.matrices import emission_tensors

_DIRECTION_DECIMALS = 12


@dataclass(frozen=True)
class PredictiveRateDistortionCurve:
    """Points of a predictive rate-distortion (causal information bottleneck) curve.

    Attributes
    ----------
    beta
        Trade-off parameters, ascending.
    rate
        Coding rate ``I[S; R]`` in bits at each ``beta``.
    relevant_information
        Retained predictive information ``I[R; X_{0:L}]`` in bits.
    distortion
        Lost predictive information ``I[S; X_{0:L}] − I[R; X_{0:L}]`` in bits.
    future_length
        Future length ``L`` used for the morphs ``P(X_{0:L} | S)``.
    predictive_information
        ``I[S; X_{0:L}]`` in bits, the supremum of ``relevant_information``
        (at most the excess entropy ``E``).
    statistical_complexity
        ``C_μ = H[S]`` in bits, the supremum of ``rate``.
    """

    beta: np.ndarray
    rate: np.ndarray
    relevant_information: np.ndarray
    distortion: np.ndarray
    future_length: int
    predictive_information: float
    statistical_complexity: float


def _future_classes(transition_stack: np.ndarray, columns: np.ndarray) -> np.ndarray:
    """Prepend every symbol to each future class and merge classes with proportional morphs.

    ``columns[:, k]`` is ``P(w | S = s)`` summed over the words ``w`` in class ``k``.
    Words with proportional columns induce the same posterior over ``S``, so
    merging them leaves every quantity of the bottleneck ``R - S - X_{0:L}``
    unchanged; prepending is linear, so merged classes extend consistently.
    """
    extended = np.concatenate([matrix @ columns for matrix in transition_stack], axis=1)
    totals = extended.sum(axis=0)
    extended = extended[:, totals > 0.0]
    totals = totals[totals > 0.0]
    keys = np.round(extended / totals, _DIRECTION_DECIMALS)
    _unique, inverse = np.unique(keys, axis=1, return_inverse=True)
    merged = np.zeros((extended.shape[0], int(inverse.max()) + 1))
    np.add.at(merged.T, inverse.ravel(), extended.T)
    return merged


def _bottleneck_inputs(
    machine: Any,
    future_length: int | None,
    *,
    tol: float,
    max_future_length: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Return ``p(s)``, the morph matrix ``P(y | s)`` over merged futures, and ``L``."""
    pi, joint = emission_tensors(machine, policy="stationary")
    pi = np.asarray(pi, dtype=float)
    stacked = np.array([np.asarray(matrix, dtype=float) for matrix in joint.values()])
    columns = np.ones((len(pi), 1))
    if future_length is not None:
        if future_length < 0:
            raise ValueError("future_length must be nonnegative")
        for _ in range(future_length):
            columns = _future_classes(stacked, columns)
        return pi, columns, future_length

    excess = float(machine.excess_entropy())
    for length in range(max_future_length + 1):
        if excess - _mutual_information(pi[:, None] * columns) <= tol:
            return pi, columns, length
        if length < max_future_length:
            columns = _future_classes(stacked, columns)
    warnings.warn(
        f"I[S; X_0:L] did not reach E within tol={tol} by L={max_future_length}",
        RuntimeWarning,
        stacklevel=3,
    )
    return pi, columns, max_future_length


def _iterate(
    p_s: np.ndarray,
    morphs: np.ndarray,
    log_morphs: np.ndarray,
    beta: float,
    encoder: np.ndarray,
    *,
    max_iter: int,
    atol: float,
) -> np.ndarray:
    """Iterate the information-bottleneck self-consistent equations from ``encoder``."""
    negentropy = np.sum(morphs * log_morphs, axis=1, keepdims=True)
    for _ in range(max_iter):
        q_r = p_s @ encoder
        joint_ry = (p_s[:, None] * encoder).T @ morphs
        with np.errstate(divide="ignore", invalid="ignore"):
            decoder = np.where(q_r[:, None] > 0.0, joint_ry / q_r[:, None], 0.0)
            log_q_r = np.where(q_r > 0.0, np.log(np.maximum(q_r, 1e-300)), -np.inf)
        divergence = negentropy - morphs @ np.log(np.maximum(decoder, 1e-300)).T
        logits = log_q_r[None, :] - beta * divergence
        logits -= logits.max(axis=1, keepdims=True)
        updated = np.exp(logits)
        updated /= updated.sum(axis=1, keepdims=True)
        if np.max(np.abs(updated - encoder)) < atol:
            return updated
        encoder = updated
    return encoder


def _information(p_s: np.ndarray, morphs: np.ndarray, encoder: np.ndarray) -> tuple[float, float]:
    joint_sr = p_s[:, None] * encoder
    return _mutual_information(joint_sr), _mutual_information(joint_sr.T @ morphs)


def predictive_rate_distortion(
    machine: Any,
    betas: Any = 50,
    *,
    beta_range: tuple[float, float] = (0.5, 500.0),
    future_length: int | None = None,
    tol: float = 1e-6,
    max_future_length: int = 64,
    restarts: int = 4,
    max_iter: int = 2000,
    seed: int | None = 0,
) -> PredictiveRateDistortionCurve:
    """Return the predictive rate-distortion curve of a finite ε-machine.

    For each trade-off ``β`` the causal states ``S`` are compressed into
    features ``R`` (with ``|R| = |S|``, which suffices) by minimizing
    :math:`I[S; R] - \\beta \\, I[R; X_{0:L}]` (Tishby et al.
    :cite:`tishby2000information`; Still et al. :cite:`still2010optimal`;
    Marzen & Crutchfield :cite:`Marzen2016`). The future morphs
    :math:`P(X_{0:L} = w \\mid S = s) = \\langle \\delta_s | T^{(w)} | \\mathbf{1}
    \\rangle` are exact; futures with proportional morph columns are merged,
    which is lossless for the bottleneck and keeps long futures cheap.

    The encoder is found by iterating the self-consistent equations
    :math:`q(r \\mid s) \\propto q(r) \\exp(-\\beta D_{KL}[P(X_{0:L} \\mid s) \\,\\|\\,
    q(X_{0:L} \\mid r)])` (Blahut--Arimoto, Cover & Thomas :cite:`Cover2006`,
    §10.8). The ``β`` grid is annealed in ascending order: each ``β`` starts
    from a perturbation of the previous solution, the one-to-one encoder
    ``R = S`` (near the bifurcation from the collapsed ``R`` the nontrivial
    branch has a small basin that random starts tend to miss), and
    ``restarts`` random encoders; the candidate with the lowest Lagrangian is
    kept.

    Parameters
    ----------
    machine
        A finite :class:`~sofic.generators.epsilon_machine.EpsilonMachine`.
        Any other HMM is converted with ``EpsilonMachine.from_hmm`` first.
    betas
        Trade-off values, or an integer number of geometrically spaced values
        over ``beta_range``. For ``β ≤ 1`` the optimum is the trivial ``R``
        (zero rate).
    beta_range
        ``(low, high)`` used when ``betas`` is an integer.
    future_length
        Future length ``L``. By default the smallest ``L`` with
        ``E − I[S; X_{0:L}] ≤ tol`` is used, where ``E`` is the excess entropy;
        ``I[S; X_{0:L}]`` increases to ``I[S; X_{0:∞}] = E``.
    tol
        Tolerance for choosing the default ``future_length``.
    max_future_length
        Cap on the default ``future_length`` (a ``RuntimeWarning`` is issued if
        it is reached before the tolerance).
    restarts
        Random restarts per ``β``.
    max_iter
        Maximum self-consistent iterations per start.
    seed
        Seed for the random restarts.

    Returns
    -------
    PredictiveRateDistortionCurve
    """
    from sofic.generators.epsilon_machine import EpsilonMachine

    if not isinstance(machine, EpsilonMachine):
        machine = EpsilonMachine.from_hmm(machine)
    if isinstance(betas, int | np.integer):
        beta_grid = np.geomspace(beta_range[0], beta_range[1], int(betas))
    else:
        beta_grid = np.sort(np.asarray(betas, dtype=float).ravel())
    if np.any(beta_grid < 0.0):
        raise ValueError("betas must be nonnegative")

    p_s, morphs_joint, length = _bottleneck_inputs(machine, future_length, tol=tol, max_future_length=max_future_length)
    keep = p_s > 0.0
    p_s = p_s[keep]
    morphs = morphs_joint[keep]
    with np.errstate(divide="ignore"):
        log_morphs = np.where(morphs > 0.0, np.log(np.maximum(morphs, 1e-300)), 0.0)
    total = _mutual_information(p_s[:, None] * morphs)
    complexity = _mutual_information(np.diag(p_s))

    rng = np.random.default_rng(seed)
    n = len(p_s)
    encoder = np.full((n, n), 1.0 / n)
    rates = np.empty(len(beta_grid))
    relevant = np.empty(len(beta_grid))
    for k, beta in enumerate(beta_grid):
        warm = 0.9 * encoder + 0.1 * rng.dirichlet(np.ones(n), size=n)
        starts = [warm, np.eye(n), *(rng.dirichlet(np.ones(n), size=n) for _ in range(restarts))]
        best = None
        for start in starts:
            candidate = _iterate(p_s, morphs, log_morphs, beta, start, max_iter=max_iter, atol=1e-12)
            rate, info = _information(p_s, morphs, candidate)
            lagrangian = rate - beta * info
            if best is None or lagrangian < best[0] - 1e-12:
                best = (lagrangian, rate, info, candidate)
        _, rates[k], relevant[k], encoder = best

    relevant = np.minimum(relevant, total)
    return PredictiveRateDistortionCurve(
        beta=beta_grid,
        rate=rates,
        relevant_information=relevant,
        distortion=total - relevant,
        future_length=length,
        predictive_information=total,
        statistical_complexity=complexity,
    )
