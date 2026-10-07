"""Statistical complexity dimension of hidden Markov processes.

The mixed states of an :math:`N`-state HMM evolve on the
:math:`(N-1)`-simplex as a place-dependent iterated function system (IFS)
with maps :math:`f^{(x)}(\\eta) = \\eta T^{(x)} / \\eta T^{(x)} \\mathbf{1}`
chosen with probabilities :math:`\\eta T^{(x)} \\mathbf{1}`. When the mixed
states do not close finitely the statistical complexity diverges, and its
divergence rate is the information dimension :math:`d_\\mu` of the Blackwell
measure. Jurgens & Crutchfield bound (and, under the open set condition,
equate) it with the Lyapunov dimension of the IFS, in which the entropy rate
plays the role of the leading Lyapunov exponent :cite:`jurgens2021divergent`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import null_space

from sofic.exceptions import MixedStateExplosionError
from sofic.generators.base import StochasticModel
from sofic.generators.markov import MarkovChain
from sofic.generators.measures import (
    _numeric_symbol_tensors,
    _row_entropies_bits,
    batch_means_stderr,
    entropy_rate_hmm,
    mixed_state_walk,
)


@dataclass(frozen=True)
class StatisticalComplexityDimension:
    """Statistical complexity dimension and the quantities it is computed from.

    ``entropy_rate`` and ``lyapunov_exponents`` are in bits (base-2 logarithms)
    per symbol, exponents sorted in decreasing order. ``finite`` is true when
    the presentation is unifilar or its mixed states close within the state
    cap, in which case ``dimension`` is zero by definition and
    ``entropy_rate`` is exact.
    """

    dimension: float
    entropy_rate: float
    entropy_rate_stderr: float
    lyapunov_exponents: np.ndarray
    finite: bool


def ifs_lyapunov_dimension(entropy_rate: float, exponents: np.ndarray) -> float:
    r"""Lyapunov dimension of a mixed-state IFS (:cite:`jurgens2021divergent`, Eq. 13).

    With the (negative) exponents sorted as :math:`\lambda_1 \ge \lambda_2 \ge
    \cdots` and :math:`k` the largest index with
    :math:`h_\mu + \sum_{i \le k} \lambda_i > 0`,

    .. math:: d_\Gamma = k + \frac{h_\mu + \sum_{i \le k} \lambda_i}{|\lambda_{k+1}|},

    saturating at the simplex dimension when :math:`h_\mu + \sum_i \lambda_i
    \ge 0`. This is the Kaplan-Yorke formula with :math:`h_\mu` as the leading
    exponent; for two-state HMMs it is :math:`-h_\mu / \lambda_1` (Eq. 15).
    Both arguments must use the same logarithm base.
    """
    ordered = np.sort(np.asarray(exponents, dtype=float))[::-1]
    total = float(entropy_rate)
    for k, value in enumerate(ordered):
        if total + value <= 0.0:
            return float(k + (total / abs(value) if total > 0.0 else 0.0))
        total += value
    return float(len(ordered))


def statistical_complexity_dimension(
    model: StochasticModel,
    *,
    n_samples: int = 100_000,
    burn_in: int = 1_000,
    seed: int | np.random.Generator | None = None,
    n_batches: int = 20,
    max_states: int | None = 1_000,
) -> StatisticalComplexityDimension:
    r"""Estimate the statistical complexity dimension :math:`d_\mu` :cite:`jurgens2021divergent`.

    Follows the algorithm of Jurgens & Crutchfield (Sec. V.B): sample a
    mixed-state trajectory from the stationary state law (discarding
    ``burn_in`` steps), estimate the entropy rate as the Blackwell time average
    of :math:`H[X \mid \eta_t]` :cite:`jurgens2021shannon`, estimate the
    :math:`N - 1` Lyapunov exponents of the IFS by QR (pull-back) iteration of
    the Jacobians of :math:`f^{(x_t)}` restricted to the simplex tangent space
    :math:`\{v : v \mathbf{1} = 0\}`, and combine them with
    :func:`ifs_lyapunov_dimension`. Constant (synchronizing) maps contribute
    exponents of :math:`-\infty`.

    The Lyapunov dimension equals the information dimension of the Blackwell
    measure when the IFS satisfies the open set condition and is otherwise an
    upper bound (Eq. 14). It describes the mixed states of *this*
    presentation, which are the causal states unless distinct mixed states
    predict identically (the paper's Appendix B).

    A process with finitely many causal states has :math:`d_\mu = 0` by
    definition (Sec. V). Unifilar presentations are therefore reported as
    finite, as are presentations whose mixed-state construction closes within
    ``max_states`` beliefs (``None`` skips that check).
    """
    pi, tensors = _numeric_symbol_tensors(model)
    rng = seed if isinstance(seed, np.random.Generator) else np.random.default_rng(seed)
    beliefs, symbols, probabilities = mixed_state_walk(pi, tensors, n_samples=n_samples, burn_in=burn_in, rng=rng)
    entropies = _row_entropies_bits(probabilities)
    rate = float(entropies.mean())
    stderr = batch_means_stderr(entropies, n_batches)
    exponents = _lyapunov_spectrum(tensors, beliefs, symbols)

    exact_rate = _finite_entropy_rate(model, max_states)
    if exact_rate is not None:
        return StatisticalComplexityDimension(0.0, exact_rate, 0.0, exponents, True)
    return StatisticalComplexityDimension(ifs_lyapunov_dimension(rate, exponents), rate, stderr, exponents, False)


def _finite_entropy_rate(model: StochasticModel, max_states: int | None) -> float | None:
    """Return the exact entropy rate when the process has finitely many causal states, else ``None``."""
    if isinstance(model, MarkovChain):
        return float(model.entropy_rate())
    is_unifilar = getattr(model, "is_unifilar", None)
    if is_unifilar is not None and is_unifilar():
        return float(entropy_rate_hmm(model))
    if max_states is None:
        return None
    from sofic.generators.mixed_state_construction import build_mixed_state_presentation

    try:
        presentation = build_mixed_state_presentation(model.to_mealy(), max_states=max_states)
    except MixedStateExplosionError:
        return None
    return float(entropy_rate_hmm(presentation.to_recurrent()))


def _lyapunov_spectrum(
    tensors: np.ndarray,
    beliefs: np.ndarray,
    symbols: np.ndarray,
) -> np.ndarray:
    """QR-iterated Lyapunov exponents (bits per step) of the projective maps along a mixed-state path.

    For row-vector beliefs the Jacobian of :math:`f^{(x)}` at :math:`\\eta` acts as
    :math:`v \\mapsto v (T^{(x)} - T^{(x)}\\mathbf{1} \\, f^{(x)}(\\eta)) / \\eta T^{(x)} \\mathbf{1}`
    and preserves the tangent space spanned by the columns of ``basis``.
    """
    n_states = tensors.shape[1]
    if n_states < 2:
        return np.zeros(0)
    basis = null_space(np.ones((1, n_states)))
    frame = np.eye(n_states - 1)
    log_growth = np.zeros(n_states - 1)
    with np.errstate(divide="ignore"):
        for eta, x in zip(beliefs, symbols, strict=True):
            matrix = tensors[x]
            mass = eta @ matrix
            p = float(mass.sum())
            jacobian = (matrix - np.outer(matrix.sum(axis=1), mass / p)) / p
            reduced = basis.T @ jacobian.T @ basis
            frame, upper = np.linalg.qr(reduced @ frame)
            log_growth += np.log2(np.abs(np.diag(upper)))
    return np.sort(log_growth / len(symbols))[::-1]
