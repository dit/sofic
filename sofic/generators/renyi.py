"""Rényi entropy rates and the thermodynamic spectrum of a stationary process.

For a stationary process with length-``n`` block law :math:`P`, the Rényi
entropy rate of order :math:`\\alpha \\ge 0` and the pressure are

.. math::

   h_\\alpha = \\lim_{n \\to \\infty} \\tfrac{1}{n} H_\\alpha[X_{0:n}], \\qquad
   \\mathcal{P}(\\beta) = \\lim_{n \\to \\infty} \\tfrac{1}{n} \\log_2 \\sum_{w} P(w)^\\beta,

so that :math:`h_\\alpha = \\mathcal{P}(\\alpha) / (1 - \\alpha)`. On a unifilar
presentation with labeled matrices :math:`T^{(x)}`, restricted to the
stationary support,

.. math:: \\mathcal{P}(\\beta) = \\log_2 \\rho\\Big(\\sum_x \\big(T^{(x)}\\big)^{\\circ \\beta}\\Big)

for :math:`\\beta \\ge 0`, where :math:`\\circ \\beta` is the entrywise power of
the positive entries and :math:`\\rho` the spectral radius. For Markov chains
this is :cite:`Rached2001`. For a unifilar presentation each start state ``s``
emits ``w`` along at most one path, of probability :math:`p(w \\mid s)`, and
:math:`P(w) = \\sum_s \\pi_s p(w \\mid s)`. For :math:`\\beta \\ge 0` and
:math:`N` support states,

.. math::

   N^{-1} \\textstyle\\sum_s (\\pi_s p(w \\mid s))^\\beta \\le P(w)^\\beta
   \\le N^\\beta \\textstyle\\sum_s (\\pi_s p(w \\mid s))^\\beta,

and :math:`\\sum_w \\sum_s (\\pi_s p(w \\mid s))^\\beta = (\\pi^{\\circ\\beta})^\\top A_\\beta^n \\mathbf{1}`
with :math:`A_\\beta = \\sum_x (T^{(x)})^{\\circ \\beta}`. Both vectors are
positive on the support, so the growth rate is :math:`\\rho(A_\\beta)`.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np
from scipy.optimize import minimize_scalar

if TYPE_CHECKING:
    from sofic.generators.base import HiddenMarkovModel
    from sofic.generators.mealy import MealyHMM

__all__ = ["pressure", "rate_function", "renyi_entropy_rate"]

_TOL = 1e-12
_BETA_MAX = 64.0


def _unifilar_mealy(model: HiddenMarkovModel, max_states: int) -> MealyHMM:
    from sofic.generators.epsilon_machine import EpsilonMachine

    mealy = model.to_mealy()
    if not mealy.is_unifilar():
        mealy = EpsilonMachine.from_hmm(mealy, max_states=max_states)
    return mealy


def _stationary_support(pi: np.ndarray, matrices: Sequence[np.ndarray]) -> np.ndarray:
    """Return the mask of states reachable from those of positive stationary mass.

    Closing under transitions keeps recurrent states whose computed stationary
    mass underflows the threshold.
    """
    adjacency = sum((np.asarray(m, dtype=float) > 0.0) for m in matrices)
    keep = pi > _TOL
    frontier = keep.copy()
    while frontier.any():
        reached = np.asarray(adjacency[frontier].sum(axis=0) > 0) & ~keep
        keep |= reached
        frontier = reached
    return keep


class _Spectrum:
    """Labeled matrices of a unifilar presentation on its stationary support."""

    def __init__(self, model: HiddenMarkovModel, *, max_states: int) -> None:
        from sofic.generators.matrices import symbol_matrices

        self.mealy = _unifilar_mealy(model, max_states)
        pi = np.asarray(self.mealy.stationary_distribution(), dtype=float)
        raw = list(symbol_matrices(self.mealy).values())
        keep = _stationary_support(pi, raw)
        matrices = [np.asarray(m, dtype=float)[np.ix_(keep, keep)] for m in raw]
        self.matrices = [m for m in matrices if np.any(m > 0.0)]
        self.size = int(keep.sum())
        self.mu = self._max_cycle_mean()

    def _max_cycle_mean(self) -> float:
        """Karp's maximum mean cycle weight of the ``log2`` edge probabilities."""
        n = self.size
        weights = np.full((n, n), -np.inf)
        for m in self.matrices:
            with np.errstate(divide="ignore"):
                weights = np.maximum(weights, np.where(m > 0.0, np.log2(np.where(m > 0.0, m, 1.0)), -np.inf))
        walks = np.zeros((n + 1, n))
        for k in range(1, n + 1):
            walks[k] = np.max(walks[k - 1][:, None] + weights, axis=0)
        best = -np.inf
        for v in range(n):
            if not np.isfinite(walks[n, v]):
                continue
            finite = np.isfinite(walks[:n, v])
            ks = np.arange(n)[finite]
            best = max(best, float(np.min((walks[n, v] - walks[ks, v]) / (n - ks))))
        return best

    def pressure(self, beta: float) -> float:
        scale = beta * self.mu
        total = np.zeros((self.size, self.size))
        for m in self.matrices:
            positive = m > 0.0
            logs = np.log2(np.where(positive, m, 1.0))
            total += np.where(positive, np.exp2(beta * logs - scale), 0.0)
        return scale + float(np.log2(np.max(np.abs(np.linalg.eigvals(total)))))

    def pressure_slope_at_zero(self, step: float = 1e-5) -> float:
        return (-3.0 * self.pressure(0.0) + 4.0 * self.pressure(step) - self.pressure(2.0 * step)) / (2.0 * step)


def _check_order(value: float, name: str) -> float:
    value = float(value)
    if np.isnan(value) or value < 0.0:
        raise ValueError(f"{name} must be non-negative, got {value}")
    return value


def renyi_entropy_rate(model: HiddenMarkovModel, alpha: float, *, max_states: int = 10_000) -> float:
    """Return the Rényi entropy rate :math:`h_\\alpha` in bits per symbol.

    :math:`h_\\alpha = \\lim_n \\tfrac{1}{n} H_\\alpha[X_{0:n}]
    = \\tfrac{1}{1 - \\alpha} \\log_2 \\rho\\big(\\sum_x (T^{(x)})^{\\circ \\alpha}\\big)`
    on a unifilar presentation (:cite:`Rached2001` for Markov chains; see the
    module docstring for unifilar HMMs). ``alpha`` ranges over
    :math:`[0, \\infty]`:

    * ``alpha = 0`` is the topological entropy of the process support,
      :math:`\\log_2 \\rho` of the support adjacency matrix;
    * ``alpha = 1`` is the Shannon entropy rate :math:`h_\\mu`
      (:meth:`~sofic.generators.base.HiddenMarkovModel.entropy_rate`);
    * ``alpha = inf`` is the min-entropy rate
      :math:`-\\lim_n \\tfrac{1}{n} \\log_2 \\max_w P(w)`, minus the maximum mean
      :math:`\\log_2` edge probability over cycles (Karp's algorithm).

    Non-unifilar input is first converted with
    :meth:`EpsilonMachine.from_hmm(model, max_states=max_states)
    <sofic.generators.epsilon_machine.EpsilonMachine.from_hmm>`, which raises
    :class:`~sofic.exceptions.MixedStateExplosionError` when the mixed states
    do not close.

    Examples
    --------
    >>> from sofic.examples import bernoulli
    >>> round(renyi_entropy_rate(bernoulli(0.25), 2.0), 6)
    0.678072
    """
    alpha = _check_order(alpha, "alpha")
    spectrum = _Spectrum(model, max_states=max_states)
    if np.isinf(alpha):
        return -spectrum.mu
    if alpha == 1.0:
        return float(spectrum.mealy.entropy_rate())
    return spectrum.pressure(alpha) / (1.0 - alpha)


def pressure(model: HiddenMarkovModel, beta: float, *, max_states: int = 10_000) -> float:
    """Return the pressure :math:`\\mathcal{P}(\\beta) = \\lim_n \\tfrac{1}{n} \\log_2 \\sum_w P(w)^\\beta`.

    Computed as :math:`\\log_2 \\rho\\big(\\sum_x (T^{(x)})^{\\circ \\beta}\\big)` on a
    unifilar presentation for finite :math:`\\beta \\ge 0` (see the module
    docstring). :math:`\\mathcal{P}` is convex and non-increasing, with
    :math:`\\mathcal{P}(0) = h_0`, :math:`\\mathcal{P}(1) = 0` and
    :math:`\\mathcal{P}(\\beta) = (1 - \\beta) h_\\beta`. Negative :math:`\\beta`
    is rejected: there the matrix formula weights each word by its least
    likely start state and need not equal the pressure. Non-unifilar input is
    handled as in :func:`renyi_entropy_rate`.

    Examples
    --------
    >>> from sofic.examples import golden_mean
    >>> abs(pressure(golden_mean(0.5), 1.0)) < 1e-12
    True
    """
    beta = _check_order(beta, "beta")
    if np.isinf(beta):
        raise ValueError("beta must be finite")
    return _Spectrum(model, max_states=max_states).pressure(beta)


def rate_function(
    model: HiddenMarkovModel,
    values: float | Sequence[float] | np.ndarray,
    *,
    beta_max: float = _BETA_MAX,
    max_states: int = 10_000,
) -> np.ndarray:
    """Return the large-deviation rate function of :math:`-\\tfrac{1}{n} \\log_2 P(X_{0:n})`.

    The scaled cumulant generating function of
    :math:`U_n = -\\tfrac{1}{n} \\log_2 P(X_{0:n})` in base 2 is
    :math:`\\lim_n \\tfrac{1}{n} \\log_2 \\mathbb{E}[2^{n t U_n}] = \\mathcal{P}(1 - t)`,
    so by the Gärtner-Ellis theorem :cite:`DemboZeitouni1998` (§2.3) the rate
    function, in bits per symbol, is the Legendre transform

    .. math:: I(u) = \\sup_{\\beta} \\big[(1 - \\beta) u - \\mathcal{P}(\\beta)\\big].

    Only :math:`\\beta \\in [0, \\infty)` is available (see :func:`pressure`), so
    ``I`` is computed on :math:`[h_\\infty, u_0]` with
    :math:`u_0 = -\\mathcal{P}'(0)`, the typical value of ``U_n`` under the
    measure of maximal entropy on the support. The supremum is taken over
    :math:`[0, \\beta_{\\max}]` by a grid search refined with
    :func:`scipy.optimize.minimize_scalar`; near :math:`h_\\infty` the optimum
    escapes to :math:`\\beta \\to \\infty`, so values there are lower bounds that
    tighten with ``beta_max``. ``I(h_\\mu) = 0`` and ``I \\ge 0``. Values below
    :math:`h_\\infty` give ``inf`` (no word is that likely) and values above
    :math:`u_0` give ``nan``. Returns an array shaped like ``values``.
    """
    beta_max = _check_order(beta_max, "beta_max")
    spectrum = _Spectrum(model, max_states=max_states)
    h_inf = -spectrum.mu
    u_0 = -spectrum.pressure_slope_at_zero()
    grid = np.concatenate([[0.0], np.geomspace(1e-3, beta_max, 96)]) if beta_max > 0 else np.zeros(1)
    pressures = np.array([spectrum.pressure(beta) for beta in grid])

    def evaluate(u: float) -> float:
        if u < h_inf - 1e-9:
            return np.inf
        if u > u_0 + 1e-6:
            return np.nan
        objective = (1.0 - grid) * u - pressures
        best = int(np.argmax(objective))
        if 0 < best < grid.size - 1:
            result = minimize_scalar(
                lambda beta: -((1.0 - beta) * u - spectrum.pressure(beta)),
                bounds=(grid[best - 1], grid[best + 1]),
                method="bounded",
                options={"xatol": 1e-10},
            )
            return max(float(-result.fun), float(objective[best]), 0.0)
        return max(float(objective[best]), 0.0)

    flat = np.asarray(values, dtype=float)
    return np.vectorize(evaluate, otypes=[float])(flat)
