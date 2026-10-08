"""Pairwise correlations of hidden Markov processes in closed form.

Every quantity here is a function of the stationary state law :math:`\\pi` and
the symbol-labeled transition matrices :math:`T^{(x)}` (see
:mod:`sofic.generators.matrices`), with :math:`T = \\sum_x T^{(x)}`. For a real
observable :math:`f` of the emitted symbol let :math:`\\Omega = \\sum_x f(x)
T^{(x)}`. The autocorrelation is :math:`\\gamma(\\tau) = \\langle \\pi | \\Omega
T^{|\\tau| - 1} \\Omega | \\mathbf{1} \\rangle` for :math:`\\tau \\neq 0` and the
continuous part of the power spectrum is a resolvent of :math:`T` (Riechers &
Crutchfield :cite:`riechers2017spectral1`, Eqs. (3) and (5)). The lagged joint
symbol law :math:`P(X_0 = x, X_\\tau = y) = \\langle \\pi | T^{(x)} T^{\\tau - 1}
T^{(y)} | \\mathbf{1} \\rangle` gives the mutual information function. Pairwise
statistics can be flat (white) even for highly structured processes (Riechers &
Crutchfield :cite:`riechers2019fraudulent`).
"""

from __future__ import annotations

from collections.abc import Callable, Hashable, Mapping
from typing import Any

import numpy as np
import scipy.linalg

from sofic.generators.matrices import emission_tensors

Observable = Mapping[Hashable, float] | Callable[[Hashable], float]

_PERIPHERAL_TOL = 1e-8


def _process_tensors(hmm: Any) -> tuple[np.ndarray, list[Hashable], np.ndarray]:
    """Return the stationary law, the symbol order, and the stacked ``T^(x)`` of shape (k, n, n)."""
    pi, joint = emission_tensors(hmm, policy="stationary")
    symbols = list(joint)
    n = len(pi)
    stacked = np.array([np.asarray(joint[symbol], dtype=float) for symbol in symbols]).reshape(len(symbols), n, n)
    return np.asarray(pi, dtype=float), symbols, stacked


def _observable_values(symbols: list[Hashable], observable: Observable | None) -> np.ndarray:
    if observable is None:
        values = []
        for symbol in symbols:
            try:
                values.append(float(symbol))
            except (TypeError, ValueError) as exc:
                raise ValueError(f"symbol {symbol!r} is not numeric; pass an explicit observable") from exc
        return np.asarray(values, dtype=float)
    if isinstance(observable, Mapping):
        missing = [symbol for symbol in symbols if symbol not in observable]
        if missing:
            raise ValueError(f"observable has no value for symbols {missing!r}")
        return np.asarray([float(observable[symbol]) for symbol in symbols], dtype=float)
    return np.asarray([float(observable(symbol)) for symbol in symbols], dtype=float)


def _correlation_operators(
    hmm: Any, observable: Observable | None
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    """Return ``T``, ``<pi| Omega``, ``Omega |1>``, ``E[f(X_0)]`` and ``E[f(X_0)^2]``."""
    pi, symbols, stacked = _process_tensors(hmm)
    values = _observable_values(symbols, observable)
    ones = np.ones(len(pi))
    transition = stacked.sum(axis=0)
    omega = np.tensordot(values, stacked, axes=1)
    symbol_mass = np.einsum("i,kij,j->k", pi, stacked, ones)
    return transition, pi @ omega, omega @ ones, float(values @ symbol_mass), float(values**2 @ symbol_mass)


def autocorrelation(
    hmm: Any,
    max_lag: int,
    *,
    observable: Observable | None = None,
    centered: bool = False,
) -> np.ndarray:
    """Return the autocorrelation ``γ(τ) = E[f(X_0) f(X_τ)]`` for ``τ = 0..max_lag``.

    Computed exactly as :math:`\\gamma(0) = E[f(X_0)^2]` and
    :math:`\\gamma(\\tau) = \\langle \\pi | \\Omega \\, T^{\\tau-1} \\, \\Omega |
    \\mathbf{1} \\rangle` for :math:`\\tau \\geq 1`, where :math:`\\Omega = \\sum_x
    f(x) T^{(x)}` (Riechers & Crutchfield :cite:`riechers2017spectral1`, Eq. (3)),
    under the stationary state law.

    Parameters
    ----------
    hmm
        Hidden Markov model (anything with ``to_mealy()``) generating the process.
    max_lag
        Largest lag ``τ`` returned.
    observable
        Real value ``f(x)`` of each symbol, as a mapping or callable. Defaults to
        ``float(symbol)``, so string symbols ``"0"``, ``"1"`` map to 0, 1.
    centered
        Return the autocovariance ``γ(τ) − E[f(X_0)]²`` instead.

    Returns
    -------
    numpy.ndarray
        Array of length ``max_lag + 1`` indexed by ``τ``.
    """
    if max_lag < 0:
        raise ValueError("max_lag must be nonnegative")
    transition, row, col, mean, second_moment = _correlation_operators(hmm, observable)
    gamma = np.empty(max_lag + 1)
    gamma[0] = second_moment
    for lag in range(1, max_lag + 1):
        gamma[lag] = row @ col
        row = row @ transition
    if centered:
        gamma -= mean**2
    return gamma


def _peripheral_parts(transition: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(Q, X)``: the spectral projector onto ``|λ| = 1`` and ``X = (TQ)^#``.

    ``Q`` projects onto the invariant subspace of the unit-modulus eigenvalues
    along the complementary invariant subspace; ``X`` inverts ``T`` on the range
    of ``Q`` and vanishes on its kernel.
    """

    def peripheral(z: complex) -> bool:
        return abs(z) > 1.0 - _PERIPHERAL_TOL

    _, right, dim = scipy.linalg.schur(transition.astype(complex), output="complex", sort=peripheral)
    _, left, left_dim = scipy.linalg.schur(transition.T.astype(complex), output="complex", sort=peripheral)
    if dim != left_dim:
        raise np.linalg.LinAlgError("inconsistent peripheral spectrum")
    u = right[:, :dim]
    v_h = left[:, :dim].T
    projector = u @ np.linalg.solve(v_h @ u, v_h)
    inverse = u @ np.linalg.solve(v_h @ transition @ u, v_h)
    return projector, inverse


def power_spectrum(
    hmm: Any,
    omegas: Any,
    *,
    observable: Observable | None = None,
) -> np.ndarray:
    """Return the continuous part ``P_c(ω)`` of the power spectrum at angular frequencies ``omegas``.

    The power spectrum is :math:`P(\\omega) = \\lim_{N \\to \\infty} \\frac{1}{N}
    E \\left| \\sum_{t=0}^{N-1} f(X_t) e^{-i \\omega t} \\right|^2
    = \\sum_{\\tau \\in \\mathbb{Z}} \\gamma(\\tau) e^{-i \\omega \\tau}`.
    For a finite HMM it splits into a continuous part plus Dirac deltas at the
    arguments of the eigenvalues of :math:`T` on the unit circle. The continuous
    part is (Riechers & Crutchfield :cite:`riechers2017spectral1`, Eq. (5))

    .. math:: P_c(\\omega) = E[f(X_0)^2] + 2 \\, \\mathrm{Re} \\, \\langle \\pi |
              \\Omega (e^{i \\omega} I - T)^{-1} \\Omega | \\mathbf{1} \\rangle .

    The resolvent is singular at the delta frequencies (always at
    :math:`\\omega = 0`, from :math:`\\lambda = 1`), so it is evaluated with the
    peripheral spectrum projected out. With :math:`Q` the spectral projector
    onto the unit-modulus eigenvalues of :math:`T`,

    .. math:: P_c(\\omega) = E[f(X_0)^2] - \\langle \\pi | \\Omega (TQ)^{\\#}
              \\Omega | \\mathbf{1} \\rangle + 2 \\, \\mathrm{Re} \\, \\langle \\pi |
              \\Omega (e^{i \\omega} I - T(I - Q))^{-1} (I - Q) \\Omega |
              \\mathbf{1} \\rangle ,

    which equals the formula above away from the delta frequencies and is its
    continuous extension onto them (``P_c(0)`` is the Green--Kubo limit). The
    middle term is the periodic part of :math:`\\gamma` at lag zero; it contains
    :math:`E[f]^2`, so ``P_c`` is the same for ``f`` and ``f − E[f]``.

    The delta peaks -- the DC term :math:`2\\pi E[f]^2 \\delta(\\omega)` and, for
    processes with periodic components, peaks at the other roots of unity in
    the spectrum of :math:`T` -- are **excluded**. A purely periodic process
    has ``P_c ≡ 0``.

    Parameters
    ----------
    hmm
        Hidden Markov model generating the process.
    omegas
        Angular frequencies (scalar or array-like, radians per time step).
    observable
        Real value of each symbol; see :func:`autocorrelation`.

    Returns
    -------
    numpy.ndarray
        Real array with the shape of ``omegas``.
    """
    transition, row, col, _mean, second_moment = _correlation_operators(hmm, observable)
    frequencies = np.asarray(omegas, dtype=float)
    projector, inverse = _peripheral_parts(transition)
    eye = np.eye(len(row))
    damped = transition @ (eye - projector)
    damped_col = (eye - projector) @ col
    baseline = second_moment - float(np.real(row @ inverse @ col))
    spectrum = np.empty(frequencies.size)
    for k, omega in enumerate(frequencies.ravel()):
        resolvent_col = np.linalg.solve(np.exp(1j * omega) * eye - damped, damped_col)
        spectrum[k] = baseline + 2.0 * float(np.real(row @ resolvent_col))
    return spectrum.reshape(frequencies.shape)


def _mutual_information(joint: np.ndarray) -> float:
    """Mutual information (bits) between the row and column variables of ``joint``."""
    joint = np.asarray(joint, dtype=float)
    rows = joint.sum(axis=1, keepdims=True)
    cols = joint.sum(axis=0, keepdims=True)
    mask = joint > 0.0
    ratio = joint[mask] / (rows @ cols)[mask]
    return max(float(np.sum(joint[mask] * np.log2(ratio))), 0.0)


def mutual_information_function(hmm: Any, max_lag: int) -> np.ndarray:
    """Return the mutual information function ``I[X_0 : X_τ]`` (bits) for ``τ = 0..max_lag``.

    Uses the exact lagged joint law :math:`P(X_0 = x, X_\\tau = y) = \\langle \\pi |
    T^{(x)} T^{\\tau-1} T^{(y)} | \\mathbf{1} \\rangle` under the stationary state
    law (cf. Riechers & Crutchfield :cite:`riechers2017spectral1`, §V.A). Entry
    ``τ = 0`` is ``I[X_0 : X_0] = H[X_0]``. Unlike :func:`autocorrelation`, the
    result depends only on which symbols are distinct, not on their values.

    Returns
    -------
    numpy.ndarray
        Array of length ``max_lag + 1`` indexed by ``τ``.
    """
    if max_lag < 0:
        raise ValueError("max_lag must be nonnegative")
    pi, _symbols, stacked = _process_tensors(hmm)
    ones = np.ones(len(pi))
    transition = stacked.sum(axis=0)
    heads = np.einsum("i,kij->kj", pi, stacked)
    tails = stacked @ ones
    info = np.empty(max_lag + 1)
    info[0] = _mutual_information(np.diag(heads @ ones))
    for lag in range(1, max_lag + 1):
        info[lag] = _mutual_information(heads @ tails.T)
        heads = heads @ transition
    return info
