"""Tests for closed-form correlations, power spectra and mutual information functions."""

from __future__ import annotations

import math
from collections import defaultdict

import numpy as np
import pytest
from hypothesis import assume, given, settings

from sofic.examples import even_process, golden_mean, iid, nonunifilar_golden_mean, periodic
from sofic.generators import EpsilonMachine, MealyHMM
from sofic.generators.correlations import autocorrelation, mutual_information_function, power_spectrum
from sofic.graph import ATTR_EMISSION, ATTR_PROB
from sofic.testing.strategies import mealy_hmms
from tests.oracles import word_distribution

OMEGAS = np.linspace(0.0, np.pi, 9)


def _relabeled(hmm: MealyHMM, symbol_map: dict | None = None) -> MealyHMM:
    """Copy ``hmm`` with states renamed (and symbols remapped by ``symbol_map``)."""
    symbol_map = symbol_map or {}
    rename = {state: ("relabeled", i) for i, state in enumerate(reversed(list(hmm.states())))}
    out = MealyHMM(
        observation_alphabet=frozenset(symbol_map.get(s, s) for s in hmm.observation_alphabet),
        initial_distribution={rename[s]: mass for s, mass in hmm.initial_distribution.items()},
    )
    for state in rename.values():
        out.graph.add_state(state)
    for transition in hmm.transitions():
        symbol = transition.data[ATTR_EMISSION]
        out.add_transition(
            rename[transition.source],
            rename[transition.target],
            symbol_map.get(symbol, symbol),
            transition.data[ATTR_PROB],
        )
    return out


def _brute_force_lagged(hmm: MealyHMM, lag: int) -> dict[tuple, float]:
    """``P(X_0, X_lag)`` by marginalizing the oracle's length ``lag + 1`` word law."""
    joint: dict[tuple, float] = defaultdict(float)
    for word, prob in word_distribution(hmm, lag + 1).items():
        joint[(word[0], word[-1])] += prob
    return joint


def _mi_bits(joint: dict[tuple, float]) -> float:
    left: dict = defaultdict(float)
    right: dict = defaultdict(float)
    for (x, y), p in joint.items():
        left[x] += p
        right[y] += p
    return sum(p * math.log2(p / (left[x] * right[y])) for (x, y), p in joint.items() if p > 0)


# --------------------------------------------------------------------------- i.i.d.


def test_iid_correlations_are_trivial() -> None:
    process = iid(2)
    gamma = autocorrelation(process, 5)
    assert gamma[0] == pytest.approx(0.5)
    np.testing.assert_allclose(gamma[1:], 0.25)
    np.testing.assert_allclose(autocorrelation(process, 5, centered=True)[1:], 0.0, atol=1e-15)
    np.testing.assert_allclose(mutual_information_function(process, 5), [1.0, 0, 0, 0, 0, 0], atol=1e-12)
    np.testing.assert_allclose(power_spectrum(process, OMEGAS), 0.25)


def test_iid_with_custom_observable_has_flat_spectrum_at_variance() -> None:
    process = iid(("a", "b", "c"))
    values = {"a": -1.0, "b": 0.0, "c": 2.0}
    mean = sum(values.values()) / 3
    var = sum(v**2 for v in values.values()) / 3 - mean**2
    np.testing.assert_allclose(autocorrelation(process, 3, observable=values)[1:], mean**2)
    np.testing.assert_allclose(power_spectrum(process, OMEGAS, observable=values), var)
    np.testing.assert_allclose(power_spectrum(process, OMEGAS, observable=values.__getitem__), var)


def test_non_numeric_symbols_need_observable() -> None:
    with pytest.raises(ValueError, match="not numeric"):
        autocorrelation(iid(("a", "b")), 2)
    with pytest.raises(ValueError, match="no value"):
        autocorrelation(iid(("a", "b")), 2, observable={"a": 1.0})
    with pytest.raises(ValueError, match="nonnegative"):
        mutual_information_function(iid(2), -1)


# --------------------------------------------------------------------------- periodic


def test_period_two() -> None:
    process = periodic("01")
    np.testing.assert_allclose(autocorrelation(process, 4), [0.5, 0.0, 0.5, 0.0, 0.5])
    np.testing.assert_allclose(mutual_information_function(process, 4), 1.0)
    np.testing.assert_allclose(power_spectrum(process, OMEGAS), 0.0, atol=1e-12)


@pytest.mark.parametrize("word", ["001", "0011", "00101"])
def test_periodic_processes_have_no_continuous_spectrum(word: str) -> None:
    np.testing.assert_allclose(power_spectrum(periodic(word), OMEGAS), 0.0, atol=1e-12)


# --------------------------------------------------------------------------- golden mean


@pytest.mark.parametrize("p", [0.3, 0.5, 0.8])
def test_golden_mean_closed_form(p: float) -> None:
    process = golden_mean(p)
    lam = p - 1.0
    mean = (1.0 - p) / (2.0 - p)
    var = mean * (1.0 - mean)
    lags = np.arange(10)
    np.testing.assert_allclose(autocorrelation(process, 9, centered=True), var * lam**lags, atol=1e-14)
    np.testing.assert_allclose(autocorrelation(process, 9), mean**2 + var * lam**lags, atol=1e-14)
    lorentzian = var * (1.0 - lam**2) / (1.0 - 2.0 * lam * np.cos(OMEGAS) + lam**2)
    np.testing.assert_allclose(power_spectrum(process, OMEGAS), lorentzian)


def test_golden_mean_mi_decays_with_squared_subdominant_eigenvalue() -> None:
    p = 0.6
    info = mutual_information_function(golden_mean(p), 10)
    ratios = info[6:] / info[5:-1]
    np.testing.assert_allclose(ratios, (p - 1.0) ** 2, rtol=1e-2)


# --------------------------------------------------------------------------- oracles


@pytest.mark.parametrize("make", [lambda: golden_mean(0.4), even_process, lambda: nonunifilar_golden_mean()])
def test_against_brute_force_word_enumeration(make) -> None:
    process = make()
    process.initial_distribution = dict(zip(process.reindex().states, process.stationary_distribution(), strict=True))
    gamma = autocorrelation(process, 4)
    info = mutual_information_function(process, 4)
    for lag in range(1, 5):
        joint = _brute_force_lagged(process, lag)
        assert gamma[lag] == pytest.approx(sum(float(x) * float(y) * p for (x, y), p in joint.items()))
        assert info[lag] == pytest.approx(_mi_bits(joint), abs=1e-12)


@pytest.mark.parametrize("make", [lambda: golden_mean(0.3), even_process, nonunifilar_golden_mean])
def test_wiener_khinchin(make) -> None:
    process = make()
    lags = 512
    cov = autocorrelation(process, lags, centered=True)
    symmetric = np.concatenate([cov, cov[-2:0:-1]])
    estimate = np.real(np.fft.fft(symmetric))
    omegas = 2.0 * np.pi * np.arange(len(symmetric)) / len(symmetric)
    np.testing.assert_allclose(power_spectrum(process, omegas), estimate, atol=1e-9)


# --------------------------------------------------------------------------- invariants


@settings(max_examples=40, deadline=None)
@given(mealy_hmms(max_states=3))
def test_mi_bounds(hmm: MealyHMM) -> None:
    info = mutual_information_function(hmm, 6)
    assert np.all(info >= 0.0)
    assert np.all(info[1:] <= info[0] + 1e-9)


@settings(max_examples=40, deadline=None)
@given(mealy_hmms(max_states=3))
def test_invariant_under_state_and_symbol_relabeling(hmm: MealyHMM) -> None:
    assume(hmm.is_irreducible())
    relabeled = _relabeled(hmm)
    np.testing.assert_allclose(autocorrelation(relabeled, 6), autocorrelation(hmm, 6), atol=1e-10)
    np.testing.assert_allclose(power_spectrum(relabeled, OMEGAS), power_spectrum(hmm, OMEGAS), atol=1e-8)
    swapped = _relabeled(hmm, {"0": "1", "1": "0"})
    np.testing.assert_allclose(mutual_information_function(swapped, 6), mutual_information_function(hmm, 6), atol=1e-10)


@settings(max_examples=30, deadline=None)
@given(mealy_hmms(max_states=3))
def test_invariant_under_epsilon_machine_presentation(hmm: MealyHMM) -> None:
    assume(hmm.is_irreducible() and hmm.is_unifilar())
    machine = EpsilonMachine.from_hmm(hmm)
    np.testing.assert_allclose(autocorrelation(machine, 6), autocorrelation(hmm, 6), atol=1e-10)
    np.testing.assert_allclose(mutual_information_function(machine, 6), mutual_information_function(hmm, 6), atol=1e-10)
    np.testing.assert_allclose(power_spectrum(machine, OMEGAS), power_spectrum(hmm, OMEGAS), atol=1e-8)


def test_nonunifilar_and_epsilon_machine_agree() -> None:
    hmm = nonunifilar_golden_mean()
    machine = EpsilonMachine.from_hmm(hmm)
    np.testing.assert_allclose(autocorrelation(machine, 8), autocorrelation(hmm, 8), atol=1e-10)
    np.testing.assert_allclose(power_spectrum(machine, OMEGAS), power_spectrum(hmm, OMEGAS), atol=1e-8)
