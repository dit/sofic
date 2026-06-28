"""Tests for CM gap-analysis measures (Phases A–C)."""

from __future__ import annotations

import pytest

from pensive.examples import even_process, fair_coin, golden_mean
from pensive.generators.directional_flow import (
    directed_information,
    independent_pair_generator,
    transfer_entropy,
)
from pensive.generators.epsilon_machine import EpsilonMachine

pytestmark = pytest.mark.measures


def test_causal_irreversibility_even_process_is_zero():
    pytest.importorskip("dit")
    eps = even_process(0.5)
    assert eps.causal_irreversibility() == pytest.approx(0.0, abs=1e-9)


def test_stored_information_decomposition_keys():
    pytest.importorskip("dit")
    eps = golden_mean(0.5)
    stored = eps.stored_information_decomposition()
    assert stored["causal_irreversibility"] == pytest.approx(
        stored["forward_complexity"] - stored["reverse_complexity"],
        abs=1e-9,
    )
    assert stored["bidirectional_complexity"] >= stored["excess_entropy"] - 1e-9


def test_block_convergence_scalars_positive_length():
    pytest.importorskip("dit")
    eps = golden_mean(0.5)
    assert eps.transient_information(4) >= 0.0
    assert eps.oracular_information(4) <= eps.statistical_complexity() + 1e-6
    assert eps.gauge_information(4) <= eps.statistical_complexity() + 1e-6
    assert eps.predictability_gain(4) == pytest.approx(0.0, abs=1e-9)


def test_structural_information_equals_excess_entropy():
    pytest.importorskip("dit")
    eps = golden_mean(0.5)
    assert eps.structural_information() == pytest.approx(eps.excess_entropy(), abs=1e-9)


def test_thermodynamic_depth_nonnegative():
    pytest.importorskip("dit")
    eps = golden_mean(0.5)
    assert eps.thermodynamic_depth() >= 0.0


def test_spectral_complexity_nonnegative():
    pytest.importorskip("dit")
    eps = golden_mean(0.5)
    assert eps.spectral_complexity() >= 0.0


def test_independent_pair_has_zero_transfer_entropy():
    pytest.importorskip("dit")
    pair = independent_pair_generator(fair_coin(), fair_coin())
    te = transfer_entropy(pair, history=1)
    assert te == pytest.approx(0.0, abs=1e-9)
    di = directed_information(pair, length=2)
    assert di == pytest.approx(0.0, abs=1e-9)


def test_time_reversed_causal_irreversibility_sign():
    pytest.importorskip("dit")
    forward = golden_mean(0.5)
    reverse = EpsilonMachine.from_time_reversed(forward)
    delta_forward = forward.causal_irreversibility()
    delta_reverse = reverse.causal_irreversibility()
    assert delta_forward == pytest.approx(-delta_reverse, abs=1e-6)
