"""Tests for CM gap-analysis measures (Phases A–C)."""

from __future__ import annotations

import pytest

from sofic.examples import even_process, fair_coin, golden_mean
from sofic.generators.directional_flow import (
    directed_information,
    independent_pair_generator,
    transfer_entropy,
)
from sofic.generators.epsilon_machine import EpsilonMachine


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


def test_information_flow_measures_on_independent_pair():
    pytest.importorskip("dit")
    from sofic.generators.directional_flow import (
        intrinsic_information_flow,
        shared_information_flow,
        synergistic_information_flow,
    )

    pair = independent_pair_generator(fair_coin(), fair_coin())
    te = transfer_entropy(pair, history=1)
    intrinsic = intrinsic_information_flow(pair, history=1)
    shared = shared_information_flow(pair, history=1)
    synergistic = synergistic_information_flow(pair, history=1)

    assert intrinsic == pytest.approx(0.0, abs=1e-6)
    assert shared == pytest.approx(0.0, abs=1e-6)
    assert synergistic == pytest.approx(0.0, abs=1e-6)
    # Synergistic flow is defined as TE - intrinsic; the identity must hold exactly.
    assert synergistic + intrinsic == pytest.approx(te, abs=1e-6)


def test_predictability_gain_transient_and_converged():
    pytest.importorskip("dit")
    eps = golden_mean(0.5)
    estimates = eps.block_entropy_estimates(4)
    h_mu = estimates.entropy_rate
    # PG(1) = h_mu(1) - h_mu = (H[1] - H[0]) - h_mu, the largest, most informative gain.
    expected_pg1 = (estimates.block_entropy[1] - estimates.block_entropy[0]) - h_mu
    assert eps.predictability_gain(1) == pytest.approx(expected_pg1, abs=1e-9)
    assert eps.predictability_gain(1) > 1e-6
    # Beyond the Markov order the finite-block rate has converged, so PG -> 0.
    assert eps.predictability_gain(4) == pytest.approx(0.0, abs=1e-9)


def test_time_reversed_causal_irreversibility_sign():
    pytest.importorskip("dit")
    forward = golden_mean(0.5)
    reverse = EpsilonMachine.from_time_reversed(forward)
    delta_forward = forward.causal_irreversibility()
    delta_reverse = reverse.causal_irreversibility()
    assert delta_forward == pytest.approx(-delta_reverse, abs=1e-6)
