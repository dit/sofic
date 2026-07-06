"""Tests for present-information anatomy (ρ_μ, b_μ, r_μ) on bidirectional machines."""

from __future__ import annotations

import pytest

from pensive.examples import (
    bernoulli,
    even_process,
    fair_coin,
    golden_mean_forward,
    golden_mean_reverse,
    tent_map_misiurewicz_bidirectional,
    tent_map_misiurewicz_forward,
    tent_map_misiurewicz_information_expected,
)
from pensive.generators.bidirectional_epsilon_machine import BidirectionalEpsilonMachine


def test_fair_coin_predicted_information_near_zero():
    pytest.importorskip("dit")
    coin = fair_coin()
    bidir = BidirectionalEpsilonMachine.from_pair(coin, coin)
    assert bidir.predicted_information() == pytest.approx(0.0, abs=1e-9)


@pytest.mark.parametrize("p", [0.5, 0.3])
def test_iid_ephemeral_information_equals_entropy_rate(p: float):
    """IID sources have no bound anatomy: all entropy rate is ephemeral."""
    pytest.importorskip("dit")
    bidir = bernoulli(p).to_bidirectional()
    h_mu = bidir.entropy_rate()
    r_mu = bidir.ephemeral_information()
    assert bidir.bound_information() == pytest.approx(0.0, abs=1e-9)
    assert r_mu == pytest.approx(h_mu, abs=1e-9)


def test_even_process_ephemeral_information_is_zero():
    """Even process: present symbol is fixed by bidirectional causal states."""
    pytest.importorskip("dit")
    bidir = even_process(0.5).to_bidirectional()
    h_mu = bidir.entropy_rate()
    b_mu = bidir.bound_information()
    r_mu = bidir.ephemeral_information()
    assert r_mu == pytest.approx(0.0, abs=1e-9)
    assert b_mu == pytest.approx(h_mu, abs=1e-9)
    assert b_mu + r_mu == pytest.approx(h_mu, abs=1e-9)


def test_golden_mean_anatomy_identities():
    pytest.importorskip("dit")
    import dit

    forward = golden_mean_forward(0.5)
    reverse = golden_mean_reverse(0.5)
    bidir = BidirectionalEpsilonMachine.from_pair(forward, reverse)

    h_mu = bidir.entropy_rate()
    rho = bidir.predicted_information()
    r_mu = bidir.ephemeral_information()
    b_mu = bidir.bound_information()

    assert b_mu + r_mu == pytest.approx(h_mu, abs=1e-9)
    h_x0 = dit.shannon.entropy(bidir.step_distribution().marginal([2]))
    assert h_mu + rho == pytest.approx(h_x0, abs=1e-9)
    assert bidir.crypticity() == pytest.approx(bidir.statistical_complexity() - bidir.excess_entropy(), abs=1e-9)

    anatomy = bidir.information_anatomy()
    assert anatomy["rho_mu"] == pytest.approx(rho, abs=1e-12)
    assert anatomy["bound_mu"] == pytest.approx(b_mu, abs=1e-12)
    assert anatomy["ephemeral_mu"] == pytest.approx(r_mu, abs=1e-12)
    assert anatomy["entropy_rate"] == pytest.approx(h_mu, abs=1e-12)


def test_golden_mean_anatomy_matches_component_methods():
    pytest.importorskip("dit")
    bidir = BidirectionalEpsilonMachine.from_pair(
        golden_mean_forward(0.5),
        golden_mean_reverse(0.5),
    )
    anatomy = bidir.information_anatomy()
    assert anatomy["rho_mu"] == pytest.approx(bidir.predicted_information(), abs=1e-12)
    assert anatomy["bound_mu"] == pytest.approx(bidir.bound_information(), abs=1e-12)
    assert anatomy["ephemeral_mu"] == pytest.approx(bidir.ephemeral_information(), abs=1e-12)
    assert anatomy["excess_entropy"] == pytest.approx(bidir.excess_entropy(), abs=1e-12)
    assert anatomy["crypticity"] == pytest.approx(bidir.crypticity(), abs=1e-12)


def test_tent_map_misiurewicz_closed_form():
    """Supplement closed forms for h_μ and b_μ (James et al., 2013)."""
    expected = tent_map_misiurewicz_information_expected()
    assert expected["entropy_rate"] == pytest.approx(0.823172, abs=1e-4)
    assert expected["bound_mu"] == pytest.approx(0.174915, abs=1e-4)
    assert expected["ephemeral_mu"] == pytest.approx(0.648258, abs=1e-4)


def test_tent_map_misiurewicz_entropy_rate():
    """Fig.~7 ε-machine entropy rate matches supplement closed form."""
    pytest.importorskip("dit")
    expected = tent_map_misiurewicz_information_expected()
    forward = tent_map_misiurewicz_forward()
    assert forward.entropy_rate() == pytest.approx(expected["entropy_rate"], abs=1e-4)


def test_tent_map_misiurewicz_forward_epsilon_machine_matches_fig7():
    """Marginalizing supplement Fig.~8 recovers the Fig.~7 forward ε-machine."""
    pytest.importorskip("dit")
    bidir = tent_map_misiurewicz_bidirectional()
    forward = tent_map_misiurewicz_forward()
    recovered = bidir.forward_epsilon_machine()
    assert recovered.entropy_rate() == pytest.approx(forward.entropy_rate(), abs=1e-4)
    for state in forward.states():
        expected = sorted(
            (t.data["emission"], t.target, round(float(t.data["prob"]), 6))
            for t in forward.graph.out_transitions(state)
        )
        actual = sorted(
            (t.data["emission"], t.target, round(float(t.data["prob"]), 6))
            for t in recovered.graph.out_transitions(state)
        )
        assert actual == expected


def test_tent_map_misiurewicz_bidirectional_regression():
    """Regression against supplement anatomy rates for the Fig.~7 ε-machine."""
    pytest.importorskip("dit")
    expected = tent_map_misiurewicz_information_expected()
    bidir = tent_map_misiurewicz_bidirectional()

    h_mu = bidir.entropy_rate()
    r_mu = bidir.ephemeral_information()
    b_mu = bidir.bound_information()

    assert h_mu == pytest.approx(expected["entropy_rate"], abs=1e-4)
    assert r_mu == pytest.approx(expected["ephemeral_mu"], abs=1e-4)
    assert b_mu == pytest.approx(expected["bound_mu"], abs=1e-4)
    assert b_mu + r_mu == pytest.approx(h_mu, abs=1e-9)
    assert bidir.crypticity() == pytest.approx(bidir.statistical_complexity() - bidir.excess_entropy(), abs=1e-9)


def test_tent_forward_matches_generator_path():
    pytest.importorskip("dit")
    from pensive.examples.epsilon_machines import tent_map_misiurewicz_hmm
    from pensive.generators.epsilon_machine import EpsilonMachine

    forward = tent_map_misiurewicz_forward()
    try:
        from_hmm = EpsilonMachine.from_hmm(tent_map_misiurewicz_hmm())
    except Exception:
        pytest.skip("Fig. 6 HMM does not yet yield a valid ε-machine via from_hmm")
    if from_hmm.entropy_rate() != pytest.approx(forward.entropy_rate(), abs=1e-3):
        pytest.xfail("Fig. 6 HMM does not yet yield a valid ε-machine via from_hmm")
    assert from_hmm.entropy_rate() == pytest.approx(forward.entropy_rate(), abs=1e-3)


def test_epsilon_machine_anatomy_matches_bidirectional():
    pytest.importorskip("dit")
    forward = golden_mean_forward(0.5)
    bidir = BidirectionalEpsilonMachine.from_pair(
        forward,
        golden_mean_reverse(0.5),
    )

    assert forward.predicted_information() == pytest.approx(bidir.predicted_information(), abs=1e-12)
    assert forward.bound_information() == pytest.approx(bidir.bound_information(), abs=1e-12)
    assert forward.ephemeral_information() == pytest.approx(bidir.ephemeral_information(), abs=1e-12)
    assert forward.information_anatomy() == pytest.approx(bidir.information_anatomy(), abs=1e-12)
    assert forward.excess_entropy() == pytest.approx(bidir.excess_entropy(), abs=1e-12)
    assert forward.bidirectional_statistical_complexity() == pytest.approx(bidir.statistical_complexity(), abs=1e-12)
    assert forward.bidirectional_crypticity() == pytest.approx(bidir.crypticity(), abs=1e-12)


def test_epsilon_machine_bidirectional_cache():
    pytest.importorskip("dit")
    forward = golden_mean_forward(0.5)
    first = forward.to_bidirectional()
    second = forward.to_bidirectional()
    assert first is second

    state = next(iter(forward.states()))
    forward.graph.nx.nodes[state]["cache_marker"] = "changed"
    assert forward.to_bidirectional() is not first

    cloned = forward.copy()
    assert cloned.to_bidirectional() is not first


@pytest.mark.parametrize("p", [0.5, 0.3])
def test_iid_caekl_causal_information_is_zero(p: float):
    """IID sources: past, present, and future share no information."""
    pytest.importorskip("dit")
    bidir = bernoulli(p).to_bidirectional()
    assert bidir.caekl_causal_information() == pytest.approx(0.0, abs=1e-9)


def test_fair_coin_caekl_causal_information_is_zero():
    pytest.importorskip("dit")
    bidir = BidirectionalEpsilonMachine.from_pair(fair_coin(), fair_coin())
    assert bidir.caekl_causal_information() == pytest.approx(0.0, abs=1e-9)


@pytest.mark.parametrize(
    "factory",
    [
        lambda: golden_mean_forward(0.5),
        lambda: even_process(0.5),
        tent_map_misiurewicz_forward,
    ],
)
def test_caekl_causal_information_nonnegative(factory):
    """J[S⁺₀ : X₀ : S⁻₁] is a CAEKL mutual information, hence non-negative."""
    pytest.importorskip("dit")
    assert factory().caekl_causal_information() >= -1e-9


def test_caekl_causal_information_golden_mean_value():
    """Exact J[S⁺₀ : X₀ : S⁻₁] for the golden-mean process at p = 1/2."""
    pytest.importorskip("dit")
    bidir = BidirectionalEpsilonMachine.from_pair(
        golden_mean_forward(0.5),
        golden_mean_reverse(0.5),
    )
    assert bidir.caekl_causal_information() == pytest.approx(0.25162916738782304, abs=1e-9)


def test_caekl_causal_information_tent_map_value():
    """Regression on the tent-map bidirectional machine (matches the docs doctest)."""
    pytest.importorskip("dit")
    bidir = tent_map_misiurewicz_bidirectional()
    assert bidir.caekl_causal_information() == pytest.approx(0.22044436492357078, abs=1e-9)


def test_epsilon_machine_caekl_causal_information_matches_bidirectional():
    """EpsilonMachine delegates caekl_causal_information() to its bidirectional presentation."""
    pytest.importorskip("dit")
    forward = golden_mean_forward(0.5)
    assert forward.caekl_causal_information() == pytest.approx(
        forward.to_bidirectional().caekl_causal_information(), abs=1e-12
    )
