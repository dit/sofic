"""Tests for present-information anatomy (ρ_μ, b_μ, r_μ) on bidirectional machines."""

from __future__ import annotations

import math

import pytest

from sofic.examples import (
    NRPS,
    bernoulli,
    butterfly_process,
    even_process,
    fair_coin,
    golden_mean_forward,
    golden_mean_reverse,
    nemo_process,
    tent_map_misiurewicz_a,
    tent_map_misiurewicz_bidirectional,
    tent_map_misiurewicz_forward,
    tent_map_misiurewicz_information_expected,
    tent_map_misiurewicz_preimage_forward,
    tent_map_misiurewicz_preimage_information_expected,
)
from sofic.generators.bidirectional_epsilon_machine import BidirectionalEpsilonMachine


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


# --- structural / gauge refinement of the anatomy ------------------------------


def _anatomy_processes():
    """Bidirectional presentations with a valid ``b_μ + r_μ = h_μ`` anatomy."""
    return {
        "bernoulli_half": bernoulli(0.5).to_bidirectional(),
        "bernoulli_biased": bernoulli(0.3).to_bidirectional(),
        "golden_mean": BidirectionalEpsilonMachine.from_pair(golden_mean_forward(0.5), golden_mean_reverse(0.5)),
        "even": even_process(0.5).to_bidirectional(),
        "butterfly": butterfly_process().to_bidirectional(),
    }


@pytest.mark.parametrize("name", ["bernoulli_half", "bernoulli_biased", "golden_mean", "even", "butterfly"])
def test_structural_gauge_split_sums_to_parents(name: str):
    """r_μ = r_μ^struct + r_μ^par and b_μ = b_μ^struct + b_μ^par, all non-negative."""
    pytest.importorskip("dit")
    bidir = _anatomy_processes()[name]

    r_mu = bidir.ephemeral_information()
    r_struct = bidir.structural_ephemeral_information()
    r_gauge = bidir.parallel_edge_information()
    b_mu = bidir.bound_information()
    b_struct = bidir.bound_structural_information()
    b_gauge = bidir.bound_parallel_edge_information()

    for atom in (r_struct, r_gauge, b_struct, b_gauge):
        assert atom >= -1e-9
    assert r_struct + r_gauge == pytest.approx(r_mu, abs=1e-9)
    assert b_struct + b_gauge == pytest.approx(b_mu, abs=1e-9)
    # The four atoms refine the entropy rate.
    assert r_struct + r_gauge + b_struct + b_gauge == pytest.approx(bidir.entropy_rate(), abs=1e-9)


def test_bernoulli_ephemeral_is_pure_gauge():
    """IID source: every ephemeral bit is parallel-edge (gauge); none is structural."""
    pytest.importorskip("dit")
    bidir = bernoulli(0.3).to_bidirectional()
    h_mu = bidir.entropy_rate()
    assert bidir.structural_ephemeral_information() == pytest.approx(0.0, abs=1e-9)
    assert bidir.parallel_edge_information() == pytest.approx(bidir.ephemeral_information(), abs=1e-9)
    assert bidir.parallel_edge_information() == pytest.approx(h_mu, abs=1e-9)


def test_golden_mean_ephemeral_is_pure_structural():
    """Golden mean has no parallel edges: r_μ is entirely structural."""
    pytest.importorskip("dit")
    bidir = BidirectionalEpsilonMachine.from_pair(golden_mean_forward(0.5), golden_mean_reverse(0.5))
    assert bidir.parallel_edge_information() == pytest.approx(0.0, abs=1e-9)
    assert bidir.structural_ephemeral_information() == pytest.approx(bidir.ephemeral_information(), abs=1e-9)
    # Its bound information is likewise entirely structural.
    assert bidir.bound_parallel_edge_information() == pytest.approx(0.0, abs=1e-9)
    assert bidir.bound_structural_information() == pytest.approx(bidir.bound_information(), abs=1e-9)


def test_butterfly_ephemeral_is_mixed():
    """Butterfly process populates both ephemeral atoms simultaneously."""
    pytest.importorskip("dit")
    bidir = butterfly_process().to_bidirectional()
    r_struct = bidir.structural_ephemeral_information()
    r_gauge = bidir.parallel_edge_information()
    assert r_struct > 1e-6
    assert r_gauge > 1e-6
    assert r_struct + r_gauge == pytest.approx(bidir.ephemeral_information(), abs=1e-9)


def test_information_anatomy_exposes_refinement_keys():
    """information_anatomy() carries the structural/gauge atoms, matching the methods."""
    pytest.importorskip("dit")
    bidir = butterfly_process().to_bidirectional()
    anatomy = bidir.information_anatomy()
    assert anatomy["ephemeral_structural"] == pytest.approx(bidir.structural_ephemeral_information(), abs=1e-12)
    assert anatomy["ephemeral_gauge"] == pytest.approx(bidir.parallel_edge_information(), abs=1e-12)
    assert anatomy["bound_structural"] == pytest.approx(bidir.bound_structural_information(), abs=1e-12)
    assert anatomy["bound_gauge"] == pytest.approx(bidir.bound_parallel_edge_information(), abs=1e-12)
    assert anatomy["ephemeral_structural"] + anatomy["ephemeral_gauge"] == pytest.approx(
        anatomy["ephemeral_mu"], abs=1e-9
    )
    assert anatomy["bound_structural"] + anatomy["bound_gauge"] == pytest.approx(anatomy["bound_mu"], abs=1e-9)


def test_epsilon_machine_refinement_delegates_to_bidirectional():
    """EpsilonMachine forwards the structural/gauge accessors to its bidirectional presentation."""
    pytest.importorskip("dit")
    forward = golden_mean_forward(0.5)
    bidir = forward.to_bidirectional()
    assert forward.structural_ephemeral_information() == pytest.approx(
        bidir.structural_ephemeral_information(), abs=1e-12
    )
    assert forward.parallel_edge_information() == pytest.approx(bidir.parallel_edge_information(), abs=1e-12)
    assert forward.bound_structural_information() == pytest.approx(bidir.bound_structural_information(), abs=1e-12)
    assert forward.bound_parallel_edge_information() == pytest.approx(
        bidir.bound_parallel_edge_information(), abs=1e-12
    )


# --- five-variable anatomy: four-atom ephemeral partition + reverse bound mirror ---


def _five_variable_processes():
    """Bidirectional presentations spanning the ephemeral-motif zoo."""
    return {
        "bernoulli_half": bernoulli(0.5).to_bidirectional(),
        "bernoulli_biased": bernoulli(0.3).to_bidirectional(),
        "golden_mean": BidirectionalEpsilonMachine.from_pair(golden_mean_forward(0.5), golden_mean_reverse(0.5)),
        "even": even_process(0.5).to_bidirectional(),
        "butterfly": butterfly_process().to_bidirectional(),
        "nemo": nemo_process().to_bidirectional(),
        "nrps": NRPS().to_bidirectional(),
        "tent": tent_map_misiurewicz_bidirectional(),
    }


_FIVE_VARIABLE_NAMES = [
    "bernoulli_half",
    "bernoulli_biased",
    "golden_mean",
    "even",
    "butterfly",
    "nemo",
    "nrps",
    "tent",
]


@pytest.mark.parametrize("name", _FIVE_VARIABLE_NAMES)
def test_ephemeral_four_atom_partition_sums_to_r_mu(name: str):
    """r_μ = r_μ^fwd + r_μ^rev + r_μ^joint + r_μ^gauge, all non-negative."""
    pytest.importorskip("dit")
    bidir = _five_variable_processes()[name]

    r_fwd = bidir.forward_only_structural_ephemeral()
    r_rev = bidir.reverse_only_structural_ephemeral()
    r_joint = bidir.joint_structural_ephemeral()
    r_gauge = bidir.pure_gauge_information()

    for atom in (r_fwd, r_rev, r_joint, r_gauge):
        assert atom >= -1e-9
    assert r_fwd + r_rev + r_joint + r_gauge == pytest.approx(bidir.ephemeral_information(), abs=1e-9)


@pytest.mark.parametrize("name", _FIVE_VARIABLE_NAMES)
def test_four_atoms_refine_coarse_structural_gauge(name: str):
    """The four atoms regroup into the coarse structural/gauge and reverse splits."""
    pytest.importorskip("dit")
    bidir = _five_variable_processes()[name]

    r_fwd = bidir.forward_only_structural_ephemeral()
    r_rev = bidir.reverse_only_structural_ephemeral()
    r_joint = bidir.joint_structural_ephemeral()
    r_gauge = bidir.pure_gauge_information()

    # r_μ^struct = r_fwd + r_joint ; r_μ^par = r_rev + r_gauge.
    assert r_fwd + r_joint == pytest.approx(bidir.structural_ephemeral_information(), abs=1e-9)
    assert r_rev + r_gauge == pytest.approx(bidir.parallel_edge_information(), abs=1e-9)
    # Reverse structural ephemeral r̄_μ^struct = r_rev + r_joint (time-reversed mirror).
    assert r_rev + r_joint == pytest.approx(bidir.reverse_structural_ephemeral_information(), abs=1e-9)


@pytest.mark.parametrize("name", _FIVE_VARIABLE_NAMES)
def test_theorem_a_prime_reverse_bound_gauge_vanishes(name: str):
    """Theorem A′: I[X₀ : S⁺₀ | S⁻₁, S⁻₀] = 0 — the reverse bound has no gauge part."""
    pytest.importorskip("dit")
    bidir = _five_variable_processes()[name]
    assert bidir.reverse_bound_gauge_information() == pytest.approx(0.0, abs=1e-9)


@pytest.mark.parametrize("name", _FIVE_VARIABLE_NAMES)
def test_bound_information_is_time_reversal_symmetric(name: str):
    """b̄_μ = b_μ, and by Theorem A′ the reverse bound is entirely structural."""
    pytest.importorskip("dit")
    bidir = _five_variable_processes()[name]
    b_mu = bidir.bound_information()
    assert bidir.reverse_bound_information() == pytest.approx(b_mu, abs=1e-9)
    assert bidir.reverse_bound_structural_information() == pytest.approx(b_mu, abs=1e-9)


@pytest.mark.parametrize("name", _FIVE_VARIABLE_NAMES)
def test_internal_markov_entropy_rate_anatomy_identities(name: str):
    """h_μ^imc = H[S⁺₁|S⁺₀] = b_μ + r_fwd + r_joint = h_μ − r_rev − r_gauge."""
    pytest.importorskip("dit")
    import dit

    bidir = _five_variable_processes()[name]
    dist = bidir.step_distribution()

    h_fwd = bidir.internal_markov_entropy_rate()
    h_rev = bidir.reverse_internal_markov_entropy_rate()
    b_mu = bidir.bound_information()
    r_fwd = bidir.forward_only_structural_ephemeral()
    r_rev = bidir.reverse_only_structural_ephemeral()
    r_joint = bidir.joint_structural_ephemeral()
    r_gauge = bidir.pure_gauge_information()
    h_mu = bidir.entropy_rate()

    # Direct definition and the two anatomy decompositions.
    assert h_fwd == pytest.approx(float(dit.shannon.conditional_entropy(dist, [3], [0])), abs=1e-9)
    assert h_rev == pytest.approx(float(dit.shannon.conditional_entropy(dist, [1], [4])), abs=1e-9)
    assert h_fwd == pytest.approx(b_mu + r_fwd + r_joint, abs=1e-9)
    assert h_rev == pytest.approx(b_mu + r_rev + r_joint, abs=1e-9)
    assert h_mu - h_fwd == pytest.approx(r_rev + r_gauge, abs=1e-9)
    assert h_mu - h_rev == pytest.approx(r_fwd + r_gauge, abs=1e-9)
    # The forward/reverse gap is exactly the arrow-of-time asymmetry.
    assert h_fwd - h_rev == pytest.approx(r_fwd - r_rev, abs=1e-9)


@pytest.mark.parametrize(
    "name,symmetric",
    [
        ("golden_mean", True),
        ("even", True),
        ("nemo", True),
        ("nrps", False),
        ("butterfly", False),
    ],
)
def test_internal_markov_rate_arrow_of_time(name: str, symmetric: bool):
    """The forward/reverse internal rates coincide iff r_fwd = r_rev."""
    pytest.importorskip("dit")
    bidir = _five_variable_processes()[name]
    h_fwd = bidir.internal_markov_entropy_rate()
    h_rev = bidir.reverse_internal_markov_entropy_rate()
    if symmetric:
        assert h_fwd == pytest.approx(h_rev, abs=1e-9)
    else:
        assert abs(h_fwd - h_rev) > 1e-6


def test_golden_mean_ephemeral_is_pure_joint():
    """Golden mean: the single branch is resolved by *both* time directions (r_joint)."""
    pytest.importorskip("dit")
    bidir = BidirectionalEpsilonMachine.from_pair(golden_mean_forward(0.5), golden_mean_reverse(0.5))
    r_mu = bidir.ephemeral_information()
    assert bidir.joint_structural_ephemeral() == pytest.approx(r_mu, abs=1e-9)
    assert bidir.joint_structural_ephemeral() == pytest.approx(0.459147917, abs=1e-6)
    assert bidir.forward_only_structural_ephemeral() == pytest.approx(0.0, abs=1e-9)
    assert bidir.reverse_only_structural_ephemeral() == pytest.approx(0.0, abs=1e-9)
    assert bidir.pure_gauge_information() == pytest.approx(0.0, abs=1e-9)


def test_nrps_ephemeral_is_pure_reverse_arrow_of_time():
    """NRPS: forward-only ephemeral vanishes while reverse-only does not — an arrow of time."""
    pytest.importorskip("dit")
    bidir = NRPS().to_bidirectional()
    r_mu = bidir.ephemeral_information()
    assert bidir.forward_only_structural_ephemeral() == pytest.approx(0.0, abs=1e-9)
    assert bidir.reverse_only_structural_ephemeral() == pytest.approx(r_mu, abs=1e-9)
    assert bidir.reverse_only_structural_ephemeral() == pytest.approx(1.0 / 6.0, abs=1e-9)
    # Forward and reverse structural ephemeral rates disagree: r_μ^struct = 0 ≠ r̄_μ^struct.
    assert bidir.structural_ephemeral_information() == pytest.approx(0.0, abs=1e-9)
    assert bidir.reverse_structural_ephemeral_information() == pytest.approx(1.0 / 6.0, abs=1e-9)


def test_nemo_four_atoms_pinned():
    """Nemo populates the forward, reverse, and gauge atoms but not the joint one."""
    pytest.importorskip("dit")
    bidir = nemo_process().to_bidirectional()
    assert bidir.forward_only_structural_ephemeral() == pytest.approx(1.0 / 6.0, abs=1e-9)
    assert bidir.reverse_only_structural_ephemeral() == pytest.approx(1.0 / 6.0, abs=1e-9)
    assert bidir.joint_structural_ephemeral() == pytest.approx(0.0, abs=1e-9)
    assert bidir.pure_gauge_information() == pytest.approx(1.0 / 12.0, abs=1e-9)


def test_butterfly_ephemeral_is_forward_plus_gauge():
    """Butterfly: forward-branching plus parallel-edge relabeling, no reverse/joint atom."""
    pytest.importorskip("dit")
    bidir = butterfly_process().to_bidirectional()
    assert bidir.forward_only_structural_ephemeral() == pytest.approx(2.25, abs=1e-9)
    assert bidir.pure_gauge_information() == pytest.approx(0.75, abs=1e-9)
    assert bidir.reverse_only_structural_ephemeral() == pytest.approx(0.0, abs=1e-9)
    assert bidir.joint_structural_ephemeral() == pytest.approx(0.0, abs=1e-9)


def test_fair_coin_ephemeral_is_pure_gauge_atom():
    """Fair coin: every ephemeral bit is the transient pure-gauge atom."""
    pytest.importorskip("dit")
    bidir = BidirectionalEpsilonMachine.from_pair(fair_coin(), fair_coin())
    assert bidir.pure_gauge_information() == pytest.approx(bidir.ephemeral_information(), abs=1e-9)
    assert bidir.pure_gauge_information() == pytest.approx(1.0, abs=1e-9)
    assert bidir.forward_only_structural_ephemeral() == pytest.approx(0.0, abs=1e-9)
    assert bidir.reverse_only_structural_ephemeral() == pytest.approx(0.0, abs=1e-9)
    assert bidir.joint_structural_ephemeral() == pytest.approx(0.0, abs=1e-9)


def test_five_variable_anatomy_exposes_atoms_matching_methods():
    """five_variable_anatomy() carries the four ephemeral atoms + reverse bound mirror."""
    pytest.importorskip("dit")
    bidir = nemo_process().to_bidirectional()
    anatomy = bidir.five_variable_anatomy()

    assert anatomy["ephemeral_forward"] == pytest.approx(bidir.forward_only_structural_ephemeral(), abs=1e-12)
    assert anatomy["ephemeral_reverse"] == pytest.approx(bidir.reverse_only_structural_ephemeral(), abs=1e-12)
    assert anatomy["ephemeral_joint"] == pytest.approx(bidir.joint_structural_ephemeral(), abs=1e-12)
    assert anatomy["ephemeral_pure_gauge"] == pytest.approx(bidir.pure_gauge_information(), abs=1e-12)
    assert anatomy["ephemeral_structural_reverse"] == pytest.approx(
        bidir.reverse_structural_ephemeral_information(), abs=1e-12
    )
    assert anatomy["bound_reverse"] == pytest.approx(bidir.reverse_bound_information(), abs=1e-12)
    assert anatomy["bound_structural_reverse"] == pytest.approx(bidir.reverse_bound_structural_information(), abs=1e-12)
    assert anatomy["bound_gauge_reverse"] == pytest.approx(bidir.reverse_bound_gauge_information(), abs=1e-12)

    # The four atoms sum to r_μ, and the coarse anatomy keys are still present.
    four = (
        anatomy["ephemeral_forward"]
        + anatomy["ephemeral_reverse"]
        + anatomy["ephemeral_joint"]
        + anatomy["ephemeral_pure_gauge"]
    )
    assert four == pytest.approx(anatomy["ephemeral_mu"], abs=1e-9)
    assert anatomy["ephemeral_structural"] == pytest.approx(bidir.structural_ephemeral_information(), abs=1e-12)


def test_epsilon_machine_five_variable_delegates_to_bidirectional():
    """EpsilonMachine forwards the five-variable accessors to its bidirectional presentation."""
    pytest.importorskip("dit")
    forward = golden_mean_forward(0.5)
    bidir = forward.to_bidirectional()

    assert forward.forward_only_structural_ephemeral() == pytest.approx(
        bidir.forward_only_structural_ephemeral(), abs=1e-12
    )
    assert forward.reverse_only_structural_ephemeral() == pytest.approx(
        bidir.reverse_only_structural_ephemeral(), abs=1e-12
    )
    assert forward.joint_structural_ephemeral() == pytest.approx(bidir.joint_structural_ephemeral(), abs=1e-12)
    assert forward.pure_gauge_information() == pytest.approx(bidir.pure_gauge_information(), abs=1e-12)
    assert forward.reverse_structural_ephemeral_information() == pytest.approx(
        bidir.reverse_structural_ephemeral_information(), abs=1e-12
    )
    assert forward.reverse_bound_information() == pytest.approx(bidir.reverse_bound_information(), abs=1e-12)
    assert forward.reverse_bound_structural_information() == pytest.approx(
        bidir.reverse_bound_structural_information(), abs=1e-12
    )
    assert forward.reverse_bound_gauge_information() == pytest.approx(
        bidir.reverse_bound_gauge_information(), abs=1e-12
    )
    assert forward.five_variable_anatomy() == pytest.approx(bidir.five_variable_anatomy(), abs=1e-12)


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


def test_tent_map_preimage_partition_presentation():
    """Refining by both preimages of ``c`` gives five states over four letters."""
    forward = tent_map_misiurewicz_preimage_forward()
    forward.validate_stochastic()

    assert sorted(forward.states()) == ["A", "B", "C", "D", "E"]
    assert sorted(forward.observation_alphabet) == [0, 1, 2, 3]
    assert forward.is_unifilar()
    # Still strictly sofic: r_μ = 0 does not buy finite memory.
    assert forward.is_strictly_sofic()

    # Reduced by the parameter's minimal polynomial ``a**3 = 2a + 2``, every
    # branching probability is a quadratic in ``a``.
    a = tent_map_misiurewicz_a()
    expected = {
        ("A", 2, "A"): (a**2 - 2) / 2,
        ("A", 3, "B"): (4 - a**2) / 2,
        ("B", 0, "D"): (a**2 - 2 * a + 2) / 6,
        ("B", 1, "A"): (4 + 2 * a - a**2) / 6,
        ("C", 2, "E"): (2 + a - a**2) / 2,
        ("C", 3, "B"): (a**2 - a) / 2,
        ("D", 1, "C"): 1.0,
        ("E", 2, "C"): 1.0,
    }
    edges = {(t.source, t.data["emission"], t.target): float(t.data["prob"]) for t in forward.graph.transitions()}
    assert edges.keys() == expected.keys()
    for edge, probability in expected.items():
        assert edges[edge] == pytest.approx(probability, abs=1e-12)


def test_tent_map_preimage_partition_stationary_distribution():
    """Causal-state weights of the four-letter tent-map presentation."""
    forward = tent_map_misiurewicz_preimage_forward()
    index = forward.reindex()
    pi = forward.stationary_distribution()
    weights = {index.state(i): float(pi[i]) for i in range(len(index.states))}

    assert weights == pytest.approx(
        {
            "A": 0.4870384416,
            "B": 0.2882345739,
            "C": 0.1123634923,
            "D": 0.0764691477,
            "E": 0.0358943445,
        },
        abs=1e-9,
    )


def test_tent_map_preimage_partition_forbidden_blocks():
    """Nine of the sixteen two-letter blocks are forbidden by the refined partition."""
    forward = tent_map_misiurewicz_preimage_forward()
    words = forward.word_probabilities(2)
    forbidden = {(0, 0), (0, 2), (0, 3), (1, 0), (1, 1), (2, 0), (2, 1), (3, 2), (3, 3)}

    for word in forbidden:
        assert float(words.get(word, 0.0)) == pytest.approx(0.0, abs=1e-12)
    allowed = {word for word, p in words.items() if float(p) > 1e-12}
    assert len(allowed) == 7
    assert allowed.isdisjoint(forbidden)


def test_tent_map_preimage_partition_is_generating():
    """Both partitions of the same dynamics share the entropy rate ``log2(a)``."""
    pytest.importorskip("dit")
    a = tent_map_misiurewicz_a()
    refined = tent_map_misiurewicz_preimage_forward()

    assert refined.entropy_rate() == pytest.approx(math.log2(a), abs=1e-9)
    assert refined.entropy_rate() == pytest.approx(tent_map_misiurewicz_forward().entropy_rate(), abs=1e-9)


def test_tent_map_preimage_ephemeral_information_vanishes():
    """The refined partition moves the whole entropy rate into bound information."""
    pytest.importorskip("dit")
    expected = tent_map_misiurewicz_preimage_information_expected()
    refined = tent_map_misiurewicz_preimage_forward()

    assert refined.ephemeral_information() == pytest.approx(0.0, abs=1e-9)
    assert refined.bound_information() == pytest.approx(expected["bound_mu"], abs=1e-9)
    assert refined.entropy_rate() == pytest.approx(expected["entropy_rate"], abs=1e-9)

    # The coarser kneading partition of the same dynamics does not.
    assert tent_map_misiurewicz_information_expected()["ephemeral_mu"] == pytest.approx(0.648258, abs=1e-4)


def test_tent_map_preimage_statistical_complexity_exceeds_excess_entropy():
    """Regression on C_μ and E; the gap is the machine's crypticity."""
    pytest.importorskip("dit")
    refined = tent_map_misiurewicz_preimage_forward()

    assert refined.statistical_complexity() == pytest.approx(1.833069440568067, abs=1e-9)
    assert refined.excess_entropy() == pytest.approx(1.1407151497907773, abs=1e-9)
    assert refined.statistical_complexity() > refined.excess_entropy()


def test_tent_forward_matches_generator_path():
    pytest.importorskip("dit")
    from sofic.examples.epsilon_machines import tent_map_misiurewicz_hmm
    from sofic.generators.epsilon_machine import EpsilonMachine

    forward = tent_map_misiurewicz_forward()
    from_hmm = EpsilonMachine.from_hmm(tent_map_misiurewicz_hmm())
    assert from_hmm.entropy_rate() == pytest.approx(forward.entropy_rate(), abs=1e-9)
    assert len(list(from_hmm.states())) == len(list(forward.states()))


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
