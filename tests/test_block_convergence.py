"""Tests for James et al. (2011) block convergence suite."""

from __future__ import annotations

import math

import pytest

from pensive.examples import even_process, fair_coin, golden_mean, noisy_random_phase_slip
from pensive.generators.block_convergence import block_caekl


def test_fair_coin_block_convergence_independent():
    diag = fair_coin().block_convergence_diagram(4)
    diag.validate_identities()
    assert diag.block_total_correlation[2:] == pytest.approx(0.0, abs=1e-12)
    assert diag.block_binding_information[2:] == pytest.approx(0.0, abs=1e-12)
    assert diag.block_caekl[2:] == pytest.approx(0.0, abs=1e-12)
    assert diag.entropy_rate == pytest.approx(1.0, abs=1e-12)
    assert diag.excess_entropy == pytest.approx(0.0, abs=1e-12)
    assert diag.bound_information == pytest.approx(0.0, abs=1e-12)
    assert diag.ephemeral_information == pytest.approx(1.0, abs=1e-12)
    assert diag.j_mu == pytest.approx(0.0, abs=1e-12)
    assert diag.caekl_rate_converged is True


def test_fair_coin_caekl_api():
    eps = fair_coin()
    assert eps.caekl_block_information(2) == pytest.approx(0.0, abs=1e-12)
    assert eps.caekl_rate(4) == pytest.approx(0.0, abs=1e-12)
    assert eps.caekl_rate_converged(4) is True


def test_block_caekl_computed_through_max_length():
    eps = golden_mean(0.5)
    diag = eps.block_convergence_diagram(6)
    assert float(diag.block_caekl[6]) > 0.0


def test_golden_mean_j_mu_stable_and_bounded():
    eps = golden_mean(0.5)
    rates = [eps.caekl_rate(length, max_caekl_length=6) for length in (4, 5, 6)]
    assert rates[0] == pytest.approx(rates[1], abs=1e-9)
    assert rates[1] == pytest.approx(rates[2], abs=1e-9)
    est = eps.block_convergence_estimates(6, max_caekl_length=6)
    assert est.j_mu == pytest.approx(0.0, abs=1e-9)
    assert est.j_mu <= est.b_mu + 1e-9
    assert est.j_mu <= est.rho_mu + 1e-9
    assert est.caekl_intercept_scalar == pytest.approx(0.25162916738782304, abs=1e-9)
    assert est.caekl_rate_converged is True


def test_golden_mean_table_i_scalars():
    est = golden_mean(0.5).block_convergence_estimates(max_length=8, max_caekl_length=6)
    est.validate_identities()
    assert est.block_entropy[1] == pytest.approx(0.91830, abs=1e-4)
    assert est.h_mu == pytest.approx(2.0 / 3.0, abs=1e-4)
    assert est.rho_mu == pytest.approx(0.25163, abs=1e-4)
    assert est.r_mu == pytest.approx(0.45915, abs=1e-4)
    assert est.b_mu == pytest.approx(0.20752, abs=1e-4)
    assert est.q_mu == pytest.approx(0.04411, abs=1e-4)
    assert est.E == pytest.approx(0.25163, abs=1e-4)


def test_even_process_table_i_scalars():
    eps = even_process(0.5)
    est = eps.block_convergence_estimates(max_length=8, max_caekl_length=6)
    est.validate_identities()
    assert est.h_mu == pytest.approx(2 / 3, abs=1e-4)
    assert est.b_mu == pytest.approx(2 / 3, abs=1e-4)
    assert est.r_mu == pytest.approx(0.0, abs=1e-4)
    assert est.q_mu == pytest.approx(est.rho_mu - est.b_mu, abs=1e-9)
    assert est.b_mu + est.r_mu == pytest.approx(est.h_mu, abs=1e-4)


def test_nrps_table_i_scalars():
    est = noisy_random_phase_slip().block_convergence_estimates(max_length=8, max_caekl_length=6)
    est.validate_identities()
    assert est.block_entropy[1] == pytest.approx(0.97987, abs=1e-4)
    assert est.h_mu == pytest.approx(0.5, abs=1e-4)
    assert est.rho_mu == pytest.approx(0.47987, abs=1e-4)
    assert est.E == pytest.approx(1.57393, abs=1e-4)
    assert est.r_mu + est.b_mu == pytest.approx(est.h_mu, abs=1e-4)
    assert est.b_mu == pytest.approx(0.33333, abs=1e-4)
    assert est.r_mu == pytest.approx(0.16667, abs=1e-4)


def test_golden_mean_caekl_ordering():
    diag = golden_mean(0.5).block_convergence_diagram(6)
    for length in range(2, len(diag.lengths)):
        j_l = float(diag.block_caekl[length])
        b_l = float(diag.block_binding_information[length])
        t_l = float(diag.block_total_correlation[length])
        assert j_l >= -1e-12
        assert j_l <= b_l + 1e-9
        assert j_l <= t_l + 1e-9
    assert diag.block_caekl[0] == pytest.approx(0.0, abs=1e-12)
    assert diag.block_caekl[1] == pytest.approx(0.0, abs=1e-12)
    assert diag.block_caekl[2] > 0.0


def test_golden_mean_coinformation_tends_to_zero():
    diag = golden_mean(0.5).block_convergence_diagram(8, max_caekl_length=6)
    assert abs(float(diag.block_coinformation[-1])) < 1e-4


def test_nrps_block_curves_qualitative():
    diag = noisy_random_phase_slip().block_convergence_diagram(8, max_caekl_length=6)
    assert diag.block_total_correlation[-1] > 0.0
    assert diag.h_mu != pytest.approx(diag.rho_mu, abs=1e-3)


def test_information_anatomy_keys():
    anatomy = golden_mean(0.5).block_convergence_estimates(6, max_caekl_length=6).information_anatomy()
    expected = {
        "rho_mu",
        "bound_mu",
        "ephemeral_mu",
        "entropy_rate",
        "excess_entropy",
        "crypticity",
        "q_mu",
        "w_mu",
        "E_B",
        "E_R",
        "E_Q",
        "E_W",
        "j_mu",
        "caekl_intercept",
        "caekl_rate_converged",
    }
    assert expected <= set(anatomy.keys())
    assert anatomy["bound_mu"] + anatomy["ephemeral_mu"] == pytest.approx(anatomy["entropy_rate"], abs=1e-9)
    assert anatomy["j_mu"] <= anatomy["bound_mu"] + 1e-9
    assert anatomy["j_mu"] <= anatomy["rho_mu"] + 1e-9


def test_nrps_j_mu_differs_from_rho_mu():
    est = noisy_random_phase_slip().block_convergence_estimates(max_length=8, max_caekl_length=6)
    assert est.j_mu != pytest.approx(est.rho_mu, abs=1e-3)


def test_max_caekl_length_cap():
    eps = golden_mean(0.5)
    capped = eps.block_convergence_estimates(8, max_caekl_length=5)
    assert capped.block_caekl[5] > 0.0
    assert capped.block_caekl[6] == pytest.approx(0.0, abs=1e-12)


def test_block_caekl_helper_matches_diagram():
    eps = golden_mean(0.5)
    assert block_caekl(eps, 3) == pytest.approx(
        eps.block_convergence_diagram(3).block_caekl[3],
        abs=1e-12,
    )


def test_plot_block_convergence_diagram():
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    _, ax = plt.subplots()
    result = golden_mean(0.5).plot_block_convergence_diagram(4, ax=ax, figure="fig4", show_legend=False)
    assert result is ax
    assert len(ax.lines) >= 2
    plt.close(ax.figure)
