"""Tests for James et al. (2011) block convergence suite."""

from __future__ import annotations

import math

import pytest

from pensive.examples import even_process, fair_coin, golden_mean, noisy_random_phase_slip


def test_fair_coin_block_convergence_independent():
    diag = fair_coin().block_convergence_diagram(4)
    diag.validate_identities()
    assert diag.block_total_correlation[2:] == pytest.approx(0.0, abs=1e-12)
    assert diag.block_binding_information[2:] == pytest.approx(0.0, abs=1e-12)
    assert diag.block_caekl[2:] == pytest.approx(0.0, abs=1e-12)
    assert diag.entropy_rate == pytest.approx(1.0, abs=1e-12)
    assert diag.excess_entropy == pytest.approx(0.0, abs=1e-12)
    assert diag.ephemeral_information == pytest.approx(1.0, abs=1e-12)
    assert diag.bound_information == pytest.approx(0.0, abs=1e-12)


def test_golden_mean_table_i_scalars():
    est = golden_mean(0.5).block_convergence_estimates(max_length=8)
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
    est = eps.block_convergence_estimates(max_length=8)
    est.validate_identities()
    assert est.q_mu < 0.0
    assert est.r_mu < est.h_mu


def test_nrps_table_i_scalars():
    est = noisy_random_phase_slip().block_convergence_estimates(max_length=8)
    est.validate_identities()
    assert est.block_entropy[1] == pytest.approx(0.97987, abs=1e-4)
    assert est.h_mu == pytest.approx(0.5, abs=1e-4)
    assert est.rho_mu == pytest.approx(0.47987, abs=1e-4)
    assert est.E == pytest.approx(1.57393, abs=1e-4)
    assert est.r_mu + est.b_mu == pytest.approx(est.h_mu, abs=1e-4)


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
    diag = golden_mean(0.5).block_convergence_diagram(8)
    assert abs(float(diag.block_coinformation[-1])) < 1e-4


def test_nrps_block_curves_qualitative():
    diag = noisy_random_phase_slip().block_convergence_diagram(8)
    assert diag.block_total_correlation[-1] > 0.0
    assert diag.h_mu != pytest.approx(diag.rho_mu, abs=1e-3)


def test_information_anatomy_keys():
    anatomy = golden_mean(0.5).block_convergence_estimates(6).information_anatomy()
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
    }
    assert expected <= set(anatomy.keys())
    assert anatomy["bound_mu"] + anatomy["ephemeral_mu"] == pytest.approx(anatomy["entropy_rate"], abs=1e-9)


def test_plot_block_convergence_diagram():
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    _, ax = plt.subplots()
    result = golden_mean(0.5).plot_block_convergence_diagram(4, ax=ax, figure="fig4", show_legend=False)
    assert result is ax
    assert len(ax.lines) >= 2
    plt.close(ax.figure)
