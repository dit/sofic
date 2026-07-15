"""Tests for epsilon-machine block entropy diagrams."""

from __future__ import annotations

import math
from unittest.mock import patch

import numpy as np
import pytest
from hypothesis import given, settings

from sofic.examples import fair_coin, golden_mean
from sofic.generators.epsilon_machine import EpsilonMachine
from sofic.generators.topological_epsilon_enumeration import idfa_string_to_epsilon_machine
from sofic.testing.strategies import epsilon_machines


def test_fair_coin_block_entropy_diagram_is_linear():
    diagram = fair_coin().block_entropy_diagram(3)

    assert diagram.lengths.tolist() == [0, 1, 2, 3]
    assert diagram.block_entropy == pytest.approx([0.0, 1.0, 2.0, 3.0], abs=1e-12)
    assert diagram.state_block_entropy == pytest.approx(diagram.block_entropy, abs=1e-12)
    assert diagram.block_state_entropy == pytest.approx(diagram.block_entropy, abs=1e-12)
    assert diagram.entropy_asymptote == pytest.approx(diagram.block_entropy, abs=1e-12)
    assert diagram.crypticity_estimate == pytest.approx(np.zeros(4), abs=1e-12)
    assert diagram.entropy_rate == pytest.approx(1.0, abs=1e-12)
    assert diagram.excess_entropy == pytest.approx(0.0, abs=1e-12)
    assert diagram.crypticity == pytest.approx(0.0, abs=1e-12)
    assert diagram.markov_order == 0
    assert diagram.cryptic_order == 0


def test_golden_mean_diagram_marks_order_one_convergence():
    diagram = golden_mean(0.5).block_entropy_diagram(3)

    assert diagram.entropy_rate == pytest.approx(2.0 / 3.0, abs=1e-12)
    assert diagram.excess_entropy == pytest.approx(0.25162916738782304, abs=1e-12)
    assert diagram.statistical_complexity == pytest.approx(0.9182958340544896, abs=1e-12)
    assert diagram.crypticity == pytest.approx(2.0 / 3.0, abs=1e-12)
    assert diagram.markov_order == 1
    assert diagram.cryptic_order == 1
    assert diagram.block_entropy[1:] == pytest.approx(diagram.entropy_asymptote[1:], abs=1e-12)
    assert diagram.block_state_entropy[1:] == pytest.approx(diagram.entropy_asymptote[1:], abs=1e-12)
    assert diagram.crypticity_estimate[1:] == pytest.approx([diagram.crypticity] * 3, abs=1e-12)


def test_block_entropy_diagram_rejects_negative_length():
    with pytest.raises(ValueError, match="max_length"):
        fair_coin().block_entropy_diagram(-1)


def test_plot_block_entropy_diagram_feature_flags():
    matplotlib = pytest.importorskip("matplotlib")

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    _, ax = plt.subplots()
    result = golden_mean(0.5).plot_block_entropy_diagram(
        3,
        ax=ax,
        show_state_block_entropy=False,
        show_block_state_entropy=False,
        show_asymptote=False,
        show_markov_order=False,
        show_cryptic_order=False,
        show_crypticity=False,
        show_grid=False,
        show_legend=False,
    )

    assert result is ax
    assert len(ax.lines) == 1
    assert ax.lines[0].get_label() == r"$H[X_{0:L}]$"
    assert not any(line.get_visible() for line in ax.xaxis.get_gridlines())
    assert not any(line.get_visible() for line in ax.yaxis.get_gridlines())
    plt.close(ax.figure)


def test_plot_block_entropy_diagram_default_features():
    matplotlib = pytest.importorskip("matplotlib")

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    _, ax = plt.subplots()
    result = golden_mean(0.5).plot_block_entropy_diagram(3, ax=ax, show_legend=False)

    labels = [line.get_label() for line in ax.lines]
    assert result is ax
    assert r"$H[S_0, X_{0:L}]$" not in labels
    assert r"$\chi(L)$" not in labels
    assert r"$\chi$" not in labels
    assert any(line.get_visible() for line in ax.xaxis.get_gridlines())
    assert any(line.get_visible() for line in ax.yaxis.get_gridlines())
    plt.close(ax.figure)


def test_plot_block_entropy_diagram_hidden_features_can_be_enabled():
    matplotlib = pytest.importorskip("matplotlib")

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    _, ax = plt.subplots()
    result = golden_mean(0.5).plot_block_entropy_diagram(
        3,
        ax=ax,
        show_state_block_entropy=True,
        show_crypticity=True,
        show_legend=False,
    )

    labels = [line.get_label() for line in ax.lines]
    assert result is ax
    assert r"$H[S_0, X_{0:L}]$" in labels
    assert r"$\chi(L)$" in labels
    assert r"$\chi$" in labels
    plt.close(ax.figure)


def test_fair_coin_block_entropy_estimates_anatomy_rates():
    estimates = fair_coin().block_entropy_estimates(3)

    assert estimates.h_mu == pytest.approx(1.0, abs=1e-12)
    assert pytest.approx(0.0, abs=1e-12) == estimates.E
    assert estimates.r_mu == pytest.approx(1.0, abs=1e-12)
    assert estimates.b_mu == pytest.approx(0.0, abs=1e-12)
    assert estimates.predicted_information == pytest.approx(0.0, abs=1e-12)
    assert estimates.information_anatomy()["bound_mu"] + estimates.information_anatomy()["ephemeral_mu"] == (
        pytest.approx(estimates.h_mu, abs=1e-12)
    )


def test_golden_mean_block_entropy_estimates_match_finite_order_values():
    estimates = golden_mean(0.5).block_entropy_estimates(4)

    assert estimates.entropy_rate == pytest.approx(2.0 / 3.0, abs=1e-12)
    assert estimates.excess_entropy == pytest.approx(0.25162916738782304, abs=1e-12)
    assert estimates.crypticity == pytest.approx(2.0 / 3.0, abs=1e-12)
    anatomy = golden_mean(0.5).approximate_information_anatomy(4)
    assert anatomy["bound_mu"] + anatomy["ephemeral_mu"] == pytest.approx(anatomy["entropy_rate"], abs=1e-12)


def test_block_entropy_estimates_fallback_when_exact_excess_entropy_fails():
    with patch.object(EpsilonMachine, "excess_entropy", side_effect=RuntimeError("no bidirectional")):
        estimates = golden_mean(0.5).block_entropy_estimates(3, use_exact=True)

    assert np.isfinite(estimates.excess_entropy)
    assert estimates.excess_entropy == pytest.approx(0.25162916738782304, abs=1e-12)


def test_block_entropy_diagram_uses_exact_excess_entropy_when_available():
    machine = idfa_string_to_epsilon_machine((0, 1, -1, 2, 0, 2), n=3, k=2, alphabet=("0", "1"))

    exact = machine.to_bidirectional().excess_entropy()
    diagram = machine.block_entropy_diagram(2)
    finite_order_estimate = diagram.block_entropy[2] - 2 * diagram.entropy_rate

    assert diagram.excess_entropy == pytest.approx(exact, abs=1e-9)
    assert finite_order_estimate == pytest.approx(exact, abs=1e-9)


def test_block_entropy_diagram_fallback_when_exact_excess_entropy_fails():
    with patch.object(EpsilonMachine, "to_bidirectional", side_effect=RuntimeError("no bidirectional")):
        diagram = golden_mean(0.5).block_entropy_diagram(1)

    assert diagram.excess_entropy == pytest.approx(0.25162916738782304, abs=1e-12)


@given(machine=epsilon_machines(max_states=3))
@settings(max_examples=25, deadline=None)
def test_finite_markov_order_block_entropy_reaches_entropy_asymptote(machine):
    markov_order = machine.markov_order()
    if not math.isfinite(markov_order):
        return

    order = int(markov_order)
    diagram = machine.block_entropy_diagram(order)

    assert diagram.block_entropy[order] == pytest.approx(
        diagram.excess_entropy + order * diagram.entropy_rate,
        abs=1e-9,
    )
