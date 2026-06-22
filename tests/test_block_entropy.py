"""Tests for epsilon-machine block entropy diagrams."""

from __future__ import annotations

import numpy as np
import pytest

from pensive.examples import fair_coin, golden_mean


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
        show_legend=False,
    )

    assert result is ax
    assert len(ax.lines) == 1
    assert ax.lines[0].get_label() == r"$H[X_{0:L}]$"
    plt.close(ax.figure)
