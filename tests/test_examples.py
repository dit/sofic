"""Tests for canonical ε-machine examples."""

from __future__ import annotations

import numpy as np
import pytest

from sofic.examples import (
    alternating_biased_coins,
    bernoulli,
    butterfly_process,
    ellison_fig9_forward,
    ellison_fig9_reverse,
    even_process,
    fair_coin,
    golden_mean,
    golden_mean_markov,
    golden_mean_shift_parry,
    nemo_process,
    restricted_golden_mean,
)
from sofic.generators.bidirectional_epsilon_machine import BidirectionalEpsilonMachine
from sofic.shifts.tmc import TopologicalMarkovChain


@pytest.mark.parametrize(
    "constructor",
    [
        fair_coin,
        even_process,
        golden_mean,
        golden_mean_markov,
        golden_mean_shift_parry,
        alternating_biased_coins,
        restricted_golden_mean,
        nemo_process,
        butterfly_process,
        ellison_fig9_forward,
        ellison_fig9_reverse,
    ],
)
def test_examples_validate(constructor):
    machine = constructor()
    machine.validate()


def test_bernoulli_parameter():
    biased = bernoulli(0.25)
    biased.validate()
    assert len(list(biased.states())) == 1


def test_golden_mean_topology():
    gm = golden_mean(0.5)
    edges = {(t.source, t.data["emission"], t.target): t.data["prob"] for t in gm.transitions()}
    assert ("A", 0, "A") in edges
    assert ("A", 1, "B") in edges
    assert ("B", 0, "A") in edges
    assert edges[("A", 0, "A")] == pytest.approx(0.5)
    assert edges[("A", 1, "B")] == pytest.approx(0.5)
    assert edges[("B", 0, "A")] == pytest.approx(1.0)
    assert len(edges) == 3


def test_even_and_bernoulli_differ_in_complexity():
    pytest.importorskip("dit")
    even = even_process(0.4)
    memoryless = bernoulli(0.4)
    assert even.statistical_complexity() > memoryless.statistical_complexity()


def test_golden_mean_markov_stationary():
    gm = golden_mean_markov(0.5)
    assert gm.initial_distribution["B"] == pytest.approx(2.0 / 3.0, abs=1e-9)
    assert gm.initial_distribution["A"] == pytest.approx(1.0 / 3.0, abs=1e-9)


def test_golden_mean_shift_parry_entropy():
    parry = golden_mean_shift_parry()
    tmc = TopologicalMarkovChain.from_adjacency(
        np.array([[1, 1], [1, 0]], dtype=float),
        symbol_alphabet=frozenset({0, 1}),
    )
    assert parry.entropy_rate() == pytest.approx(tmc.topological_entropy() / np.log(2), rel=0.05)


def test_butterfly_statistical_complexity():
    pytest.importorskip("dit")
    butterfly = butterfly_process()
    pi = butterfly.stationary_distribution()
    import dit

    expected = float(dit.shannon.entropy(dit.Distribution([(i,) for i in range(len(pi))], pi)))
    assert butterfly.statistical_complexity() == pytest.approx(expected, abs=1e-9)


def test_ellison_fig9_information_identities():
    pytest.importorskip("dit")
    forward = ellison_fig9_forward()
    reverse = ellison_fig9_reverse()
    bidir = BidirectionalEpsilonMachine.from_pair(forward, reverse)

    c_plus = forward.statistical_complexity()
    c_minus = reverse.statistical_complexity()
    excess = bidir.excess_entropy()
    c_bidir = bidir.statistical_complexity()

    assert c_plus == pytest.approx(1.0, abs=1e-9)
    assert c_minus == pytest.approx(1.5, abs=1e-9)
    assert excess == pytest.approx(0.5, abs=1e-9)
    assert c_bidir == pytest.approx(2.0, abs=1e-9)


def test_restricted_golden_mean_k1_stationary():
    rgm = restricted_golden_mean(1)
    gm = golden_mean(0.5)
    assert rgm.initial_distribution["A"] == pytest.approx(gm.initial_distribution["A"], abs=1e-9)
    assert rgm.initial_distribution["B"] == pytest.approx(gm.initial_distribution["B"], abs=1e-9)
