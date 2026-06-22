"""Tests for Markov and cryptic orders."""

import math

import pytest

from pensive.examples.epsilon_machines import (
    bernoulli,
    butterfly_process,
    golden_mean,
    golden_mean_markov,
    nemo_process,
    phase_slip_backtrack,
    restricted_golden_mean,
)
from pensive.generators.synchronization import graph_from_epsilon_machine, markov_order_from_graph


def test_bernoulli_orders():
    eps = bernoulli()
    assert eps.markov_order() == 0
    assert eps.cryptic_order() == 0
    assert eps.is_exactly_synchronizable()


def test_golden_mean_orders():
    eps = golden_mean()
    assert eps.markov_order() == 1
    assert eps.cryptic_order() == 1


def test_golden_mean_markov_order():
    assert golden_mean_markov().markov_order() == 1


@pytest.mark.parametrize("k", [1, 2, 3])
def test_restricted_golden_mean_cryptic_order(k: int):
    eps = restricted_golden_mean(k)
    assert eps.cryptic_order() == k
    assert eps.markov_order() == k


def test_phase_slip_backtrack_markov_order():
    eps = phase_slip_backtrack()
    assert eps.markov_order() == 3


def test_butterfly_infinite_markov_finite_cryptic():
    eps = butterfly_process()
    assert eps.markov_order() == math.inf
    assert eps.cryptic_order() == 3
    assert not eps.is_exactly_synchronizable()


def test_nemo_infinite_orders():
    eps = nemo_process()
    assert eps.markov_order() == math.inf
    assert eps.cryptic_order() == math.inf


def test_orders_invariant_under_probability_rescaling():

    eps = golden_mean(0.5)
    graph = graph_from_epsilon_machine(eps)
    base_r = markov_order_from_graph(graph)
    for p in (0.2, 0.8):
        other = golden_mean(p)
        assert markov_order_from_graph(graph_from_epsilon_machine(other)) == base_r


def test_cryptic_order_bounded_by_markov_order_when_finite():
    eps = restricted_golden_mean(2)
    r = eps.markov_order()
    k = eps.cryptic_order()
    assert r != math.inf and k != math.inf
    assert k <= r
