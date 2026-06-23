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
from pensive.serialization import model_from_yaml

INFINITE_ORDER_EPSILON_MACHINE_YAML = """
schema: pensive.model
version: 1
class: pensive.generators.epsilon_machine.EpsilonMachine
graph:
  nodes:
  - id: 0
    attrs:
      __pensive_type__: dict
      items: []
  - id: 1
    attrs:
      __pensive_type__: dict
      items: []
  - id: 2
    attrs:
      __pensive_type__: dict
      items: []
  edges:
  - source: 0
    target: 1
    key: 0
    attrs:
      __pensive_type__: dict
      items:
      - key: prob
        value: 0.5
      - key: emission
        value: 0
  - source: 0
    target: 2
    key: 0
    attrs:
      __pensive_type__: dict
      items:
      - key: prob
        value: 0.5
      - key: emission
        value: 1
  - source: 1
    target: 0
    key: 0
    attrs:
      __pensive_type__: dict
      items:
      - key: prob
        value: 0.5
      - key: emission
        value: 0
  - source: 1
    target: 2
    key: 0
    attrs:
      __pensive_type__: dict
      items:
      - key: prob
        value: 0.5
      - key: emission
        value: 1
  - source: 2
    target: 0
    key: 0
    attrs:
      __pensive_type__: dict
      items:
      - key: prob
        value: 1.0
      - key: emission
        value: 1
metadata:
  __pensive_type__: dict
  items:
  - key: initial_distribution
    value:
      __pensive_type__: dict
      items:
      - key: 0
        value: 0.44444444444444464
      - key: 1
        value: 0.22222222222222218
      - key: 2
        value: 0.3333333333333333
"""


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


def test_yaml_machine_with_transient_belief_self_loops_has_infinite_orders():
    eps = model_from_yaml(INFINITE_ORDER_EPSILON_MACHINE_YAML)

    assert eps.markov_order() == math.inf
    assert eps.cryptic_order() == math.inf
    assert not eps.is_exactly_synchronizable()


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
