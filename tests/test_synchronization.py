"""Tests for Markov and cryptic orders."""

import math

import pytest
from hypothesis import given, settings

from sofic.examples.epsilon_machines import (
    alternating_biased_coins,
    bernoulli,
    butterfly_process,
    even_process,
    golden_mean,
    golden_mean_markov,
    nemo_process,
    phase_slip_backtrack,
    restricted_golden_mean,
)
from sofic.generators.synchronization import (
    graph_from_epsilon_machine,
    is_asymptotically_synchronizable_from_graph,
    is_definite_from_graph,
    is_exactly_synchronizable,
    markov_order_from_graph,
    reset_threshold_from_graph,
    shortest_synchronizing_word_from_graph,
)
from sofic.serialization import model_from_yaml
from sofic.testing.strategies import epsilon_machines

INFINITE_ORDER_EPSILON_MACHINE_YAML = """
schema: sofic.model
version: 1
class: sofic.generators.epsilon_machine.EpsilonMachine
graph:
  nodes:
  - id: 0
    attrs:
      __sofic_type__: dict
      items: []
  - id: 1
    attrs:
      __sofic_type__: dict
      items: []
  - id: 2
    attrs:
      __sofic_type__: dict
      items: []
  edges:
  - source: 0
    target: 1
    key: 0
    attrs:
      __sofic_type__: dict
      items:
      - key: prob
        value: 0.5
      - key: emission
        value: 0
  - source: 0
    target: 2
    key: 0
    attrs:
      __sofic_type__: dict
      items:
      - key: prob
        value: 0.5
      - key: emission
        value: 1
  - source: 1
    target: 0
    key: 0
    attrs:
      __sofic_type__: dict
      items:
      - key: prob
        value: 0.5
      - key: emission
        value: 0
  - source: 1
    target: 2
    key: 0
    attrs:
      __sofic_type__: dict
      items:
      - key: prob
        value: 0.5
      - key: emission
        value: 1
  - source: 2
    target: 0
    key: 0
    attrs:
      __sofic_type__: dict
      items:
      - key: prob
        value: 1.0
      - key: emission
        value: 1
metadata:
  __sofic_type__: dict
  items:
  - key: initial_distribution
    value:
      __sofic_type__: dict
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
    assert eps.is_definite()
    assert eps.reset_threshold() == 0
    assert eps.synchronizing_word() == []


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
    assert eps.markov_order() == math.inf


def test_butterfly_infinite_orders():
    eps = butterfly_process()
    assert eps.markov_order() == math.inf
    assert eps.cryptic_order() == math.inf
    # Exactly synchronizable (a single sync symbol resets it) despite infinite
    # Markov order -- so it is not definite.
    assert eps.is_exactly_synchronizable()
    assert eps.reset_threshold() == 1
    assert not eps.is_definite()


def test_nemo_infinite_orders():
    eps = nemo_process()
    assert eps.markov_order() == math.inf
    assert eps.cryptic_order() == math.inf


def test_yaml_machine_with_transient_belief_self_loops_has_infinite_orders():
    eps = model_from_yaml(INFINITE_ORDER_EPSILON_MACHINE_YAML)

    assert eps.markov_order() == math.inf
    assert eps.cryptic_order() == math.inf
    # Infinite orders, but a length-2 word still synchronizes the belief.
    assert eps.is_exactly_synchronizable()
    assert eps.reset_threshold() == 2
    assert not eps.is_definite()


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


@given(machine=epsilon_machines(max_states=3))
@settings(max_examples=25)
def test_cryptic_order_never_exceeds_markov_order(machine):
    assert machine.markov_order() >= machine.cryptic_order()


def test_golden_mean_reset_threshold_and_definite():
    eps = golden_mean()
    assert eps.reset_threshold() == 1
    assert eps.is_exactly_synchronizable()
    assert eps.is_definite()
    assert eps.synchronizing_word() == [0]


def test_even_process_exact_but_not_definite():
    eps = even_process()
    assert eps.markov_order() == math.inf
    assert eps.reset_threshold() == 1
    assert eps.is_exactly_synchronizable()
    assert not eps.is_definite()
    assert len(eps.synchronizing_word()) == 1


def test_alternating_biased_coins_asymptotic_but_not_exact():
    eps = alternating_biased_coins()
    assert eps.reset_threshold() == math.inf
    assert not eps.is_exactly_synchronizable()
    assert eps.synchronizing_word() is None
    # Asymptotic synchronization still holds for every finite-state machine.
    assert eps.is_asymptotically_synchronizable()


def test_phase_slip_backtrack_reset_word():
    eps = phase_slip_backtrack()
    assert eps.markov_order() == math.inf
    assert eps.is_exactly_synchronizable()
    reset = eps.reset_threshold()
    assert reset == 2
    word = eps.synchronizing_word()
    assert word is not None
    assert len(word) == reset


def test_reset_threshold_at_most_markov_order_when_finite():
    eps = restricted_golden_mean(2)
    r = eps.markov_order()
    reset = eps.reset_threshold()
    assert r != math.inf and reset != math.inf
    assert reset <= r


def test_definite_implies_exactly_synchronizable():
    for eps in (bernoulli(), golden_mean(), golden_mean_markov(), restricted_golden_mean(3)):
        assert eps.is_definite()
        assert eps.is_exactly_synchronizable()


@given(machine=epsilon_machines(max_states=3))
@settings(max_examples=25)
def test_synchronization_invariants(machine):
    graph = graph_from_epsilon_machine(machine)
    reset = reset_threshold_from_graph(graph)
    markov = markov_order_from_graph(graph)
    exact = is_exactly_synchronizable(graph)
    word = shortest_synchronizing_word_from_graph(graph)

    # Exact synchronizability <=> a finite reset threshold (Travers Def. 7).
    assert exact == (reset != math.inf)
    # Definiteness (finite Markov order) implies exact synchronizability.
    if is_definite_from_graph(graph):
        assert exact
    # The shortest reset word never exceeds the worst case (Markov order).
    if markov != math.inf:
        assert reset <= markov
    # The word realizes the reset threshold, and is None iff not exact.
    if exact:
        assert word is not None
        assert len(word) == reset
    else:
        assert word is None
    # Asymptotic synchronizability holds exactly for non-empty presentations.
    assert is_asymptotically_synchronizable_from_graph(graph) == bool(graph.states)
