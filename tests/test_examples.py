"""Tests for canonical ε-machine examples."""

from __future__ import annotations

import numpy as np
import pytest

from sofic.examples import (
    alternating_biased_coins,
    bernoulli,
    butterfly_process,
    ellison_fig9_reverse,
    even_process,
    golden_mean,
    irreversible_two_state,
    nemo_process,
    restricted_golden_mean,
)
from sofic.generators.bidirectional_epsilon_machine import BidirectionalEpsilonMachine
from sofic.shifts.tmc import TopologicalMarkovChain


@pytest.mark.parametrize(
    "constructor",
    [
        bernoulli,
        even_process,
        golden_mean,
        alternating_biased_coins,
        restricted_golden_mean,
        nemo_process,
        butterfly_process,
        irreversible_two_state,
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
    assert ("A", "0", "A") in edges
    assert ("A", "1", "B") in edges
    assert ("B", "0", "A") in edges
    assert edges[("A", "0", "A")] == pytest.approx(0.5)
    assert edges[("A", "1", "B")] == pytest.approx(0.5)
    assert edges[("B", "0", "A")] == pytest.approx(1.0)
    assert len(edges) == 3


def test_even_and_bernoulli_differ_in_complexity():
    pytest.importorskip("dit")
    even = even_process(0.4)
    memoryless = bernoulli(0.4)
    assert even.statistical_complexity() > memoryless.statistical_complexity()


def test_golden_mean_stationary():
    gm = golden_mean(0.5)
    assert gm.initial_distribution["A"] == pytest.approx(2.0 / 3.0, abs=1e-9)
    assert gm.initial_distribution["B"] == pytest.approx(1.0 / 3.0, abs=1e-9)


def test_golden_mean_parry_measure_attains_topological_entropy():
    phi = (1 + np.sqrt(5)) / 2
    parry = golden_mean(1 / phi)
    tmc = TopologicalMarkovChain.from_adjacency(
        np.array([[1, 1], [1, 0]], dtype=float),
        symbol_alphabet=frozenset({"0", "1"}),
    )
    assert parry.entropy_rate() == pytest.approx(np.log2(phi), abs=1e-12)
    assert parry.entropy_rate() == pytest.approx(tmc.topological_entropy(), abs=1e-12)
    for p in (0.3, 0.5, 0.7):
        assert golden_mean(p).entropy_rate() < parry.entropy_rate()


def test_butterfly_statistical_complexity():
    pytest.importorskip("dit")
    butterfly = butterfly_process()
    pi = butterfly.stationary_distribution()
    import dit

    expected = float(dit.shannon.entropy(dit.Distribution([(i,) for i in range(len(pi))], pi)))
    assert butterfly.statistical_complexity() == pytest.approx(expected, abs=1e-9)


def test_ellison_fig9_information_identities():
    pytest.importorskip("dit")
    forward = irreversible_two_state()
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


def _brute_force_state_uncertainty(machine, lag: int, horizon: int) -> float:
    """``H[S_lag | X_{0:horizon}]`` in bits by enumerating every stationary path."""
    pi = machine.stationary_distribution()
    states = list(machine.states())
    joint: dict[tuple, float] = {}
    frontier = [((), states[i], states[i], float(pi[i])) for i in range(len(states))]
    for step in range(horizon):
        extended = []
        for word, state, state_at_lag, prob in frontier:
            for t in machine.graph.out_transitions(state):
                lag_state = t.target if step + 1 == lag else state_at_lag
                extended.append((word + (t.data["emission"],), t.target, lag_state, prob * t.data["prob"]))
        frontier = extended
    for word, _state, state_at_lag, prob in frontier:
        joint[(word, state_at_lag)] = joint.get((word, state_at_lag), 0.0) + prob
    words: dict[tuple, float] = {}
    for (word, _), prob in joint.items():
        words[word] = words.get(word, 0.0) + prob
    return float(-sum(p * np.log2(p / words[w]) for (w, _), p in joint.items() if p > 0))


def test_butterfly_matches_mahoney_fig3():
    """Mahoney, Ellison & Crutchfield (2009) Fig. 3: five states, C_mu = log2 5, chi = 3/10, 2-cryptic."""
    butterfly = butterfly_process()
    assert len(list(butterfly.states())) == 5
    assert butterfly.statistical_complexity() == pytest.approx(np.log2(5), abs=1e-9)
    assert butterfly.entropy_rate() == pytest.approx(1.0, abs=1e-9)
    assert butterfly.crypticity() == pytest.approx(0.3, abs=1e-9)
    assert butterfly.excess_entropy() == pytest.approx(np.log2(5) - 0.3, abs=1e-9)
    assert _brute_force_state_uncertainty(butterfly, lag=1, horizon=9) > 0.1
    assert _brute_force_state_uncertainty(butterfly, lag=2, horizon=10) == pytest.approx(0.0, abs=1e-12)
    assert butterfly.cryptic_order() == 2


_SYMBOL_ATTRS = ("emission", "symbol", "output", "future_symbol")


def _default_example_factories():
    import inspect

    import sofic.examples as examples

    for name in examples.__all__:
        factory = getattr(examples, name)
        if not inspect.isfunction(factory):
            continue
        required = [p for p in inspect.signature(factory).parameters.values() if p.default is inspect.Parameter.empty]
        if not required:
            yield pytest.param(factory, id=name)


@pytest.mark.parametrize("factory", list(_default_example_factories()))
def test_example_factories_emit_string_symbols(factory):
    machine = factory()
    graph = getattr(machine, "graph", None)
    if graph is None:
        pytest.skip("not a graph-backed model")
    for transition in graph.transitions():
        for attr in _SYMBOL_ATTRS:
            value = transition.data.get(attr)
            assert value is None or isinstance(value, str), (attr, value)
    for attr in ("observation_alphabet", "input_alphabet", "output_alphabet", "symbol_alphabet"):
        alphabet = getattr(machine, attr, None)
        if alphabet:
            assert all(isinstance(symbol, str) for symbol in alphabet), attr
