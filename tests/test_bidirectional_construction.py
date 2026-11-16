"""Edge cases of the Eq. (15) helpers in :mod:`sofic.generators.bidirectional_construction`."""

from __future__ import annotations

import math

import pytest

from sofic.examples import even_process, golden_mean
from sofic.exceptions import StochasticValidationError
from sofic.generators import bidirectional_construction as bc
from sofic.generators.bidirectional_construction import infer_reverse_epsilon_machine
from sofic.generators.epsilon_machine import EpsilonMachine
from sofic.graph import ATTR_EMISSION, ATTR_FUTURE_SYMBOL, ATTR_PROB, TransitionGraph


def _graph(edges) -> TransitionGraph:
    graph = TransitionGraph()
    for source, target, symbol, prob in edges:
        graph.add_state(source)
        graph.add_state(target)
        graph.add_transition(source, target, **{ATTR_EMISSION: symbol, ATTR_PROB: prob})
    return graph


def test_mismatched_alphabets_are_rejected():
    forward = golden_mean()
    other = EpsilonMachine.from_hmm(even_process())
    other.observation_alphabet = frozenset({"0", "1", "2"})
    with pytest.raises(StochasticValidationError, match="alphabets must match"):
        bc.build_bidirectional_epsilon_machine(forward, other)


def test_graph_from_raw_aggregates_and_skips_massless_rows():
    a, b, c = ("a", "x"), ("b", "x"), ("c", "x")
    raw = {a: [(b, "0", 0.25), (b, "0", 0.25), (a, "1", 0.5)], b: [], c: [(a, "0", 0.0)]}
    graph = bc._graph_from_raw(raw)
    assert not list(graph.out_transitions(b))
    assert not list(graph.out_transitions(c))
    probs = {(t.target, t.data[ATTR_EMISSION]): t.data[ATTR_PROB] for t in graph.out_transitions(a)}
    assert probs == {(b, "0"): 0.5, (a, "1"): 0.5}


def test_graph_from_raw_keeps_symbolic_weights_exact():
    sp = pytest.importorskip("sympy")
    p = sp.Symbol("p", positive=True)
    a, b = ("a", "x"), ("b", "x")
    graph = bc._graph_from_raw({a: [(a, "0", p), (b, "1", 1 - p)]})
    probs = {t.target: t.data[ATTR_PROB] for t in graph.out_transitions(a)}
    assert sp.simplify(probs[a] - p) == 0
    assert sp.simplify(probs[b] - (1 - p)) == 0


def test_clean_probability_rounds_only_float_noise():
    sp = pytest.importorskip("sympy")
    assert bc._clean_probability(0.1 + 0.2) == 0.3
    x = sp.Symbol("x")
    assert bc._clean_probability(x * 2 / 2) == x


def test_prune_of_an_empty_graph_is_empty():
    graph, initial = bc._prune_to_stationary_support(TransitionGraph(), golden_mean(), golden_mean())
    assert initial == {}
    assert not list(graph.states())


def test_future_symbol_annotations_filter_compatible_pairs():
    forward = golden_mean()
    reverse = infer_reverse_epsilon_machine(forward)
    states = sorted(reverse.states(), key=repr)
    reverse.graph.nx.nodes[states[0]][ATTR_FUTURE_SYMBOL] = "1"
    reverse.graph.nx.nodes[states[1]][ATTR_FUTURE_SYMBOL] = "9"
    mapping = bc._infer_future_symbols(reverse)
    assert mapping[states[0]] == "1"
    pairs = bc._compatible_pairs(forward, reverse)
    assert all(gamma != states[1] for _alpha, gamma in pairs)
    assert all(bc._forward_emits(forward, alpha, "1") for alpha, gamma in pairs if gamma == states[0])


def test_states_named_by_symbols_carry_their_future_symbol():
    machine = EpsilonMachine(observation_alphabet=frozenset({"0", "1"}))
    for state in ("0", "1", "z"):
        machine.graph.add_state(state)
    assert bc._infer_future_symbols(machine) == {"0": "0", "1": "1"}


def test_joint_pi_on_pair_subset_rejects_leaky_or_empty_components():
    a, b = ("a", "x"), ("b", "x")
    graph = _graph([(a, a, "0", 0.5), (a, b, "1", 0.5), (b, a, "0", 1.0)])
    assert bc._joint_pi_on_pair_subset(graph, []) is None
    assert bc._joint_pi_on_pair_subset(graph, [a]) is None
    joint = bc._joint_pi_on_pair_subset(graph, [a, b])
    assert joint == {a: pytest.approx(2 / 3), b: pytest.approx(1 / 3)}


def test_joint_pi_on_pair_subset_with_symbolic_rows():
    sp = pytest.importorskip("sympy")
    p = sp.Symbol("p", positive=True)
    a, b = ("a", "x"), ("b", "x")
    graph = _graph([(a, a, "0", p), (a, b, "1", 1 - p), (b, a, "0", sp.Integer(1))])
    joint = bc._joint_pi_on_pair_subset(graph, [a, b])
    assert joint is not None
    assert sp.simplify(joint[a] - 1 / (2 - p)) == 0
    leaky = _graph([(a, a, "0", p), (b, a, "0", sp.Integer(1))])
    assert bc._joint_pi_on_pair_subset(leaky, [a, b]) is None


def test_symbolic_marginal_errors():
    sp = pytest.importorskip("sympy")
    p = sp.Symbol("p", positive=True)
    assert bc._abs_marginal_error(0.25, 0.25) == 0.0
    assert bc._abs_marginal_error(0.25, 0.5) == 0.25
    assert bc._abs_marginal_error(sp.Rational(1, 2), sp.Rational(1, 4)) == 0.25
    assert bc._abs_marginal_error(p, sp.Rational(1, 4)) == math.inf


def test_symbolic_stationary_state_probabilities():
    sp = pytest.importorskip("sympy")
    from sofic.examples.epsilon_machines import from_symbol_matrices

    p = sp.Symbol("p", positive=True)
    one, zero = sp.Integer(1), sp.Integer(0)
    machine = from_symbol_matrices(
        ["A", "B"],
        ("0", "1"),
        {"0": [[1 - p, zero], [one, zero]], "1": [[zero, p], [zero, zero]]},
    )
    probs = bc._stationary_state_probabilities(machine)
    assert sp.simplify(probs["A"] - 1 / (1 + p)) == 0
