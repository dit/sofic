"""Capacity bounds for ε-transducers, checked against known capacities."""

import math
import warnings

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from sofic.examples.processes import binary_channel_et, gm_to_even, ising_channel_et, trapdoor_et
from sofic.exceptions import StochasticValidationError
from sofic.generators.capacity import (
    CapacityBounds,
    ChannelTensors,
    QGraph,
    capacity,
    channel_tensors,
    duality_upper_bound,
    duality_upper_bound_no_feedback,
    feedback_capacity,
    markov_input_lower_bound,
    qgraph_bounds,
    qgraph_upper_bound,
    value_iteration_estimate,
)
from sofic.generators.capacity._belief import SimplexGrid

LOG2_PHI = math.log2((1 + math.sqrt(5)) / 2)
ISING_FB = 0.575522  # Elishco & Permuter (2014)
TRAPDOOR_GRAPH = [[0, 1], [0, 2], [1, 2]]
ISING_GRAPH = [[1, 2], [0, 2], [0, 3], [0, 2]]


@pytest.fixture(autouse=True)
def _quiet():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        yield


def _graph(table):
    return QGraph(("0", "1"), np.array(table), tuple(range(len(table))))


def _bsc_capacity(p, q):
    from dit.algorithms.channelcapacity import channel_capacity

    return channel_capacity(np.array([[1 - p, p], [q, 1 - q]]))[0]


# -- infrastructure -----------------------------------------------------------


def test_channel_tensors_of_trapdoor():
    tensors = channel_tensors(trapdoor_et())
    assert tensors.P.shape == (2, 2, 2)
    assert np.allclose(tensors.P.sum(axis=2), 1.0)
    assert tensors.allowed.all()
    assert np.allclose(tensors.initial, 0.5)
    assert tensors.trivial_upper_bound() == pytest.approx(1.0)


def test_non_strongly_connected_channel_rejected():
    with pytest.raises(StochasticValidationError, match="strongly connected"):
        feedback_capacity(_absorbing_channel())


def _absorbing_channel():
    P = np.zeros((2, 1, 2))
    F = np.full((2, 1, 2), -1)
    P[0, 0] = [0.5, 0.5]
    F[0, 0] = [0, 1]
    P[1, 0, 1] = 1.0
    F[1, 0, 1] = 1
    return ChannelTensors(("a", "b"), ("x",), ("0", "1"), P, F, P.sum(axis=2) > 0, np.array([1.0, 0.0]))


def test_markov_qgraph_structure():
    graph = QGraph.markov(("0", "1"), 2)
    assert graph.n_nodes == 4
    assert graph.labels == ("00", "01", "10", "11")
    assert graph.table[graph.labels.index("01"), 1] == graph.labels.index("11")
    assert QGraph.markov(("0", "1"), 0).table.tolist() == [[0, 0]]


def test_qgraph_from_dfa_round_trip():
    from sofic.automata.dfa import DFA

    dfa = DFA(input_alphabet=frozenset({"0", "1"}), initial_states=frozenset({"a"}))
    for state in "abc":
        dfa.graph.add_state(state)
    for source, row in zip("abc", TRAPDOOR_GRAPH, strict=True):
        for symbol, target in zip("01", row, strict=True):
            dfa.add_transition(source, "abc"[target], symbol)
    graph = QGraph.from_dfa(dfa)
    assert graph.table.tolist() == TRAPDOOR_GRAPH


def test_capacity_bounds_value():
    assert CapacityBounds(0.5, 0.50001, "x", feedback=True).value == pytest.approx(0.500005)
    assert CapacityBounds(0.5, 0.6, "x", feedback=True).value is None


@settings(max_examples=30, deadline=None)
@given(st.integers(2, 5), st.integers(1, 6), st.integers(0, 2**32 - 1))
def test_simplex_interpolation_reproduces_linear_functions(n_vertices, resolution, seed):
    grid = SimplexGrid.build(n_vertices, resolution)
    rng = np.random.default_rng(seed)
    queries = np.vstack([rng.dirichlet(np.ones(n_vertices), 20), np.eye(n_vertices)])
    indices, weights = grid.interpolation(queries)
    assert np.allclose(weights.sum(axis=1), 1.0)
    assert (weights >= -1e-12).all()
    assert np.allclose(np.einsum("mk,mks->ms", weights, grid.points[indices]), queries)


# -- memoryless channel -------------------------------------------------------


@pytest.mark.parametrize(("p", "q"), [(0.1, 0.2), (0.05, 0.05), (0.3, 0.0)])
def test_memoryless_channel_matches_blahut_arimoto(p, q):
    channel = binary_channel_et(p, q)
    reference = _bsc_capacity(p, q)
    fb = qgraph_bounds(channel, max_order=1, restarts=1)
    assert fb.lower == pytest.approx(reference, abs=1e-6)
    assert fb.upper == pytest.approx(reference, abs=1e-6)
    assert duality_upper_bound(channel, max_order=0).upper == pytest.approx(reference, abs=1e-6)
    no_fb = capacity(channel, order=0, max_order=0)
    assert no_fb.lower == pytest.approx(reference, abs=1e-6)
    assert no_fb.upper == pytest.approx(reference, abs=1e-6)


def test_value_iteration_on_memoryless_channel_is_uncertified_estimate():
    reference = _bsc_capacity(0.1, 0.2)
    estimate = value_iteration_estimate(binary_channel_et(0.1, 0.2), belief_resolution=4, action_resolution=40)
    assert not estimate.certified
    assert estimate.upper == pytest.approx(reference, abs=1e-3)


# -- trapdoor channel ---------------------------------------------------------


def test_trapdoor_feedback_capacity_is_log2_phi():
    bounds = trapdoor_et().feedback_capacity(qgraphs=[_graph(TRAPDOOR_GRAPH)], max_order=1)
    assert bounds.lower <= LOG2_PHI + 1e-9
    assert bounds.upper >= LOG2_PHI - 1e-9
    assert bounds.gap < 1e-4
    assert bounds.value == pytest.approx(LOG2_PHI, abs=1e-4)


def test_trapdoor_markov_qgraphs_are_not_tight():
    tensors = channel_tensors(trapdoor_et())
    upper, _joint = qgraph_upper_bound(tensors, QGraph.markov(tensors.outputs, 2))
    assert upper == pytest.approx(0.704837, abs=1e-5)


def test_trapdoor_value_iteration_estimate():
    estimate = value_iteration_estimate(trapdoor_et(), belief_resolution=100, action_resolution=20)
    assert estimate.lower == pytest.approx(LOG2_PHI, abs=2e-3)
    assert estimate.details["policy"].shape[1:] == (2, 2)


def test_trapdoor_duality_bound_is_valid():
    bound = duality_upper_bound(trapdoor_et(), qgraphs=[_graph(TRAPDOOR_GRAPH)], max_order=-1, optimize=False)
    assert LOG2_PHI - 1e-9 <= bound.upper < LOG2_PHI + 1e-3


@pytest.mark.slow
def test_trapdoor_duality_bound_is_tight():
    bound = duality_upper_bound(trapdoor_et(), qgraphs=[_graph(TRAPDOOR_GRAPH)], max_order=-1)
    assert bound.upper == pytest.approx(LOG2_PHI, abs=1e-6)


def test_trapdoor_no_feedback_bracket():
    lower = markov_input_lower_bound(trapdoor_et(), order=2)
    assert 0.571 < lower.lower <= 0.584
    upper = duality_upper_bound_no_feedback(trapdoor_et(), max_order=0, belief_resolution=20, optimize=False)
    assert upper.upper >= 0.572


@pytest.mark.slow
def test_trapdoor_no_feedback_bounds_calibrated():
    lower = markov_input_lower_bound(trapdoor_et(), order=3)
    upper = duality_upper_bound_no_feedback(trapdoor_et(), max_order=1, belief_resolution=24)
    assert lower.lower > 0.5726
    assert 0.584 <= upper.upper < 0.594
    assert upper.upper <= feedback_capacity(trapdoor_et(), qgraphs=[_graph(TRAPDOOR_GRAPH)], max_order=0).upper + 1e-3


# -- Ising channel ------------------------------------------------------------


def test_ising_feedback_capacity():
    bounds = ising_channel_et().feedback_capacity(qgraphs=[_graph(ISING_GRAPH)], max_order=0)
    assert bounds.gap < 1e-4
    assert bounds.lower == pytest.approx(ISING_FB, abs=1e-5)
    assert bounds.upper == pytest.approx(ISING_FB, abs=1e-5)


def test_ising_duality_bound_is_valid():
    bound = duality_upper_bound(ising_channel_et(), qgraphs=[_graph(ISING_GRAPH)], max_order=-1, optimize=False)
    assert bound.upper >= ISING_FB - 1e-6


@pytest.mark.slow
def test_ising_duality_bound_is_tight():
    bound = duality_upper_bound(ising_channel_et(), qgraphs=[_graph(ISING_GRAPH)], max_order=-1)
    assert bound.upper == pytest.approx(ISING_FB, abs=1e-5)


@pytest.mark.slow
def test_feedback_capacity_all_methods():
    bounds = feedback_capacity(
        ising_channel_et(), method="all", qgraphs=[_graph(ISING_GRAPH)], max_order=1, belief_resolution=60
    )
    assert bounds.lower == pytest.approx(ISING_FB, abs=1e-5)
    assert bounds.upper == pytest.approx(ISING_FB, abs=1e-5)
    assert not bounds.details["value_iteration"].certified


# -- invariants on random channels --------------------------------------------


@st.composite
def unifilar_channels(draw):
    """Random strongly connected binary-input binary-output channels on two states."""
    seed = draw(st.integers(0, 2**32 - 1))
    rng = np.random.default_rng(seed)
    P = rng.dirichlet(np.ones(2), size=(2, 2))
    P = np.where(P < 0.05, 0.0, P)
    P /= P.sum(axis=2, keepdims=True)
    F = rng.integers(0, 2, size=(2, 2, 2))
    F = np.where(P > 0, F, -1)
    tensors = ChannelTensors(("a", "b"), ("0", "1"), ("0", "1"), P, F, np.ones((2, 2), dtype=bool), np.full(2, 0.5))
    reach = {(s, int(F[s, x, y])) for s in range(2) for x in range(2) for y in range(2) if P[s, x, y] > 0}
    if not {(0, 1), (1, 0)} <= reach:
        F[0, 0, int(np.argmax(P[0, 0]))] = 1
        F[1, 1, int(np.argmax(P[1, 1]))] = 0
    return tensors


@settings(max_examples=8, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(unifilar_channels())
def test_bounds_are_ordered_on_random_channels(tensors):
    fb = qgraph_bounds(tensors, max_order=1, restarts=1)
    for _, lower, upper in fb.details["per_qgraph"]:
        assert lower <= upper + 1e-6
    no_fb_lower = markov_input_lower_bound(tensors, order=1, restarts=1)
    no_fb_upper = duality_upper_bound_no_feedback(tensors, max_order=0, belief_resolution=10, optimize=False)
    assert no_fb_lower.lower <= fb.upper + 1e-6
    assert no_fb_lower.lower <= no_fb_upper.upper + 1e-6
    for value in (fb.lower, fb.upper, no_fb_lower.lower, no_fb_upper.upper):
        assert -1e-9 <= value <= 1.0 + 1e-9


def test_input_constrained_channel_rejected_without_feedback():
    channel = gm_to_even()
    from sofic import EpsilonTransducer

    eps = EpsilonTransducer.from_channel(channel)
    with pytest.raises(StochasticValidationError):
        markov_input_lower_bound(eps)
