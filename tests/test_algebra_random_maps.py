"""Tests for random-map representations, semigroup walks, and minimal ideals."""

import dit
import numpy as np
import pytest
from dit.shannon import mutual_information
from hypothesis import given
from hypothesis import strategies as st

from sofic.algebra import TransformationSemigroup
from sofic.algebra.random_maps import (
    classify_elements,
    coupled_random_maps,
    element_walk,
    independent_random_maps,
    random_maps,
    walk_information,
)
from sofic.automata.unifilar import UnifilarAutomaton
from sofic.examples.epsilon_machines import even_process


@st.composite
def stochastic_matrices(draw, max_states=4):
    n = draw(st.integers(1, max_states))
    rows = []
    for _ in range(n):
        support = draw(st.lists(st.integers(0, n - 1), min_size=1, max_size=min(n, 3), unique=True))
        raw = [draw(st.integers(1, 5)) for _ in support]
        row = np.zeros(n)
        row[support] = np.asarray(raw) / sum(raw)
        rows.append(row)
    return np.asarray(rows)


@st.composite
def semigroups(draw, max_degree=5, max_generators=3):
    degree = draw(st.integers(1, max_degree))
    count = draw(st.integers(1, max_generators))
    gens = {
        f"g{i}": tuple(draw(st.lists(st.integers(0, degree - 1), min_size=degree, max_size=degree)))
        for i in range(count)
    }
    return TransformationSemigroup(gens)


# -- representations ------------------------------------------------------------------


@given(stochastic_matrices())
def test_both_constructions_reproduce_the_matrix(matrix):
    for rep in (independent_random_maps(matrix), coupled_random_maps(matrix)):
        assert np.allclose(rep.matrices()["t"], matrix)


@given(stochastic_matrices())
def test_coupling_uses_few_maps(matrix):
    supports = (matrix > 0).sum(axis=1)
    assert len(coupled_random_maps(matrix).maps("t")) <= 1 + int((supports - 1).sum())


def test_independent_support_is_the_product_of_row_supports():
    matrix = np.array([[0.5, 0.5, 0.0], [0.0, 0.2, 0.8], [1.0, 0.0, 0.0]])
    assert len(independent_random_maps(matrix).maps("t")) == 4
    with pytest.raises(ValueError, match="max_maps"):
        independent_random_maps(matrix, max_maps=3)


def test_explicit_maps_are_checked():
    matrix = np.array([[0.5, 0.5], [0.5, 0.5]])
    random_maps({"t": {(0, 1): 0.5, (1, 0): 0.5}}, matrices=matrix)
    with pytest.raises(ValueError, match="reproduce"):
        random_maps({"t": {(0, 0): 0.5, (0, 1): 0.5}}, matrices=matrix)
    with pytest.raises(ValueError, match="sum"):
        random_maps({"t": {(0, 1): 0.4}})


def test_markov_models_are_read_through_their_state_matrix():
    rep = independent_random_maps(even_process())
    assert rep.states == ("A", "B")
    assert len(rep.maps("t")) == 2


def test_rejects_non_stochastic_input():
    with pytest.raises(ValueError, match="row-stochastic"):
        independent_random_maps(np.array([[0.5, 0.2], [0.0, 1.0]]))


# -- walks ------------------------------------------------------------------------------


@given(stochastic_matrices(), st.integers(0, 6))
def test_walk_marginals_are_matrix_powers(matrix, steps):
    for rep in (independent_random_maps(matrix), coupled_random_maps(matrix)):
        sg = rep.semigroup()
        walk = element_walk(rep, steps)
        assert np.allclose(walk.sum(axis=1), 1.0)
        n = len(matrix)
        reached = np.zeros((n, n))
        for i, element in enumerate(sg):
            reached[np.arange(n), element.images] += walk[steps, i]
        assert np.allclose(reached, np.linalg.matrix_power(matrix, steps))


def test_multi_letter_walks_need_inputs():
    rep = random_maps({"a": {(1, 0): 1.0}, "b": {(0, 0): 1.0}})
    with pytest.raises(ValueError, match="inputs"):
        element_walk(rep, 3)
    walk = element_walk(rep, inputs=["a", "a", "b"])
    sg = rep.semigroup()
    assert sg[int(np.argmax(walk[2]))] == sg.evaluate([])
    assert sg[int(np.argmax(walk[3]))].images == (0, 0)


@given(stochastic_matrices(), st.integers(0, 5))
def test_information_losses_are_nonnegative_and_add_up(matrix, steps):
    info = walk_information(coupled_random_maps(matrix), steps)
    h0 = np.log2(len(matrix))
    assert (info.irreversibility >= -1e-9).all()
    assert (info.dissipation >= -1e-9).all()
    assert np.allclose(info.irreversibility + info.dissipation + info.mutual, h0)


def test_walk_information_matches_dit():
    matrix = np.array([[0.7, 0.3, 0.0], [0.0, 0.4, 0.6], [0.5, 0.0, 0.5]])
    rep = independent_random_maps(matrix)
    info = walk_information(rep, 3)
    p_t = np.linalg.matrix_power(matrix, 3) / 3
    joint = dit.Distribution.from_ndarray(p_t)
    assert info.mutual[3] == pytest.approx(mutual_information(joint, [0], [1]))


def test_permutation_noise_never_irreversible():
    rep = coupled_random_maps(np.array([[0.9, 0.1], [0.1, 0.9]]))
    info = walk_information(rep, 50)
    assert np.allclose(info.irreversibility, 0.0)
    assert info.dissipation[-1] == pytest.approx(1.0, abs=1e-6)


# -- minimal ideal, synchronization, Schutzenberger groups -------------------------------


@given(semigroups())
def test_synchronizing_iff_minimal_rank_one(sg):
    green = sg.green()
    assert sg.is_synchronizing() == (green.minimal_rank() == 1)
    ideal = set(green.minimal_ideal().elements())
    lowest = min(e.rank() for e in sg)
    assert ideal == {i for i, e in enumerate(sg) if e.rank() == lowest}


@given(semigroups())
def test_schutzenberger_group_order_is_h_class_size(sg):
    green = sg.green()
    for members in green.h_classes:
        assert int(green.schutzenberger_group(members[0]).order()) == len(members)
    for e, group in green.maximal_subgroups():
        assert int(green.schutzenberger_group(e).order()) == int(group.order())


def unifilar(alphabet, edges):
    aut = UnifilarAutomaton(input_alphabet=frozenset(alphabet))
    for source, target, _ in edges:
        aut.graph.add_state(source)
        aut.graph.add_state(target)
    for source, target, symbol in edges:
        aut.graph.add_transition(source, target, symbol=symbol)
    return aut


def test_synchronization_agrees_with_reset_threshold():
    cerny = unifilar("ab", [(0, 1, "a"), (1, 2, "a"), (2, 0, "a"), (0, 0, "b"), (1, 1, "b"), (2, 0, "b")])
    assert cerny.transition_semigroup().is_synchronizing()
    assert cerny.reset_threshold() == 4
    flip = unifilar("a", [(0, 1, "a"), (1, 0, "a")])
    assert not flip.transition_semigroup().is_synchronizing()
    assert flip.reset_threshold() == float("inf")


def test_classify_elements_of_t3():
    t3 = TransformationSemigroup({"t": (1, 0, 2), "c": (1, 2, 0), "e": (0, 0, 2)})
    classes = classify_elements(t3)
    assert len(classes["units"]) == 6
    assert classes["resets"] == classes["minimal_ideal"]
    assert len(classes["resets"]) == 3
    assert len(classes["identity"]) == 1
