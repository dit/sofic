"""Tests for strong lumpability (state aggregation) of chains and HMMs."""

import numpy as np
import pytest

from sofic.exceptions import LumpabilityError
from sofic.generators.lumping import is_lumpable, lump, normalize_partition
from sofic.generators.markov import MarkovChain
from sofic.generators.mealy import MealyHMM
from sofic.generators.moore import MooreHMM
from sofic.properties import transition_matrix


def _symmetric_chain() -> MarkovChain:
    """Three-state chain where B and C are interchangeable (so {A},{B,C} lumps)."""
    mc = MarkovChain(initial_distribution={"A": 1.0})
    for state in ("A", "B", "C"):
        mc.graph.add_state(state)
    mc.add_transition("A", "B", 0.5)
    mc.add_transition("A", "C", 0.5)
    mc.add_transition("B", "A", 0.5)
    mc.add_transition("B", "B", 0.25)
    mc.add_transition("B", "C", 0.25)
    mc.add_transition("C", "A", 0.5)
    mc.add_transition("C", "B", 0.5)
    return mc


def _mealy() -> MealyHMM:
    """Mealy HMM where {A},{B,C} is strongly lumpable."""
    m = MealyHMM(initial_distribution={"A": 1.0}, observation_alphabet=frozenset({0, 1}))
    for state in ("A", "B", "C"):
        m.graph.add_state(state)
    m.add_transition("A", "B", 0, 0.5)
    m.add_transition("A", "C", 1, 0.5)
    m.add_transition("B", "A", 0, 0.5)
    m.add_transition("B", "B", 1, 0.25)
    m.add_transition("B", "C", 1, 0.25)
    m.add_transition("C", "A", 0, 0.5)
    m.add_transition("C", "B", 1, 0.5)
    return m


def _unifilar_mealy() -> MealyHMM:
    """Unifilar Mealy HMM whose {A},{B,C} lumping is also unifilar."""
    m = MealyHMM(initial_distribution={"A": 1.0}, observation_alphabet=frozenset({0, 1}))
    for state in ("A", "B", "C"):
        m.graph.add_state(state)
    m.add_transition("A", "B", 0, 0.5)
    m.add_transition("A", "C", 1, 0.5)
    m.add_transition("B", "A", 0, 0.5)
    m.add_transition("B", "B", 1, 0.5)
    m.add_transition("C", "A", 0, 0.5)
    m.add_transition("C", "C", 1, 0.5)
    return m


def _moore() -> MooreHMM:
    """Moore HMM where {A},{B,C} is strongly lumpable (B, C share an emission)."""
    mo = MooreHMM(initial_distribution={"A": 1.0}, observation_alphabet=frozenset({"x", "y"}))
    for state in ("A", "B", "C"):
        mo.graph.add_state(state)
    mo.set_emission_distribution("A", {"x": 1.0})
    mo.set_emission_distribution("B", {"y": 1.0})
    mo.set_emission_distribution("C", {"y": 1.0})
    mo.add_transition("A", "B", 0.5)
    mo.add_transition("A", "C", 0.5)
    mo.add_transition("B", "A", 0.5)
    mo.add_transition("B", "B", 0.25)
    mo.add_transition("B", "C", 0.25)
    mo.add_transition("C", "A", 0.5)
    mo.add_transition("C", "B", 0.5)
    return mo


def _words_match(left, right, length: int) -> bool:
    a = left.words_of_length(length)
    b = right.words_of_length(length)
    return all(np.isclose(a.get(key, 0.0), b.get(key, 0.0)) for key in set(a) | set(b))


# --- MarkovChain ---------------------------------------------------------


def test_markov_is_lumpable():
    assert _symmetric_chain().is_lumpable([{"A"}, {"B", "C"}])


def test_markov_lump_transition_matrix_and_initial():
    lumped = lump(_symmetric_chain(), [{"A"}, {"B", "C"}])
    lumped.validate()
    matrix, order = transition_matrix(lumped)
    assert order == ["A", "B+C"]
    assert np.allclose(matrix, [[0.0, 1.0], [0.5, 0.5]])
    assert lumped.initial_distribution == {"A": 1.0}


def test_markov_not_lumpable_returns_false_and_raises():
    chain = _symmetric_chain()
    assert not chain.is_lumpable([{"A", "B"}, {"C"}])
    with pytest.raises(LumpabilityError):
        chain.lump([{"A", "B"}, {"C"}])


def test_markov_lump_check_false_builds_validating_model():
    lumped = _symmetric_chain().lump([{"A", "B"}, {"C"}], check=False)
    lumped.validate()
    matrix, _order = transition_matrix(lumped)
    assert np.allclose(matrix.sum(axis=1), 1.0)


def test_single_block_partition_always_lumps():
    chain = _symmetric_chain()
    assert chain.is_lumpable([{"A", "B", "C"}])
    lumped = chain.lump([{"A", "B", "C"}])
    matrix, _order = transition_matrix(lumped)
    assert np.allclose(matrix, [[1.0]])


def test_singleton_partition_preserves_labels_and_dynamics():
    chain = _symmetric_chain()
    lumped = chain.lump([{"A"}, {"B"}, {"C"}])
    assert set(lumped.states()) == {"A", "B", "C"}
    original, order = transition_matrix(chain)
    recovered, _ = transition_matrix(lumped, states=order)
    assert np.allclose(original, recovered)


# --- partition validation ------------------------------------------------


def test_partition_mapping_input_matches_block_input():
    chain = _symmetric_chain()
    blocks = normalize_partition(chain, {"A": 0, "B": 1, "C": 1})
    assert blocks == [frozenset({"A"}), frozenset({"B", "C"})]


def test_partition_overlap_raises():
    with pytest.raises(ValueError, match="overlap"):
        normalize_partition(_symmetric_chain(), [{"A", "B"}, {"B", "C"}])


def test_partition_incomplete_raises():
    with pytest.raises(ValueError, match="does not cover"):
        normalize_partition(_symmetric_chain(), [{"A"}, {"B"}])


def test_partition_unknown_state_raises():
    with pytest.raises(ValueError, match="unknown"):
        normalize_partition(_symmetric_chain(), [{"A"}, {"B", "C", "Z"}])


def test_colliding_labels_raise():
    with pytest.raises(ValueError, match="not distinct"):
        _symmetric_chain().lump([{"A"}, {"B", "C"}], labels=lambda block: "same")


def test_custom_labels_mapping():
    lumped = _symmetric_chain().lump([{"A"}, {"B", "C"}], labels={frozenset({"B", "C"}): "BC"})
    assert set(lumped.states()) == {"A", "BC"}


def test_unsupported_type_raises():
    from sofic.automata.nfa import NFA

    nfa = NFA(initial_states={"q0"}, accepting_states={"q0"})
    nfa.graph.add_state("q0")
    with pytest.raises(TypeError):
        is_lumpable(nfa, [{"q0"}])


# --- MealyHMM ------------------------------------------------------------


def test_mealy_lump_preserves_process():
    mealy = _mealy()
    assert mealy.is_lumpable([{"A"}, {"B", "C"}])
    lumped = mealy.lump([{"A"}, {"B", "C"}])
    assert isinstance(lumped, MealyHMM)
    assert set(lumped.states()) == {"A", "B+C"}
    for length in (1, 2, 3, 4):
        assert _words_match(mealy, lumped, length)


def test_mealy_not_lumpable():
    mealy = _mealy()
    # A alone against B,C merged with A breaks the per-symbol block masses.
    assert not mealy.is_lumpable([{"A", "B"}, {"C"}])


def test_epsilon_machine_lumps_to_plain_mealy():
    from sofic.examples import golden_mean

    eps = golden_mean(0.5)
    partition = [{state} for state in eps.states()]
    lumped = eps.lump(partition)
    assert type(lumped) is MealyHMM
    assert _words_match(eps, lumped, 3)


# --- MooreHMM ------------------------------------------------------------


def test_moore_lump_preserves_process():
    moore = _moore()
    assert moore.is_lumpable([{"A"}, {"B", "C"}])
    lumped = moore.lump([{"A"}, {"B", "C"}])
    assert isinstance(lumped, MooreHMM)
    assert set(lumped.states()) == {"A", "B+C"}
    for length in (1, 2, 3):
        assert _words_match(moore, lumped, length)


def test_moore_differing_emissions_not_lumpable():
    moore = _moore()
    moore.set_emission_distribution("C", {"x": 0.5, "y": 0.5})
    assert not moore.is_lumpable([{"A"}, {"B", "C"}])
    with pytest.raises(LumpabilityError):
        moore.lump([{"A"}, {"B", "C"}])


# --- dit-backed invariants -----------------------------------------------


def test_hmm_entropy_rate_preserved_under_lumping():
    pytest.importorskip("dit")
    mealy = _unifilar_mealy()
    lumped = mealy.lump([{"A"}, {"B", "C"}])
    assert mealy.is_unifilar()
    assert lumped.is_unifilar()
    assert np.isclose(mealy.entropy_rate(), lumped.entropy_rate())
