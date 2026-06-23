"""Tests for finite-word distributions."""

import pytest

from pensive.generators.markov import MarkovChain
from pensive.generators.moore import MooreHMM
from pensive.generators.nmachine import NMachine
from pensive.generators.pfa import ProbabilisticFiniteAutomaton
from pensive.generators.quasi_realization import QuasiRealization
from pensive.graph import ATTR_EMISSION, ATTR_EMISSION_DIST, ATTR_PROB, ATTR_QUASIPROB


def _mealy_like_pfa() -> ProbabilisticFiniteAutomaton:
    pfa = ProbabilisticFiniteAutomaton(
        initial_distribution={"q0": 1.0},
        output_alphabet=frozenset({"0", "1"}),
    )
    pfa.graph.add_state("q0")
    pfa.graph.add_transition("q0", "q0", **{ATTR_PROB: 0.25, ATTR_EMISSION: "0"})
    pfa.graph.add_transition("q0", "q0", **{ATTR_PROB: 0.75, ATTR_EMISSION: "1"})
    return pfa


def _moore() -> MooreHMM:
    hmm = MooreHMM(
        initial_distribution={"q0": 1.0},
        observation_alphabet=frozenset({"0", "1"}),
    )
    hmm.graph.add_state("q0", **{ATTR_EMISSION_DIST: {"0": 0.25, "1": 0.75}})
    hmm.graph.add_transition("q0", "q0", **{ATTR_PROB: 1.0})
    return hmm


def test_hmm_words_of_length_include_probabilities():
    hmm = _mealy_like_pfa().to_mealy()
    assert hmm.words_of_length(2) == {
        ("0", "0"): pytest.approx(0.0625),
        ("0", "1"): pytest.approx(0.1875),
        ("1", "0"): pytest.approx(0.1875),
        ("1", "1"): pytest.approx(0.5625),
    }


def test_moore_words_of_length_include_probabilities():
    assert _moore().words_of_length(2) == {
        ("0", "0"): pytest.approx(0.0625),
        ("0", "1"): pytest.approx(0.1875),
        ("1", "0"): pytest.approx(0.1875),
        ("1", "1"): pytest.approx(0.5625),
    }


def test_pfa_words_of_length_include_probabilities():
    assert _mealy_like_pfa().words_of_length(1) == {
        ("0",): pytest.approx(0.25),
        ("1",): pytest.approx(0.75),
    }


def test_quasi_words_of_length_include_signed_probabilities():
    nm = NMachine(
        initial_quasidistribution={"q0": 1.0},
        observation_alphabet=frozenset({"0", "1"}),
    )
    nm.graph.add_state("q0")
    nm.graph.add_transition("q0", "q0", **{ATTR_QUASIPROB: 1.25, ATTR_EMISSION: "0"})
    nm.graph.add_transition("q0", "q0", **{ATTR_QUASIPROB: -0.25, ATTR_EMISSION: "1"})
    assert nm.words_of_length(1) == {("0",): pytest.approx(1.25), ("1",): pytest.approx(-0.25)}


def test_quasi_realization_words_of_length():
    import numpy as np

    qr = QuasiRealization(
        pi=np.array([1.0]),
        tau=np.array([1.0]),
        symbol_maps={"0": np.array([[0.25]]), "1": np.array([[0.75]])},
    )
    assert qr.words_of_length(1) == {("0",): pytest.approx(0.25), ("1",): pytest.approx(0.75)}


def test_markov_words_of_length_are_visible_paths():
    chain = MarkovChain(initial_distribution={"A": 1.0})
    chain.graph.add_state("A")
    chain.graph.add_state("B")
    chain.graph.add_transition("A", "A", **{ATTR_PROB: 0.25})
    chain.graph.add_transition("A", "B", **{ATTR_PROB: 0.75})
    chain.graph.add_transition("B", "B", **{ATTR_PROB: 1.0})
    assert chain.words_of_length(2) == {("A", "A"): pytest.approx(0.25), ("A", "B"): pytest.approx(0.75)}
