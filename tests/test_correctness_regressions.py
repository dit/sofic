"""Regression tests for correctness fixes found in the package review."""

import numpy as np
import pytest

from sofic.automata.algorithms import equivalent
from sofic.automata.dfa import DFA
from sofic.examples import even_process, golden_mean
from sofic.exceptions import SoficValidationError
from sofic.generators.markov import MarkovChain
from sofic.generators.mealy import MealyHMM
from sofic.graph import ATTR_MULTIPLICITY, ATTR_SYMBOL
from sofic.shifts.sft import ShiftOfFiniteType
from sofic.shifts.tmc import TopologicalMarkovChain


def test_tmc_multiplicity_counts_in_entropy_and_parry_measure():
    tmc = TopologicalMarkovChain(symbol_alphabet=frozenset({"a"}))
    tmc.graph.add_state("s")
    tmc.graph.add_transition("s", "s", **{ATTR_SYMBOL: "a", ATTR_MULTIPLICITY: 2})
    assert tmc.topological_entropy() == pytest.approx(1.0)

    matrix_tmc = TopologicalMarkovChain.from_adjacency(np.array([[2, 1], [1, 0]]), symbol_alphabet=frozenset("ab"))
    expected = np.log2(max(abs(np.linalg.eigvals(np.array([[2, 1], [1, 0]])))))
    assert matrix_tmc.topological_entropy() == pytest.approx(expected)
    parry = matrix_tmc.parry_measure()
    parry.validate()
    weights = {}
    for t in parry.transitions():
        weights[(t.source, t.target)] = weights.get((t.source, t.target), 0.0) + t.data["prob"]
    for state in parry.states():
        assert sum(w for (s, _), w in weights.items() if s == state) == pytest.approx(1.0)


def test_golden_mean_topological_entropy_is_log2_golden_ratio():
    tmc = TopologicalMarkovChain.from_adjacency(np.array([[1, 1], [1, 0]]), symbol_alphabet=frozenset("01"))
    assert tmc.topological_entropy() == pytest.approx(np.log2((1 + np.sqrt(5)) / 2))


def test_sft_construction_raises_instead_of_truncating():
    forbidden = {tuple("0" * 9)}
    with pytest.raises(SoficValidationError, match="max_states"):
        ShiftOfFiniteType.from_forbidden_words(forbidden, frozenset("01"), max_states=8)


def test_markov_words_of_length_zero_respects_initial_distribution():
    chain = MarkovChain(initial_distribution={"a": 1.0})
    chain.graph.add_state("a")
    chain.graph.add_state("b")
    chain.add_transition("a", "b", 1.0)
    chain.add_transition("b", "a", 1.0)
    assert chain.words_of_length(0) == {(): pytest.approx(1.0)}
    assert chain.words_of_length(2) == {("a", "b"): pytest.approx(1.0)}

    stationary = MarkovChain()
    stationary.graph.add_state("a")
    stationary.graph.add_state("b")
    stationary.add_transition("a", "b", 1.0)
    stationary.add_transition("b", "a", 1.0)
    words = stationary.words_of_length(1)
    assert words == {("a",): pytest.approx(0.5), ("b",): pytest.approx(0.5)}


def test_markov_sample_path_starts_from_initial_distribution():
    chain = MarkovChain(initial_distribution={"b": 1.0})
    chain.graph.add_state("a")
    chain.graph.add_state("b")
    chain.add_transition("a", "b", 1.0)
    chain.add_transition("b", "a", 1.0)
    for seed in range(5):
        assert chain.sample_path(3, rng=np.random.default_rng(seed)) == ["b", "a", "b"]


def test_sample_without_initial_mass_raises():
    hmm = MealyHMM(initial_distribution={"A": 0.0}, observation_alphabet=frozenset({0}))
    hmm.graph.add_state("A")
    hmm.add_transition("A", "A", 0, 1.0)
    with pytest.raises(ValueError, match="no mass"):
        hmm.sample(3, rng=np.random.default_rng(0))


def test_block_entropy_estimates_exact_crypticity_is_cmu_minus_excess_entropy():
    machine = even_process()
    estimates = machine.block_entropy_estimates(3, use_exact=True)
    assert estimates.crypticity == pytest.approx(estimates.statistical_complexity - estimates.excess_entropy)
    assert estimates.crypticity == pytest.approx(machine.crypticity(), abs=1e-9)


def test_log_likelihood_is_in_bits():
    machine = golden_mean(0.5)
    observations = list("0000")
    assert 2.0 ** machine.log_likelihood(observations) == pytest.approx(machine.word_probability(observations))


def _single_symbol_dfa(symbols: set[str]) -> DFA:
    dfa = DFA(input_alphabet=frozenset(symbols), initial_states=frozenset({0}), accepting_states=frozenset({1}))
    dfa.graph.add_state(0)
    dfa.graph.add_state(1)
    for symbol in symbols:
        dfa.add_transition(0, 1, symbol)
    return dfa


def test_equivalent_does_not_hide_differences_outside_the_given_alphabet():
    only_a = _single_symbol_dfa({"a"})
    a_or_b = _single_symbol_dfa({"a", "b"})
    assert not equivalent(only_a, a_or_b, frozenset({"a"}))
    assert equivalent(only_a, _single_symbol_dfa({"a"}))
