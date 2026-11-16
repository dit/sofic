"""Tests for probabilistic finite automata."""

import pytest

from sofic.exceptions import StochasticValidationError
from sofic.generators.pfa import ProbabilisticFiniteAutomaton
from sofic.graph import ATTR_EMISSION, ATTR_PROB


def _pfa() -> ProbabilisticFiniteAutomaton:
    pfa = ProbabilisticFiniteAutomaton(
        initial_distribution={"q0": 1.0},
        output_alphabet=frozenset({"x"}),
    )
    pfa.graph.add_state("q0")
    pfa.add_transition("q0", "q0", "x", 1.0)
    return pfa


def test_validate():
    _pfa().validate()


def test_add_transition_sets_symbol_and_probability():
    edge = next(_pfa().transitions())
    assert edge.data[ATTR_EMISSION] == "x"
    assert edge.data[ATTR_PROB] == pytest.approx(1.0)


def test_negative_probability():
    pfa = _pfa()
    pfa.add_transition("q0", "q0", "x", -0.1)
    with pytest.raises(StochasticValidationError):
        pfa.validate()


def _two_state() -> ProbabilisticFiniteAutomaton:
    pfa = ProbabilisticFiniteAutomaton(initial_distribution={"a": 1.0}, output_alphabet=frozenset({"x", "y"}))
    pfa.add_transition("a", "a", "x", 0.5)
    pfa.add_transition("a", "b", "y", 0.5)
    pfa.add_transition("b", "a", "x", 1.0)
    return pfa


def test_string_probabilities_sum_over_words():
    pfa = _two_state()
    assert pfa.string_probability(()) == pytest.approx(1.0)
    assert pfa.string_probability(("x", "x")) == pytest.approx(0.25)
    assert pfa.string_probability(("y", "x")) == pytest.approx(0.5)
    assert pfa.string_probability(("y", "y")) == pytest.approx(0.0)
    words = pfa.words_of_length(3)
    assert sum(words.values()) == pytest.approx(1.0)
    for word, prob in words.items():
        assert pfa.string_probability(word) == pytest.approx(prob)


def test_unifilarity_and_mealy_conversion():
    pfa = _two_state()
    assert pfa.is_unifilar()
    pfa.add_transition("b", "b", "x", 0.0)
    assert not pfa.is_unifilar()
    mealy = _two_state().to_mealy()
    assert mealy.word_probability(("y", "x")) == pytest.approx(0.5)


def test_sample_follows_the_edges():
    import numpy as np

    pfa = _two_state()
    sample = pfa.sample(200, rng=np.random.default_rng(0))
    assert len(sample) == 200
    assert "yy" not in "".join(sample)
    assert ProbabilisticFiniteAutomaton(initial_distribution={}).sample(5) == []


def test_sample_stops_at_dead_ends():
    import numpy as np

    pfa = ProbabilisticFiniteAutomaton(initial_distribution={"a": 1.0}, output_alphabet=frozenset({"x"}))
    pfa.add_transition("a", "b", "x", 1.0)
    pfa.graph.add_state("b")
    assert pfa.sample(5, rng=np.random.default_rng(0)) == ["x"]
    blocked = ProbabilisticFiniteAutomaton(initial_distribution={"a": 1.0}, output_alphabet=frozenset({"x"}))
    blocked.add_transition("a", "a", "x", 0.0)
    assert blocked.sample(5, rng=np.random.default_rng(0)) == []
    silent = ProbabilisticFiniteAutomaton(initial_distribution={"a": 1.0}, output_alphabet=frozenset({"x"}))
    silent.graph.add_transition("a", "a", **{ATTR_PROB: 1.0})
    assert silent.sample(3, rng=np.random.default_rng(0)) == []
