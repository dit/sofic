"""Tests adapted from cmpy's Bayesian inference modules."""

from __future__ import annotations

import numpy as np
import pytest

from pensive.examples.processes import SNS, Even
from pensive.inference.bayesian import (
    BayesianInferenceError,
    InferEM,
    InferMC,
    ModelComparisonMC,
    PathCountEM,
    WordCountsMC,
    pretty_symbol,
    pretty_word,
)


def test_pretty_symbol_and_word():
    assert pretty_symbol((11, 22)) == "11:22"
    assert pretty_symbol("12") == "12"
    assert pretty_word(("0", "1", "1")) == "0,1,1"


def test_word_counts_markov_chain():
    counts = WordCountsMC(list("01010"), order=1)
    assert counts.get_word_count(("0", "1")) == 2
    assert counts.get_word_count(("1", "0")) == 2
    assert counts.get_word_count(("0", "*")) == 2
    counts.set_word_count(("0", "0"), 1)
    assert counts.get_word_count(("0", "*")) == 3


def test_infer_mc_transition_estimates_and_evidence():
    posterior = InferMC(["0", "1"], list("01010"), order=1)
    assert posterior.transition_probability_mle(("0",), "1") == pytest.approx((1.0, 0.0))
    prob, var = posterior.transition_probability_pme(("0",), "1")
    assert prob == pytest.approx(0.75)
    assert var == pytest.approx(0.0375)
    assert posterior.log_evidence() == pytest.approx(-2.1972245773362196)


def test_infer_mc_generates_valid_mealy_hmm():
    posterior = InferMC(["0", "1"], list("01010"), order=1)
    machine = posterior.generate_mealy_hmm("PME")
    machine.validate()
    assert len(list(machine.transitions())) == 4


def test_model_comparison_mc_probabilities_normalize():
    comparison = ModelComparisonMC(["0", "1"], list("01010"), 0, 2)
    probs = comparison.model_probabilities()
    assert set(probs) == {0, 1, 2}
    assert sum(probs.values()) == pytest.approx(1.0)


def test_path_count_em_rejects_nonunifilar_topology():
    with pytest.raises(BayesianInferenceError):
        PathCountEM(SNS(), list("01"))


def test_path_count_em_counts_even_process():
    data = list("1111101100")
    counts = PathCountEM(Even(), data)
    assert counts.get_possible_start_nodes() == ["B"]
    assert counts.get_edge_count("B", ("A", "0")) == 3
    assert counts.get_edge_count("B", ("A", "1")) == 3
    assert counts.get_edge_count("B", ("B", "1")) == 4
    assert counts.get_node_count("B", "A") == 6
    assert counts.get_node_count("B", "B") == 4


def test_infer_em_start_marginalization_and_sample():
    posterior = InferEM(Even(), list("1111101100"))
    assert posterior.start_node_probabilities() == {"B": pytest.approx(1.0)}
    assert posterior.log_evidence() < 0
    start, machine = posterior.generate_sample(rng=np.random.default_rng(0))
    assert start == "B"
    machine.validate()


def test_markov_pymc_model_smoke():
    pm = pytest.importorskip("pymc")
    posterior = InferMC(["0", "1"], list("01010"), order=1)
    model = posterior.as_pymc_model()
    assert isinstance(model, pm.Model)
