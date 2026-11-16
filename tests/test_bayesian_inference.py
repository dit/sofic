"""Tests adapted from cmpy's Bayesian inference modules."""

from __future__ import annotations

import numpy as np
import pytest

from sofic.examples import even_process
from sofic.examples.processes import sns
from sofic.graph import ATTR_PROB
from sofic.inference.bayesian import (
    BayesianInferenceError,
    DirichletPriorMC,
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
        PathCountEM(sns(), list("01"))


def test_path_count_em_counts_even_process():
    data = list("1111101100")
    counts = PathCountEM(even_process(), data)
    assert counts.get_possible_start_nodes() == ["B"]
    assert counts.get_edge_count("B", ("A", "0")) == 3
    assert counts.get_edge_count("B", ("A", "1")) == 3
    assert counts.get_edge_count("B", ("B", "1")) == 4
    assert counts.get_node_count("B", "A") == 6
    assert counts.get_node_count("B", "B") == 4


def test_infer_em_start_marginalization_and_sample():
    posterior = InferEM(even_process(), list("1111101100"))
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


def test_markov_pymc_model_likelihoods():
    pytest.importorskip("pymc")
    posterior = InferMC(["0", "1"], list("01010"), order=1)
    assert "sequence_loglik" in posterior.as_pymc_model().named_vars
    counts = posterior.as_pymc_model(observed_as_counts=True)
    assert "counts" in counts.named_vars
    assert "sequence_loglik" not in counts.named_vars


def test_epsilon_machine_pymc_model():
    pytest.importorskip("pymc")
    from sofic.inference.bayesian.pymc_backend import _safe_name

    posterior = InferEM(even_process(), list("1111101100"))
    default = posterior.as_pymc_model()
    explicit = posterior.as_pymc_model(start_node="B")
    assert set(default.named_vars) == set(explicit.named_vars)
    # Only A has a choice of edges in the even process.
    assert _safe_name("theta", "A") == "theta_A"
    assert {"theta_A", "sequence_loglik_A"} <= set(default.named_vars)
    with pytest.raises(ValueError, match="not viable"):
        posterior.as_pymc_model(start_node="A")


def test_dirichlet_prior_uniform_and_explicit_alphas():
    prior = DirichletPriorMC(["0", "1"], order=1)
    assert prior.get_alpha(("0", "1")) == 1.0
    assert prior.get_alpha(("0", "*")) == 2.0
    assert prior.get_alpha(("0",)) is None
    assert "alpha(0 -> *) = 2.0" in str(prior)
    prior.set_alpha(("0", "1"), 3)
    prior.set_alpha(("0", "0"), 2)
    prior.set_alpha(("0", "1"), 5)
    assert not prior.uniform
    assert prior.get_alpha(("0", "1")) == 5.0
    assert prior.get_alpha(("0", "*")) == 7.0
    assert prior.get_alpha(("1", "0")) is None


def test_dirichlet_prior_random_rows_sum_to_root():
    prior = DirichletPriorMC(["0", "1", "2"], order=1)
    prior.create_random_prior(1, 6, rng=np.random.default_rng(0))
    for context in "012":
        cells = [prior.get_alpha((context, symbol)) for symbol in "012"]
        assert all(alpha is not None and 1 <= alpha < 6 for alpha in cells)
        assert prior.get_alpha((context, "*")) == sum(cells)


def test_markov_posterior_prior_types():
    assert not InferMC(["0", "1"], list("0101"), order=1, prior_type="random").prior.uniform
    with pytest.raises(BayesianInferenceError, match="prior type"):
        InferMC(["0", "1"], list("0101"), order=1, prior_type="jeffreys")


def test_markov_posterior_unseen_context_and_added_counts():
    posterior = InferMC(["0", "1"], list("0000"), order=1)
    assert posterior.transition_probability_mle(("1",), "0") == (0.0, 0.0)
    posterior.add_counts_from(list("0101"))
    assert posterior.transition_probability_mle(("1",), "0")[0] == pytest.approx(1.0)


def test_markov_posterior_missing_alphas_raise():
    posterior = InferMC(["0", "1"], list("0101"), order=1)
    posterior.prior = DirichletPriorMC(["0", "1"], order=1, uniform=False)
    with pytest.raises(BayesianInferenceError):
        posterior.transition_probability_pme(("0",), "1")
    with pytest.raises(BayesianInferenceError):
        posterior.posterior_alpha_matrix()
    with pytest.raises(BayesianInferenceError):
        posterior.log_evidence()
    posterior.prior.set_alpha(("0", "0"), 1.0)
    with pytest.raises(BayesianInferenceError):
        posterior.log_evidence()


def test_markov_posterior_mean_matrix_matches_pme():
    posterior = InferMC(["0", "1"], list("0110100110010110"), order=2)
    mean = posterior.posterior_mean_matrix()
    np.testing.assert_allclose(mean.sum(axis=1), 1.0)
    for i, context in enumerate(posterior.contexts):
        for j, symbol in enumerate(posterior.alphabet):
            assert mean[i, j] == pytest.approx(posterior.transition_probability_pme(context, symbol)[0])


def test_markov_posterior_entropy_rate_concentrates_on_fair_coin():
    rng = np.random.default_rng(0)
    small = InferMC(["0", "1"], list(rng.choice(["0", "1"], size=100)), order=0)
    large = InferMC(["0", "1"], list(rng.choice(["0", "1"], size=10_000)), order=0)
    assert large.average_relative_entropy_plus_entropy_rate() == pytest.approx(1.0, abs=1e-3)
    assert (
        0.0 < large.variance_relative_entropy_plus_entropy_rate() < small.variance_relative_entropy_plus_entropy_rate()
    )


def test_markov_posterior_reports():
    posterior = InferMC(["0", "1"], list("01010"), order=1)
    mle = list(posterior.transition_probability_mle_iter())
    pme = list(posterior.transition_probability_pme_iter())
    assert len(mle) == len(pme) == 4
    assert ("0", "1", 1.0, 0.0) in mle
    assert posterior.transition_probability_mle_string().count("Pr(") == 4
    assert posterior.transition_probability_pme_string().count("StdDev") == 4
    assert posterior.counts_string()
    assert posterior.prior_string() == str(posterior.prior)


def test_markov_posterior_mle_machine_and_validation():
    posterior = InferMC(["0", "1"], list("0110100110010110"), order=1)
    machine = posterior.generate_mealy_hmm("MLE")
    machine.validate()
    assert set(machine.states()) == {("0",), ("1",)}
    with pytest.raises(BayesianInferenceError, match="threshold"):
        posterior.generate_mealy_hmm(threshold=1.5)
    with pytest.raises(BayesianInferenceError, match="method"):
        posterior.generate_mealy_hmm("MAP")


def test_order_zero_markov_machine_has_one_state():
    machine = InferMC(["0", "1"], list("0001"), order=0).generate_mealy_hmm()
    assert list(machine.states()) == ["A"]
    assert {t.target for t in machine.transitions()} == {"A"}


def test_markov_posterior_samples_are_valid_machines():
    posterior = InferMC(["0", "1"], list("0110100110010110"), order=1)
    machines = list(posterior.sample_mealy_hmms(3, rng=np.random.default_rng(1)))
    assert len(machines) == 3
    for machine in machines:
        machine.validate()
        for state in machine.states():
            assert sum(t.data[ATTR_PROB] for t in machine.graph.out_transitions(state)) == pytest.approx(1.0)
    with pytest.raises(BayesianInferenceError, match="threshold"):
        next(iter(posterior.sample_mealy_hmms(threshold=-0.1)))
