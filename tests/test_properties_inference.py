"""Property-based tests for HMM filtering, decoding, EM, and model selection.

Every message, posterior, and likelihood is checked against the brute-force
path-enumeration oracles in :mod:`tests.oracles` on random Mealy HMMs.
"""

from __future__ import annotations

import math
from collections.abc import Hashable
from typing import Any

import numpy as np
import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from sofic.generators.mealy import MealyHMM
from sofic.graph import ATTR_EMISSION, ATTR_PROB
from sofic.inference import hmm as inf
from sofic.inference import learn_epsilon_machine_cssr, learn_epsilon_machine_spectral, learn_epsilon_machine_subtree
from sofic.inference.model_selection import count_free_parameters, score_model
from sofic.testing.strategies import epsilon_machines, mealy_hmms
from tests import oracles

ALPHABET = ("0", "1")
LN2 = math.log(2.0)

small_hmms = mealy_hmms(max_states=3)
observations = st.lists(st.sampled_from(ALPHABET), max_size=5)

# Learners need long samples (~0.5 s each); keep their example budget modest.
learner_settings = settings(max_examples=max(4, settings.default.max_examples // 20))
markov_eps = epsilon_machines(max_states=3).filter(lambda m: m.markov_order() <= 3)
SWAP = {"0": "1", "1": "0"}
LEARNERS = {
    "cssr": (lambda x, alphabet: learn_epsilon_machine_cssr(x, alphabet=alphabet, max_history=4), 1e-12),
    "subtree": (lambda x, alphabet: learn_epsilon_machine_subtree(x, alphabet=alphabet, max_history=4), 1e-12),
    "spectral": (lambda x, alphabet: learn_epsilon_machine_spectral([x], alphabet=alphabet), 1e-5),
}


def order(hmm: MealyHMM) -> list[Hashable]:
    return list(hmm.reindex().states)


def edge_table(hmm: MealyHMM) -> dict[tuple[Hashable, Any, Hashable], float]:
    return {(tr.source, tr.data[ATTR_EMISSION], tr.target): float(tr.data[ATTR_PROB]) for tr in hmm.transitions()}


def with_edges(hmm: MealyHMM, edges: dict[tuple[Hashable, Any, Hashable], float]) -> MealyHMM:
    out = MealyHMM(observation_alphabet=hmm.observation_alphabet, initial_distribution=dict(hmm.initial_distribution))
    for state in hmm.states():
        out.graph.add_state(state)
    for (source, symbol, target), prob in edges.items():
        out.add_transition(source, target, symbol, prob)
    return out


def ln_likelihood(hmm: MealyHMM, obs: list[Any]) -> float:
    return math.log(oracles.word_probability(hmm, obs))


def possible(hmm: MealyHMM, obs: list[Any], *, floor: float = 0.0) -> bool:
    return oracles.word_probability(hmm, obs) > floor


# --------------------------------------------------------------------------- filtering and smoothing


@given(small_hmms, observations)
def test_forward_backward_match_oracle(hmm, obs):
    states = order(hmm)
    np.testing.assert_allclose(inf.forward(hmm, obs), oracles.forward(hmm, obs, states=states), atol=1e-12)
    np.testing.assert_allclose(inf.backward(hmm, obs), oracles.backward(hmm, obs, states=states), atol=1e-12)


@given(small_hmms, observations)
def test_forward_backward_product_is_constant_likelihood(hmm, obs):
    alpha, beta = inf.forward(hmm, obs), inf.backward(hmm, obs)
    np.testing.assert_allclose((alpha * beta).sum(axis=1), oracles.word_probability(hmm, obs), atol=1e-12)


@given(small_hmms, observations)
def test_normalized_messages_are_rescaled_raw_messages(hmm, obs):
    assume(possible(hmm, obs))
    alpha = inf.forward(hmm, obs)
    np.testing.assert_allclose(inf.forward(hmm, obs, normalize=True), alpha / alpha.sum(axis=1, keepdims=True))
    beta = inf.backward(hmm, obs)
    np.testing.assert_allclose(inf.backward(hmm, obs, normalize=True), beta / beta.sum(axis=1, keepdims=True))


def test_normalized_backward_final_row_sums_to_one():
    """Regression: ``backward(normalize=True)`` left ``beta[n]`` as all ones."""
    hmm = MealyHMM(observation_alphabet=frozenset(ALPHABET), initial_distribution={0: 1.0})
    for state in (0, 1, 2):
        hmm.graph.add_state(state)
        hmm.add_transition(state, (state + 1) % 3, "0", 0.5)
        hmm.add_transition(state, state, "1", 0.5)
    for obs in ([], ["0"], ["0", "1"]):
        np.testing.assert_allclose(inf.backward(hmm, obs, normalize=True).sum(axis=1), 1.0)


@given(small_hmms, observations)
def test_smooth_matches_oracle_and_rows_sum_to_one(hmm, obs):
    gamma = inf.smooth(hmm, obs)
    expected = oracles.gamma(hmm, obs, states=order(hmm))
    np.testing.assert_allclose(gamma, expected, atol=1e-10)
    if possible(hmm, obs):
        np.testing.assert_allclose(gamma.sum(axis=1), 1.0, atol=1e-12)
        assert np.all(gamma >= 0.0)
    else:
        assert not gamma.any()


@given(small_hmms, observations)
def test_two_slice_marginals_match_oracle_and_marginalize_to_gamma(hmm, obs):
    xi = inf.two_slice_marginals(hmm, obs)
    states = order(hmm)
    assert xi.shape == (len(obs), len(states), len(states))
    np.testing.assert_allclose(xi, oracles.xi(hmm, obs, states=states), atol=1e-10)
    gamma = inf.smooth(hmm, obs)
    np.testing.assert_allclose(xi.sum(axis=2), gamma[:-1], atol=1e-10)
    np.testing.assert_allclose(xi.sum(axis=1), gamma[1:], atol=1e-10)


@given(small_hmms, observations)
def test_log_likelihood_is_log2_oracle_probability(hmm, obs):
    prob = oracles.word_probability(hmm, obs)
    ll = inf.log_likelihood(hmm, obs)
    if prob > 0.0:
        assert ll == pytest.approx(math.log2(prob), abs=1e-9)
    else:
        assert ll == -math.inf


@given(small_hmms, st.lists(observations, min_size=1, max_size=3))
def test_log_likelihood_of_iid_sequences_is_additive(hmm, seqs):
    assume(all(possible(hmm, seq) for seq in seqs))
    scores = score_model(hmm, seqs)
    assert scores.log_likelihood == pytest.approx(sum(inf.log_likelihood(hmm, seq) for seq in seqs), abs=1e-9)


# --------------------------------------------------------------------------- decoding


@given(small_hmms, observations)
def test_viterbi_achieves_oracle_maximum(hmm, obs):
    path = inf.viterbi(hmm, obs)
    best, argmaxes = oracles.viterbi_paths(hmm, obs)
    if best == 0.0:
        assert path == []
        return
    assert len(path) == len(obs) + 1
    assert oracles.path_probability(hmm, path, obs) == pytest.approx(best, rel=1e-9)
    assert tuple(path) in argmaxes


# --------------------------------------------------------------------------- score and information


@given(small_hmms, st.lists(st.sampled_from(ALPHABET), min_size=1, max_size=5))
def test_score_matches_finite_differences(hmm, obs):
    assume(possible(hmm, obs, floor=1e-8))
    edges = edge_table(hmm)
    gradient = inf.score(hmm, obs)
    assert set(gradient) == set(edges)
    step = 1e-6
    for key, prob in edges.items():
        up = with_edges(hmm, {**edges, key: prob + step})
        down = with_edges(hmm, {**edges, key: prob - step})
        numeric = (ln_likelihood(up, obs) - ln_likelihood(down, obs)) / (2 * step)
        assert gradient[key] == pytest.approx(numeric, rel=1e-5, abs=1e-5), key


@given(small_hmms, st.lists(st.sampled_from(ALPHABET), min_size=1, max_size=4))
def test_observed_information_matches_finite_difference_hessian(hmm, obs):
    assume(possible(hmm, obs, floor=1e-6))
    edges = edge_table(hmm)
    labels = inf.free_parameter_labels(hmm)
    assume(labels)
    references = {}
    for source in {label[0] for label in labels}:
        out = sorted((k for k in edges if k[0] == source), key=lambda k: (str(k[1]), k[2]))
        references[source] = out[-1]

    def shifted(deltas: dict[int, float]) -> float:
        table = dict(edges)
        for index, delta in deltas.items():
            label = labels[index]
            table[label] += delta
            table[references[label[0]]] -= delta
        return ln_likelihood(with_edges(hmm, table), obs)

    h = 1e-4
    size = len(labels)
    numeric = np.zeros((size, size))
    for a in range(size):
        for b in range(size):
            if a == b:
                numeric[a, a] = (shifted({a: h}) - 2 * shifted({}) + shifted({a: -h})) / h**2
            else:
                numeric[a, b] = (
                    shifted({a: h, b: h}) - shifted({a: h, b: -h}) - shifted({a: -h, b: h}) + shifted({a: -h, b: -h})
                ) / (4 * h**2)
    np.testing.assert_allclose(inf.observed_information(hmm, obs), -numeric, rtol=1e-3, atol=1e-3)


# --------------------------------------------------------------------------- EM


@given(small_hmms, st.lists(st.lists(st.sampled_from(ALPHABET), min_size=1, max_size=8), min_size=1, max_size=3))
def test_baum_welch_trace_is_non_decreasing(hmm, seqs):
    assume(all(possible(hmm, seq) for seq in seqs))
    fitted, trace = inf.baum_welch(hmm, seqs, max_iter=15, tol=0.0)
    assert trace
    assert trace[0] == pytest.approx(sum(inf.log_likelihood(hmm, seq) for seq in seqs), abs=1e-9)
    assert np.all(np.diff(trace) >= -1e-9)
    final = sum(inf.log_likelihood(fitted, seq) for seq in seqs)
    assert final >= trace[-1] - 1e-9
    assert set(edge_table(fitted)) <= set(edge_table(hmm))
    fitted.validate()


# --------------------------------------------------------------------------- model selection


@given(small_hmms, st.lists(st.sampled_from(ALPHABET), min_size=1, max_size=6), st.booleans())
def test_information_criteria_formulas(hmm, obs, include_initial):
    scores = score_model(hmm, obs, include_initial=include_initial)
    k = count_free_parameters(hmm, include_initial=include_initial)
    n = len(obs)
    assert (scores.num_parameters, scores.num_observations) == (k, n)
    assert k == sum(max(0, len(list(hmm.graph.out_transitions(s))) - 1) for s in hmm.states()) + (
        max(0, sum(1 for p in hmm.initial_distribution.values() if p > 0) - 1) if include_initial else 0
    )
    if not possible(hmm, obs):
        assert scores.log_likelihood == -math.inf
        assert scores.aic == scores.bic == scores.mdl == math.inf
        return
    ln_l = LN2 * inf.log_likelihood(hmm, obs)
    assert scores.log_likelihood == pytest.approx(ln_l / LN2, abs=1e-9)
    assert scores.aic == pytest.approx(2 * k - 2 * ln_l, abs=1e-9)
    assert scores.bic == pytest.approx(k * math.log(n) - 2 * ln_l, abs=1e-9)
    assert scores.mdl == pytest.approx(0.5 * k * math.log2(n) - ln_l / LN2, abs=1e-9)
    if n - k - 1 > 0:
        assert scores.aicc == pytest.approx(scores.aic + 2 * k * (k + 1) / (n - k - 1), abs=1e-9)
    else:
        assert scores.aicc == math.inf
    for criterion in ("aic", "aicc", "bic", "mdl"):
        assert scores.value(criterion) == getattr(scores, criterion)


# --------------------------------------------------------------------------- metamorphic


@given(small_hmms, observations, st.data())
def test_state_relabeling_permutes_posteriors(hmm, obs, data):
    states = order(hmm)
    names = data.draw(st.permutations([f"q{i}" for i in range(len(states))]))
    mapping = dict(zip(states, names, strict=True))
    relabeled = MealyHMM(
        observation_alphabet=hmm.observation_alphabet,
        initial_distribution={mapping[s]: p for s, p in hmm.initial_distribution.items()},
    )
    for state in states:
        relabeled.graph.add_state(mapping[state])
    for (source, symbol, target), prob in edge_table(hmm).items():
        relabeled.add_transition(mapping[source], mapping[target], symbol, prob)
    ll, ll2 = inf.log_likelihood(hmm, obs), inf.log_likelihood(relabeled, obs)
    assert ll == ll2 == -math.inf or ll == pytest.approx(ll2, abs=1e-12)
    position = [order(relabeled).index(mapping[s]) for s in states]
    np.testing.assert_allclose(inf.smooth(relabeled, obs)[:, position], inf.smooth(hmm, obs), atol=1e-12)
    path = inf.viterbi(relabeled, obs)
    best, _ = oracles.viterbi_paths(hmm, obs)
    if best > 0.0:
        inverse = {v: k for k, v in mapping.items()}
        assert oracles.path_probability(hmm, [inverse[s] for s in path], obs) == pytest.approx(best, rel=1e-9)


# --------------------------------------------------------------------------- learners


def words4(machine: MealyHMM, rename: dict[str, str] | None = None) -> dict[tuple[str, ...], float]:
    dist = machine.word_probabilities(4, sparse=False)
    return dist if rename is None else {tuple(rename[x] for x in w): p for w, p in dist.items()}


def total_variation(left: dict, right: dict) -> float:
    return 0.5 * sum(abs(left.get(w, 0.0) - right.get(w, 0.0)) for w in set(left) | set(right))


@pytest.mark.parametrize("name", sorted(LEARNERS))
@learner_settings
@given(machine=markov_eps)
def test_learners_recover_small_markov_machines(name, machine):
    learn, _ = LEARNERS[name]
    sample, _ = machine.sample(20_000, np.random.default_rng(0))
    learned = learn(sample, ALPHABET)
    assert total_variation(words4(learned), words4(machine)) < 0.05
    assert len(list(learned.states())) == len(list(machine.states()))


@pytest.mark.parametrize("name", sorted(LEARNERS))
@learner_settings
@given(machine=markov_eps)
def test_learners_are_equivariant_under_symbol_relabeling(name, machine):
    learn, atol = LEARNERS[name]
    sample, _ = machine.sample(20_000, np.random.default_rng(1))
    learned = learn(sample, ALPHABET)
    relabeled = learn([SWAP[x] for x in sample], ALPHABET[::-1])
    assert len(list(relabeled.states())) == len(list(learned.states()))
    assert total_variation(words4(relabeled, SWAP), words4(learned)) < atol
    assert relabeled.entropy_rate() == pytest.approx(learned.entropy_rate(), abs=atol)
    assert relabeled.statistical_complexity() == pytest.approx(learned.statistical_complexity(), abs=atol)
