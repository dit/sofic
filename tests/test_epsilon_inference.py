"""Tests for sample-based ε-machine reconstruction."""

from __future__ import annotations

from collections.abc import Hashable
from itertools import permutations
from typing import Any

import numpy as np
import pytest

from sofic.examples.epsilon_machines import bernoulli, even_process, golden_mean
from sofic.generators.epsilon_machine import EpsilonMachine
from sofic.generators.sampling import sample
from sofic.graph import ATTR_EMISSION, ATTR_PROB
from sofic.inference.cssr import learn_epsilon_machine_cssr, learn_epsilon_machine_subtree
from sofic.inference.spectral import learn_epsilon_machine_spectral


def _transition_signature(hmm: EpsilonMachine) -> dict[Hashable, tuple[tuple[Any, Hashable, float], ...]]:
    signature: dict[Hashable, tuple[tuple[Any, Hashable, float], ...]] = {}
    for state in hmm.states():
        edges = []
        for transition in hmm.graph.out_transitions(state):
            emission = transition.data.get(ATTR_EMISSION)
            prob = float(transition.data.get(ATTR_PROB, 0.0))
            edges.append((emission, transition.target, round(prob, 6)))
        signature[state] = tuple(sorted(edges, key=lambda item: (repr(item[0]), repr(item[1]))))
    return signature


def _signatures_isomorphic(
    left: EpsilonMachine,
    right: EpsilonMachine,
    *,
    prob_tol: float = 0.08,
) -> bool:
    left_sig = _transition_signature(left)
    right_sig = _transition_signature(right)
    if len(left_sig) != len(right_sig):
        return False
    left_states = list(left_sig)
    right_states = list(right_sig)
    if len(left_states) <= 6:
        for perm in permutations(right_states):
            mapping = dict(zip(left_states, perm, strict=True))
            if _mapping_preserves_structure(left_sig, right_sig, mapping, prob_tol=prob_tol):
                return True
        return False
    return len(left_sig) == len(right_sig)


def _mapping_preserves_structure(
    left_sig: dict[Hashable, tuple[tuple[Any, Hashable, float], ...]],
    right_sig: dict[Hashable, tuple[tuple[Any, Hashable, float], ...]],
    mapping: dict[Hashable, Hashable],
    *,
    prob_tol: float,
) -> bool:
    for left_state, edges in left_sig.items():
        right_state = mapping[left_state]
        mapped = tuple(
            sorted(
                ((emission, mapping.get(target, target), round(prob, 6)) for emission, target, prob in edges),
                key=lambda item: (repr(item[0]), repr(item[1])),
            )
        )
        right_edges = right_sig[right_state]
        if len(mapped) != len(right_edges):
            return False
        for left_edge, right_edge in zip(mapped, right_edges, strict=True):
            if left_edge[0] != right_edge[0]:
                return False
            if left_edge[1] != right_edge[1]:
                return False
            if abs(left_edge[2] - right_edge[2]) > prob_tol:
                return False
    return True


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(0)


def test_cssr_bernoulli_single_state(rng: np.random.Generator):
    oracle = bernoulli(0.3)
    observations, _ = sample(oracle, 500, rng)
    inferred = learn_epsilon_machine_cssr(observations, alpha=0.01)
    inferred.validate()
    assert len(list(inferred.states())) == 1


def test_from_sequence_cssr_dispatch(rng: np.random.Generator):
    oracle = bernoulli(0.4)
    observations, _ = sample(oracle, 400, rng)
    inferred = EpsilonMachine.from_sequence(observations, method="cssr", alpha=0.01)
    inferred.validate()
    assert len(list(inferred.states())) == 1


def test_cssr_golden_mean_recovers_two_states(rng: np.random.Generator):
    oracle = golden_mean(0.5)
    observations, _ = sample(oracle, 8000, rng)
    inferred = learn_epsilon_machine_cssr(observations, max_history=3, alpha=0.001)
    inferred.validate()
    assert len(list(inferred.states())) == 2
    assert _signatures_isomorphic(inferred, oracle, prob_tol=0.1)


def test_cssr_even_process_recovers_two_states(rng: np.random.Generator):
    oracle = even_process(0.5)
    observations, _ = sample(oracle, 12000, rng)
    inferred = learn_epsilon_machine_cssr(observations, max_history=4, alpha=0.001)
    inferred.validate()
    assert len(list(inferred.states())) == 2
    assert _signatures_isomorphic(inferred, oracle, prob_tol=0.12)


def test_subtree_merge_golden_mean(rng: np.random.Generator):
    oracle = golden_mean(0.5)
    observations, _ = sample(oracle, 10000, rng)
    inferred = learn_epsilon_machine_subtree(observations, max_history=2, delta=0.0)
    inferred.validate()
    assert len(list(inferred.states())) == 2
    assert _signatures_isomorphic(inferred, oracle, prob_tol=0.12)


def test_from_sequence_subtree_dispatch(rng: np.random.Generator):
    oracle = golden_mean(0.5)
    observations, _ = sample(oracle, 6000, rng)
    inferred = EpsilonMachine.from_sequence(observations, method="subtree", max_history=2)
    inferred.validate()
    assert len(list(inferred.states())) >= 2


def test_cssr_initial_distribution_matches_occupation(rng: np.random.Generator):
    """The reconstructed initial law should approximate the occupation/stationary law.

    Counting each nested history suffix (the old behavior) over-weighted short-history
    states; counting the longest matching suffix once per step recovers the stationary
    occupation of the inferred causal states.
    """
    oracle = golden_mean(0.5)
    observations, _ = sample(oracle, 8000, rng)
    inferred = learn_epsilon_machine_cssr(observations, max_history=3, alpha=0.001)
    inferred.validate()

    idx = inferred.reindex()
    pi = inferred.stationary_distribution()
    initial = np.array([inferred.initial_distribution.get(idx.state(i), 0.0) for i in range(len(idx))])
    assert initial.sum() == pytest.approx(1.0, abs=1e-9)
    assert np.allclose(initial, pi, atol=0.05)


def test_cssr_short_sequence_raises():
    with pytest.raises(ValueError, match="at least two"):
        learn_epsilon_machine_cssr([0])


def test_subtree_merge_short_sequence_raises():
    with pytest.raises(ValueError, match="at least two"):
        learn_epsilon_machine_subtree([1], max_history=1)


def test_spectral_bernoulli_single_state():
    oracle = bernoulli(0.3)
    alphabet = sorted(oracle.observation_alphabet, key=repr)
    inferred = learn_epsilon_machine_spectral(
        word_probability=oracle.word_probability, alphabet=alphabet, prefix_length=2, rank=1
    )
    inferred.validate()
    assert len(list(inferred.states())) == 1
    assert inferred.entropy_rate() == pytest.approx(oracle.entropy_rate(), abs=1e-9)


def test_spectral_golden_mean_recovers_two_states():
    oracle = golden_mean(0.5)
    alphabet = sorted(oracle.observation_alphabet, key=repr)
    inferred = learn_epsilon_machine_spectral(
        word_probability=oracle.word_probability, alphabet=alphabet, prefix_length=3, rank=2
    )
    inferred.validate()
    assert len(list(inferred.states())) == 2
    assert inferred.entropy_rate() == pytest.approx(oracle.entropy_rate(), abs=1e-6)
    assert inferred.statistical_complexity() == pytest.approx(oracle.statistical_complexity(), abs=1e-6)
    assert _signatures_isomorphic(inferred, oracle, prob_tol=0.05)


def test_spectral_even_process_recovers_two_states():
    oracle = even_process(0.5)
    alphabet = sorted(oracle.observation_alphabet, key=repr)
    inferred = learn_epsilon_machine_spectral(
        word_probability=oracle.word_probability, alphabet=alphabet, prefix_length=3, rank=2
    )
    inferred.validate()
    assert len(list(inferred.states())) == 2
    assert inferred.entropy_rate() == pytest.approx(oracle.entropy_rate(), abs=1e-6)
    assert inferred.statistical_complexity() == pytest.approx(oracle.statistical_complexity(), abs=1e-6)
    assert _signatures_isomorphic(inferred, oracle, prob_tol=0.05)


def test_from_sequence_spectral_dispatch(rng: np.random.Generator):
    oracle = bernoulli(0.4)
    observations, _ = sample(oracle, 400, rng)
    inferred = EpsilonMachine.from_sequence(observations, method="spectral", prefix_length=2, rank=1)
    inferred.validate()
    assert len(list(inferred.states())) == 1


def test_from_sequence_unknown_method():
    with pytest.raises(ValueError, match="unknown inference method"):
        EpsilonMachine.from_sequence([0, 1, 0], method="nsd")


@pytest.mark.parametrize(
    ("name", "max_history"),
    [("even_process", 3), ("even_process", 5), ("golden_mean_forbid_00", 3), ("nemo_process", 4), ("rk_gm", 5)],
)
def test_cssr_recovers_synchronizable_processes(name: str, max_history: int):
    """Regression: appended (not prepended) suffixes and untruncated successors
    dropped edges, so these raised StochasticValidationError or returned h_mu = 0."""
    from sofic import examples

    oracle = examples.rk_gm(5, 3) if name == "rk_gm" else getattr(examples, name)()
    observations, _ = sample(oracle, 20000, np.random.default_rng(5))
    inferred = learn_epsilon_machine_cssr(observations, max_history=max_history, alpha=0.001)
    inferred.validate()
    assert inferred.is_unifilar()
    assert len(list(inferred.states())) == len(list(oracle.states()))
    assert inferred.entropy_rate() == pytest.approx(oracle.entropy_rate(), abs=0.02)


def test_cssr_even_process_ignores_truncated_nonsynchronizing_suffix():
    """At max_history = 3 the successor of ``011`` on ``1`` truncates to the ambiguous ``111``."""
    oracle = even_process(0.5)
    observations, _ = sample(oracle, 20000, np.random.default_rng(5))
    inferred = learn_epsilon_machine_cssr(observations, max_history=3, alpha=0.01)
    assert _signatures_isomorphic(inferred, oracle, prob_tol=0.03)


def test_cssr_default_lmax_does_not_oversplit():
    for oracle, n_states in [(even_process(0.5), 2), (bernoulli(0.3), 1)]:
        observations, _ = sample(oracle, 20000, np.random.default_rng(7))
        inferred = learn_epsilon_machine_cssr(observations)
        inferred.validate()
        assert len(list(inferred.states())) == n_states


def test_cssr_short_lmax_still_emits_every_symbol():
    """max_history below the Markov order cannot recover rk_gm(5, 3), but must not collapse to a trap state."""
    from sofic.examples import processes

    observations, _ = sample(processes.rk_gm(5, 3), 20000, np.random.default_rng(5))
    inferred = learn_epsilon_machine_cssr(observations, max_history=3, alpha=0.001)
    inferred.validate()
    assert {t.data[ATTR_EMISSION] for t in inferred.transitions()} == {"0", "1"}
    assert inferred.entropy_rate() > 0.0


def test_cssr_non_synchronizable_process_returns_valid_machine():
    from sofic.examples import alternating_biased_coins

    oracle = alternating_biased_coins(0.25, 0.75)
    observations, _ = sample(oracle, 20000, np.random.default_rng(5))
    inferred = learn_epsilon_machine_cssr(observations, max_history=4, alpha=0.001)
    inferred.validate()
    assert inferred.entropy_rate() >= oracle.entropy_rate() - 0.02


@pytest.mark.parametrize(
    ("name", "L", "n_states"), [("even_process", 3, 2), ("golden_mean_forbid_00", 2, 2), ("rk_gm", 5, 8)]
)
def test_subtree_merge_default_delta_recovers_process(name: str, L: int, n_states: int):
    """Regression: the default delta = 0 compared sampled morphs to within 1e-3 and
    successors were never truncated, so this raised StochasticValidationError."""
    from sofic import examples

    oracle = examples.rk_gm(5, 3) if name == "rk_gm" else getattr(examples, name)()
    observations, _ = sample(oracle, 20000, np.random.default_rng(5))
    inferred = learn_epsilon_machine_subtree(observations, max_history=L)
    inferred.validate()
    assert len(list(inferred.states())) == n_states
    assert inferred.entropy_rate() == pytest.approx(oracle.entropy_rate(), abs=0.02)


def _has_markov_order_selection() -> bool:
    import dit.inference

    return hasattr(dit.inference, "select_markov_order")


needs_markov_order = pytest.mark.skipif(
    not _has_markov_order_selection(), reason="needs dit.inference.select_markov_order"
)


@needs_markov_order
def test_suggest_lmax_markov_sources():
    from sofic.examples import processes
    from sofic.inference.cssr import suggest_max_history

    observations, _ = sample(golden_mean(0.5), 4000, np.random.default_rng(1))
    assert suggest_max_history(observations) == 1
    observations, _ = sample(processes.rk_gm(3, 2), 20000, np.random.default_rng(2))
    assert suggest_max_history(observations, method="bic") == 3


@needs_markov_order
def test_suggest_lmax_grows_for_even_process():
    """The even process has infinite Markov order, so the suggestion grows with data."""
    from sofic.inference.cssr import suggest_max_history

    short, _ = sample(even_process(0.5), 300, np.random.default_rng(3))
    long, _ = sample(even_process(0.5), 30000, np.random.default_rng(3))
    assert suggest_max_history(long, method="bic") > suggest_max_history(short, method="bic")


@needs_markov_order
def test_cssr_auto_lmax_golden_mean(rng: np.random.Generator):
    observations, _ = sample(golden_mean(0.5), 8000, rng)
    inferred = learn_epsilon_machine_cssr(observations, max_history="auto", alpha=0.001)
    assert len(list(inferred.states())) == 2
    assert _signatures_isomorphic(inferred, golden_mean(0.5), prob_tol=0.1)


def test_exact_morph_test_small_counts():
    """With tiny counts the exact test is calibrated where the chi-squared limit is not."""
    from sofic.inference.cssr import SuffixCounts, morphs_differ

    rng = np.random.default_rng(4)
    rejections = {"g": 0, "exact": 0}
    trials = 300
    for _ in range(trials):
        counts = SuffixCounts(alphabet=(0, 1, 2))
        for history in ((0,), (1,)):
            for symbol in rng.choice(3, size=6, p=[0.8, 0.1, 0.1]):
                counts.next_counts[history][int(symbol)] += 1
                counts.history_counts[history] += 1
        for test in rejections:
            rejections[test] += morphs_differ(counts, {(0,)}, {(1,)}, alpha=0.05, test=test)
    assert rejections["exact"] / trials <= 0.08
    assert rejections["exact"] <= rejections["g"]


def test_exact_morph_test_is_deterministic(rng: np.random.Generator):
    observations, _ = sample(golden_mean(0.5), 3000, rng)
    first = learn_epsilon_machine_cssr(observations, max_history=3, alpha=0.01, test="exact")
    second = learn_epsilon_machine_cssr(observations, max_history=3, alpha=0.01, test="exact")
    assert _transition_signature(first) == _transition_signature(second)
    assert len(list(first.states())) == 2


def test_cssr_bonferroni_reduces_spurious_states():
    """An i.i.d. source with a long max_history: the corrected test keeps a single state."""
    observations, _ = sample(bernoulli(0.3), 3000, np.random.default_rng(6))
    inferred = learn_epsilon_machine_cssr(observations, max_history=6, alpha=0.05, correction="bonferroni")
    assert len(list(inferred.states())) == 1
    with pytest.raises(ValueError, match="unknown correction"):
        learn_epsilon_machine_cssr(observations, max_history=2, correction="holm")


@pytest.mark.parametrize("kwargs", [{"test": "exact"}, {"correction": "bonferroni"}, {"alpha": 0.001}])
def test_subtree_merge_options_golden_mean(kwargs):
    observations, _ = sample(golden_mean(0.5), 6000, np.random.default_rng(7))
    inferred = learn_epsilon_machine_subtree(observations, max_history=2, **kwargs)
    assert len(list(inferred.states())) == 2
    with pytest.raises(ValueError, match="unknown correction"):
        learn_epsilon_machine_subtree(observations, max_history=2, correction="holm")


@needs_markov_order
def test_subtree_merge_auto_depth():
    observations, _ = sample(golden_mean(0.5), 6000, np.random.default_rng(8))
    assert len(list(learn_epsilon_machine_subtree(observations, max_history="auto").states())) == 2
