"""Tests for sample-based ε-machine reconstruction."""

from __future__ import annotations

from collections.abc import Hashable
from itertools import permutations
from typing import Any

import numpy as np
import pytest

from pensive.examples.epsilon_machines import bernoulli, even_process, golden_mean
from pensive.generators.epsilon_inference import cssr, subtree_merge
from pensive.generators.epsilon_machine import EpsilonMachine
from pensive.generators.hmm_inference import sample
from pensive.graph import ATTR_EMISSION, ATTR_PROB


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
    inferred = cssr(observations, alpha=0.01)
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
    inferred = cssr(observations, Lmax=3, alpha=0.001)
    inferred.validate()
    assert len(list(inferred.states())) == 2
    assert _signatures_isomorphic(inferred, oracle, prob_tol=0.1)


def test_cssr_even_process_recovers_two_states(rng: np.random.Generator):
    oracle = even_process(0.5)
    observations, _ = sample(oracle, 12000, rng)
    inferred = cssr(observations, Lmax=4, alpha=0.001)
    inferred.validate()
    assert len(list(inferred.states())) == 2
    assert _signatures_isomorphic(inferred, oracle, prob_tol=0.12)


def test_subtree_merge_golden_mean(rng: np.random.Generator):
    oracle = golden_mean(0.5)
    observations, _ = sample(oracle, 10000, rng)
    inferred = subtree_merge(observations, L=2, delta=0.0)
    inferred.validate()
    assert len(list(inferred.states())) == 2
    assert _signatures_isomorphic(inferred, oracle, prob_tol=0.12)


def test_from_sequence_subtree_dispatch(rng: np.random.Generator):
    oracle = golden_mean(0.5)
    observations, _ = sample(oracle, 6000, rng)
    inferred = EpsilonMachine.from_sequence(observations, method="subtree", L=2)
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
    inferred = cssr(observations, Lmax=3, alpha=0.001)
    inferred.validate()

    idx = inferred.reindex()
    pi = inferred.stationary_distribution()
    initial = np.array([inferred.initial_distribution.get(idx.state(i), 0.0) for i in range(len(idx))])
    assert initial.sum() == pytest.approx(1.0, abs=1e-9)
    assert np.allclose(initial, pi, atol=0.05)


def test_cssr_short_sequence_raises():
    with pytest.raises(ValueError, match="at least two"):
        cssr([0])


def test_subtree_merge_short_sequence_raises():
    with pytest.raises(ValueError, match="at least two"):
        subtree_merge([1], L=1)
