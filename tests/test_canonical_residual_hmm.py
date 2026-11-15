"""Canonical residual HMM: extreme future morphs of a finite ε-machine."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from sofic.examples import bernoulli, even_process, golden_mean
from sofic.examples.epsilon_machines import from_symbol_matrices
from sofic.exceptions import MixedStateExplosionError
from sofic.generators.canonical_residual import canonical_residual_hmm
from sofic.generators.epsilon_machine import EpsilonMachine
from sofic.generators.matrices import symbol_matrices
from sofic.generators.mealy import MealyHMM
from sofic.testing.strategies import epsilon_machines

MAX_LENGTH = 6


def _same_process(left, right, max_length: int = MAX_LENGTH) -> None:
    for n in range(1, max_length + 1):
        a, b = left.words_of_length(n), right.words_of_length(n)
        for word in set(a) | set(b):
            assert a.get(word, 0.0) == pytest.approx(b.get(word, 0.0), abs=1e-9), word


def _hmm(edges: list[tuple[str, str, str, float]]) -> MealyHMM:
    hmm = MealyHMM(observation_alphabet=frozenset(symbol for _s, _t, symbol, _p in edges))
    for source, target, symbol, prob in edges:
        hmm.graph.add_state(source)
        hmm.graph.add_state(target)
        hmm.add_transition(source, target, symbol, prob)
    pi = hmm.stationary_distribution()
    hmm.initial_distribution = dict(zip(hmm.reindex().states, map(float, pi), strict=True))
    hmm.validate()
    return hmm


def _mixing_hmm(q: float = 0.3, r: float = 0.4, split: float = 0.5, s: float = 0.2, t: float = 0.5) -> MealyHMM:
    """Two-state generator whose ε-machine has a third causal state, a mixture of the other two.

    ``2`` and ``3`` synchronize to ``a`` and ``b``; ``0`` is emitted with
    probability ``q`` from both states and keeps the belief; ``1`` is emitted
    only by ``a`` and moves to ``a`` or ``b`` in proportion ``split``. The
    recurrent beliefs are ``a``, ``b`` and ``(split, 1 - split)``.
    """
    return _hmm(
        [
            ("a", "a", "0", q),
            ("a", "a", "1", r * split),
            ("a", "b", "1", r * (1 - split)),
            ("a", "a", "2", (1 - q - r) * s),
            ("a", "b", "3", (1 - q - r) * (1 - s)),
            ("b", "b", "0", q),
            ("b", "a", "2", (1 - q) * t),
            ("b", "b", "3", (1 - q) * (1 - t)),
        ]
    )


def _lohr_ay(p: float = 0.1) -> MealyHMM:
    """Example 3.6 of :cite:`LohrAy2009`: a two-state generator with infinitely many causal states."""
    return _hmm(
        [
            ("0", "0", "0", 1 - 2 * p),
            ("0", "0", "1", p),
            ("0", "1", "1", p),
            ("1", "1", "1", 1 - 2 * p),
            ("1", "0", "0", p),
            ("1", "1", "0", p),
        ]
    )


@st.composite
def weighted_machines(draw, max_states: int = 3) -> EpsilonMachine:
    """Topological ε-machine skeleton with random positive edge probabilities."""
    skeleton = draw(epsilon_machines(max_states=max_states))
    states = list(skeleton.reindex().states)
    matrices = {symbol: np.array(m, dtype=float) for symbol, m in symbol_matrices(skeleton).items()}
    for i in range(len(states)):
        edges = [(symbol, j) for symbol, m in matrices.items() for j in np.flatnonzero(m[i])]
        weights = np.array([draw(st.floats(0.1, 1.0)) for _ in edges])
        for (symbol, j), weight in zip(edges, weights / weights.sum(), strict=True):
            matrices[symbol][i, j] = weight
    return from_symbol_matrices(states, tuple(matrices), matrices)


def test_requires_experimental_flag() -> None:
    with pytest.raises(RuntimeError, match="experimental"):
        canonical_residual_hmm(golden_mean(0.5))


@pytest.mark.parametrize("model", [golden_mean(0.4), even_process(0.3), bernoulli(0.3)], ids=["gm", "even", "iid"])
def test_affinely_independent_morphs_return_the_epsilon_machine(model: EpsilonMachine) -> None:
    residual = canonical_residual_hmm(model, experimental=True)
    assert set(residual.states()) == set(model.states())
    expected = {(t.source, t.target, t.data["emission"]): t.data["prob"] for t in model.transitions()}
    actual = {(t.source, t.target, t.data["emission"]): t.data["prob"] for t in residual.transitions()}
    assert actual.keys() == expected.keys()
    for key, prob in expected.items():
        assert actual[key] == pytest.approx(prob, abs=1e-9)
    _same_process(residual, model)


def test_strictly_smaller_than_epsilon_machine() -> None:
    hmm = _mixing_hmm()
    eps = EpsilonMachine.from_hmm(hmm)
    assert len(list(eps.states())) == 3
    residual = canonical_residual_hmm(eps, experimental=True)
    residual.validate()
    assert len(list(residual.states())) == 2
    assert not residual.is_unifilar()
    _same_process(residual, eps)
    _same_process(residual, hmm)


def test_non_unifilar_input_is_converted() -> None:
    hmm = _mixing_hmm()
    assert len(list(canonical_residual_hmm(hmm, experimental=True).states())) == 2


def test_infinite_epsilon_machine_raises() -> None:
    with pytest.raises(MixedStateExplosionError):
        canonical_residual_hmm(_lohr_ay(), experimental=True, max_states=200)


def test_duplicate_states_are_merged() -> None:
    model = from_symbol_matrices(
        ("A", "B"),
        ("0", "1"),
        {"0": np.array([[0.3, 0.0], [0.0, 0.3]]), "1": np.array([[0.0, 0.7], [0.7, 0.0]])},
    )
    residual = canonical_residual_hmm(model, experimental=True)
    assert len(list(residual.states())) == 1
    _same_process(residual, bernoulli(0.7))


@settings(max_examples=40)
@given(weighted_machines())
def test_random_epsilon_machines(model: EpsilonMachine) -> None:
    residual = canonical_residual_hmm(model, experimental=True)
    residual.validate()
    size = len(list(model.states()))
    assert len(list(residual.states())) <= size
    _same_process(residual, model, max_length=5)
    futures = np.array(
        [
            [model.word_probability(word, start=state) for n in range(size) for word in model.words_of_length(n)]
            for state in model.reindex().states
        ]
    )
    if np.linalg.matrix_rank(futures, tol=1e-6) == size:
        assert set(residual.states()) == set(model.states())


unit = st.floats(0.1, 0.9)


@settings(max_examples=25)
@given(st.floats(0.05, 0.45), st.floats(0.05, 0.45), unit, unit, unit)
def test_mixed_causal_state_is_dropped(q: float, r: float, split: float, s: float, t: float) -> None:
    hmm = _mixing_hmm(q, r, split, s, t)
    eps = EpsilonMachine.from_hmm(hmm)
    assert len(list(eps.states())) == 3
    residual = canonical_residual_hmm(eps, experimental=True)
    assert len(list(residual.states())) == 2
    _same_process(residual, hmm, max_length=4)
