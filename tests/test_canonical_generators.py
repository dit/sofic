"""Stochastic analogues of canonical NFAs as generators, and canonical NFA presentations of sofic shifts."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings

from sofic.automata.algorithms import equivalent
from sofic.examples import even_process, golden_mean
from sofic.examples.epsilon_machines import from_symbol_matrices
from sofic.generators.canonical_generators import atomaton_hmm, canonical_rfsa_hmm, maximized_prime_atomaton_hmm
from sofic.generators.epsilon_machine import EpsilonMachine
from sofic.generators.mealy import MealyHMM
from sofic.generators.minimal_generative_model import _equivalence_error
from sofic.shifts.canonical_presentations import (
    atomaton_presentation,
    canonical_rfsa_presentation,
    distromaton_presentation,
    factor_language_nfa,
    follower_sets_intersection_closed,
    maximized_prime_atomaton_presentation,
    universal_presentation,
)
from sofic.shifts.sofic import SoficShift
from sofic.testing.strategies import sofic_shifts


def _state_entropy(hmm: MealyHMM) -> float:
    p = np.array([v for v in hmm.initial_distribution.values() if v > 0])
    return float(-(p * np.log2(p)).sum())


def _reset_mixture() -> EpsilonMachine:
    """Two hidden states, three causal states (process rank 2); symbols 2 and 3 reset, 1 is emitted only from a."""
    hmm = MealyHMM(observation_alphabet=frozenset("0123"))
    for s, t, x, p in [
        ("a", "a", "0", 0.3), ("a", "a", "1", 0.2), ("a", "b", "1", 0.2), ("a", "a", "2", 0.06), ("a", "b", "3", 0.24),
        ("b", "b", "0", 0.3), ("b", "a", "2", 0.35), ("b", "b", "3", 0.35),
    ]:  # fmt: skip
        hmm.graph.add_state(s)
        hmm.graph.add_state(t)
        hmm.add_transition(s, t, x, p)
    hmm.initial_distribution = dict(zip(hmm.reindex().states, map(float, hmm.stationary_distribution()), strict=True))
    return EpsilonMachine.from_hmm(hmm)


def _paired_reverse_compression() -> EpsilonMachine:
    """A 3-state ε-machine whose reverse machine has 7 states, 3 of them extreme (dataset process n3-t088-d4)."""
    a, b, c = 0.7353692009410088, 0.03138587493847493, 0.145902298340887  # code ((0,1),(2,2),(0,2))
    mats = {0: np.array([[a, 0, 0], [0, 0, b], [c, 0, 0]]), 1: np.array([[0, 1 - a, 0], [0, 0, 1 - b], [0, 0, 1 - c]])}
    machine = from_symbol_matrices((0, 1, 2), (0, 1), mats)
    return machine if isinstance(machine, EpsilonMachine) else EpsilonMachine.from_hmm(machine)


@pytest.mark.parametrize("build", [canonical_rfsa_hmm, maximized_prime_atomaton_hmm, atomaton_hmm])
@pytest.mark.parametrize("machine", [golden_mean(0.4), even_process(), _reset_mixture()])
def test_canonical_generators_generate_the_process(build, machine) -> None:
    hmm = build(machine)
    assert _equivalence_error(hmm, machine) < 1e-10


def test_two_state_epsilon_machines_are_fixed_points() -> None:
    gm = golden_mean(0.4)
    assert len(list(canonical_rfsa_hmm(gm).states())) == 2
    assert _state_entropy(canonical_rfsa_hmm(gm)) == pytest.approx(float(gm.statistical_complexity()))


def test_rfsa_analogue_drops_the_mixture_causal_state() -> None:
    machine = _reset_mixture()
    rfsa = canonical_rfsa_hmm(machine)
    assert len(list(machine.states())) == 3
    assert len(list(rfsa.states())) == 2
    assert not rfsa.is_unifilar()
    assert _state_entropy(rfsa) < float(machine.statistical_complexity())


def test_prime_atomaton_analogue_compresses_the_reverse_machine() -> None:
    machine = _paired_reverse_compression()
    prime = maximized_prime_atomaton_hmm(machine)
    atom = atomaton_hmm(machine)
    assert len(list(atom.states())) > len(list(prime.states())) == 3
    assert _equivalence_error(prime, machine) < 1e-10
    assert _state_entropy(prime) < float(machine.statistical_complexity()) - 0.15  # 1.0994 against C_mu = 1.2809


def test_generator_search_starts_from_the_prime_atomaton_analogue() -> None:
    from sofic.generators.bidirectional_epsilon_machine import BidirectionalEpsilonMachine
    from sofic.generators.generative_search import generative_model_from_bidirectional

    bidir = BidirectionalEpsilonMachine.from_forward(_paired_reverse_compression())
    result = generative_model_from_bidirectional(
        bidir, stages=("machines",), coverage="off", generator_caps=0, rng=np.random.default_rng(0)
    )
    assert result.upper_source == "maximized prime átomaton generator"
    assert result.upper == pytest.approx(_state_entropy(maximized_prime_atomaton_hmm(bidir)), abs=1e-9)
    assert result.reproduction_error < 1e-10


@pytest.mark.parametrize(
    "present",
    [
        canonical_rfsa_presentation,
        maximized_prime_atomaton_presentation,
        atomaton_presentation,
        distromaton_presentation,
        universal_presentation,
    ],
)
@settings(max_examples=25, deadline=None, derandomize=True)
@given(shift=sofic_shifts(max_states=3))
def test_canonical_presentations_present_the_same_shift(present, shift) -> None:
    presented = present(shift)
    assert equivalent(factor_language_nfa(shift), factor_language_nfa(presented))
    if present is not universal_presentation:
        assert len(list(presented.states())) <= 2 ** len(list(shift.trim_transient().states()))


def _presents(shift: SoficShift, states: int, edges) -> bool:
    candidate = SoficShift(symbol_alphabet=shift.symbol_alphabet)
    for state in range(states):
        candidate.graph.add_state(state)
    for source, target, symbol in edges:
        candidate.add_transition(source, target, symbol)
    return equivalent(factor_language_nfa(shift), factor_language_nfa(candidate))


@settings(max_examples=40, deadline=None, derandomize=True)
@given(shift=sofic_shifts(max_states=3))
def test_intersection_closed_follower_sets_make_the_rfsa_presentation_minimal(shift) -> None:
    """Every labeled graph with fewer states than the canonical RFSA presentation presents another shift."""
    if not follower_sets_intersection_closed(shift):
        return
    k = len(list(canonical_rfsa_presentation(shift).states()))
    if not 2 <= k <= 3:
        return
    m = k - 1
    possible = [(s, t, a) for s in range(m) for t in range(m) for a in sorted(shift.symbol_alphabet)]
    for mask in range(1 << len(possible)):
        assert not _presents(shift, m, [e for i, e in enumerate(possible) if mask >> i & 1])


def test_follower_sets_of_the_golden_mean_shift_are_intersection_closed() -> None:
    shift = golden_mean(0.4).to_sofic_shift()
    assert follower_sets_intersection_closed(shift)
    assert len(list(canonical_rfsa_presentation(shift).states())) == 2
    assert equivalent(factor_language_nfa(shift), factor_language_nfa(shift.distromaton_presentation()))
    assert equivalent(factor_language_nfa(shift), factor_language_nfa(shift.universal_presentation()))


def test_shift_methods_and_supports() -> None:
    shift = even_process().to_sofic_shift()
    assert equivalent(factor_language_nfa(shift), factor_language_nfa(shift.canonical_rfsa_presentation()))
    assert equivalent(factor_language_nfa(shift), factor_language_nfa(shift.maximized_prime_atomaton_presentation()))
    # the support of each stochastic analogue presents the same shift
    for build in (canonical_rfsa_hmm, maximized_prime_atomaton_hmm, atomaton_hmm):
        support = build(_reset_mixture()).to_sofic_shift()
        assert equivalent(factor_language_nfa(_reset_mixture().to_sofic_shift()), factor_language_nfa(support))
