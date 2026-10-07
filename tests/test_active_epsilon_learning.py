"""Tests for L*-style active learning of ε-machines."""

from __future__ import annotations

from fractions import Fraction

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from sofic.examples import butterfly_process, even_process, golden_mean, nemo_process, restricted_golden_mean
from sofic.examples.epsilon_machines import from_symbol_matrices
from sofic.exceptions import MixedStateExplosionError
from sofic.generators.epsilon_machine import EpsilonMachine
from sofic.generators.topological_epsilon_enumeration import iter_topological_epsilon_machines
from sofic.graph import ATTR_EMISSION, ATTR_PROB
from sofic.inference.active import ProbabilityOracle, ProcessOracle, learn_epsilon_machine_active
from sofic.testing.strategies import epsilon_machines


def R(numerator, denominator):
    """Exact rational; skips the calling test when sympy is not installed."""
    return pytest.importorskip("sympy").Rational(numerator, denominator)


def _n_states(machine):
    return len(list(machine.states()))


def _alphabet(machine):
    return sorted(machine.observation_alphabet, key=repr)


def _assert_recovers(learned, target):
    assert isinstance(learned, EpsilonMachine)
    assert _n_states(learned) == _n_states(target)
    assert learned.is_equal_process(target)
    assert learned.statistical_complexity() == pytest.approx(target.statistical_complexity(), abs=1e-9)
    assert learned.entropy_rate() == pytest.approx(target.entropy_rate(), abs=1e-9)


def _with_probabilities(machine, rng):
    """A new ε-machine on ``machine``'s topology with Dirichlet-random edge probabilities."""
    states = list(machine.states())
    symbols = _alphabet(machine)
    index = {state: i for i, state in enumerate(states)}
    matrices = {symbol: np.zeros((len(states), len(states))) for symbol in symbols}
    for state in states:
        out = list(machine.graph.out_transitions(state))
        for transition, prob in zip(out, rng.dirichlet(np.full(len(out), 2.0)), strict=True):
            matrices[transition.data[ATTR_EMISSION]][index[state], index[transition.target]] = prob
    return from_symbol_matrices(states, symbols, matrices)


def _synchronizing_word(machine):
    """Shortest word whose subset-construction image is one state."""
    symbols = _alphabet(machine)
    start = frozenset(machine.states())
    seen, queue = {start}, [(start, ())]
    for subset, word in queue:
        if len(subset) == 1:
            return word
        for symbol in symbols:
            image = frozenset(
                t.target for s in subset for t in machine.graph.out_transitions(s) if t.data[ATTR_EMISSION] == symbol
            )
            if image and image not in seen:
                seen.add(image)
                queue.append((image, (*word, symbol)))
    raise AssertionError("no synchronizing word")


def _exact(states, symbols, matrices):
    return from_symbol_matrices(states, symbols, {k: np.array(v, dtype=object) for k, v in matrices.items()})


def _exact_even():
    return _exact(("A", "B"), ("0", "1"), {"0": [[R(1, 3), 0], [0, 0]], "1": [[0, R(2, 3)], [1, 0]]})


def _infinite_transients(exact=True):
    """``1`` fixes both states with different probabilities, ``0`` swaps them, ``2`` synchronizes to ``A``.

    Along ``1^n`` the mixed states never repeat, so the mixed-state presentation is infinite.
    """
    q = R(1, 4) if exact else 0.25
    a, b = (R(1, 2), R(1, 4)) if exact else (0.5, 0.25)
    matrices = {"0": [[0, q], [q, 0]], "1": [[a, 0], [0, b]], "2": [[1 - a - q, 0], [1 - b - q, 0]]}
    if exact:
        return _exact(("A", "B"), ("0", "1", "2"), matrices)
    return from_symbol_matrices(("A", "B"), ("0", "1", "2"), {k: np.array(v, dtype=float) for k, v in matrices.items()})


class _ProbabilityOnly:
    def __init__(self, target):
        self._oracle = ProcessOracle(target)

    def word_probability(self, word):
        return self._oracle.word_probability(word)


class _FractionOracle:
    def __init__(self, target):
        self._oracle = ProcessOracle(target)

    def word_probability(self, word):
        value = self._oracle.word_probability(word)
        return Fraction(int(value.p), int(value.q))

    def equivalent(self, machine):
        return self._oracle.equivalent(machine)


# --- named processes ------------------------------------------------------------


@pytest.mark.parametrize(
    "target",
    [golden_mean(0.3), even_process(0.4), nemo_process(0.3, 0.6), butterfly_process(), restricted_golden_mean(3)],
    ids=["golden_mean", "even", "nemo", "butterfly", "rgm3"],
)
def test_recovers_named_processes(target):
    oracle = ProcessOracle(target)
    learned = learn_epsilon_machine_active(oracle, _alphabet(target))
    _assert_recovers(learned, target)
    assert oracle.equivalence_queries <= _n_states(target) + 2


def test_learned_machine_is_unifilar_with_stationary_start():
    learned = learn_epsilon_machine_active(ProcessOracle(nemo_process(0.3, 0.6)), ("0", "1"))
    learned.validate()
    pi = learned.stationary_distribution()
    start = [learned.initial_distribution[state] for state in learned.reindex().states]
    assert np.allclose(start, pi)


def test_presentation_passed_to_oracle_starts_at_seed_row():
    seen = []

    class Recording(ProcessOracle):
        def equivalent(self, machine, history=()):
            seen.append(machine)
            return super().equivalent(machine, history)

    target = even_process(0.5)
    learn_epsilon_machine_active(Recording(target), ("0", "1"))
    presentation = seen[-1]
    assert presentation.initial_distribution == {"s0": 1}
    assert presentation.is_equal_process(target)
    assert _n_states(presentation) == 4


# --- random ε-machines ------------------------------------------------------------


@pytest.mark.parametrize(("k", "n"), [(2, 1), (2, 2), (2, 3), (3, 2)])
def test_recovers_random_enumerated_epsilon_machines(k, n):
    rng = np.random.default_rng(1000 * k + n)
    symbols = tuple("abc"[:k])
    for count, topology in enumerate(iter_topological_epsilon_machines(k, n, alphabet=symbols)):
        if count >= 25:
            break
        target = _with_probabilities(topology, rng)
        oracle = ProcessOracle(target)
        learned = learn_epsilon_machine_active(oracle, symbols, history=_synchronizing_word(target))
        _assert_recovers(learned, target)
        assert oracle.equivalence_queries <= n + 1


@settings(max_examples=25, deadline=None)
@given(epsilon_machines(max_states=3), st.integers(min_value=0, max_value=2**32 - 1))
def test_recovers_random_strategy_epsilon_machines(topology, seed):
    target = _with_probabilities(topology, np.random.default_rng(seed))
    learned = learn_epsilon_machine_active(
        ProcessOracle(target), _alphabet(target), history=_synchronizing_word(target)
    )
    _assert_recovers(learned, target)


# --- exact arithmetic ------------------------------------------------------------


def test_exact_sympy_oracle_gives_exact_machine():
    sp = pytest.importorskip("sympy")
    target = _exact_even()
    learned = learn_epsilon_machine_active(ProcessOracle(target), ("0", "1"))
    probs = sorted((t.data[ATTR_EMISSION], t.data[ATTR_PROB]) for t in learned.transitions())
    assert probs == [("0", R(1, 3)), ("1", R(2, 3)), ("1", 1)]
    assert sorted(learned.initial_distribution.values()) == [R(2, 5), R(3, 5)]
    assert sp.simplify(learned.statistical_complexity() - target.statistical_complexity()) == 0
    assert ProcessOracle(target).equivalent(learned) is None


def test_fraction_oracle_gives_exact_machine():
    learned = learn_epsilon_machine_active(_FractionOracle(_exact_even()), ("0", "1"))
    assert sorted(t.data[ATTR_PROB] for t in learned.transitions()) == [R(1, 3), R(2, 3), 1]


def test_infinite_transients_explode_unless_seeded():
    target = _infinite_transients()
    with pytest.raises(MixedStateExplosionError):
        learn_epsilon_machine_active(ProcessOracle(target), ("0", "1", "2"), max_states=30)
    oracle = ProcessOracle(target)
    learned = learn_epsilon_machine_active(oracle, ("0", "1", "2"), history=("2",))
    assert _n_states(learned) == 2
    assert ProcessOracle(target).equivalent(learned) is None
    assert oracle.equivalence_queries == 1


def test_float_tolerance_truncates_converging_transients():
    target = _infinite_transients(exact=False)
    learned = learn_epsilon_machine_active(ProcessOracle(target), ("0", "1", "2"))
    _assert_recovers(learned, target)


# --- queries, termination, tolerance ------------------------------------------------


def test_query_counts_are_polynomial():
    counts = []
    for k in range(1, 9):
        oracle = ProcessOracle(restricted_golden_mean(k))
        learned = learn_epsilon_machine_active(oracle, ("0", "1"))
        n = k + 1
        assert _n_states(learned) == n
        assert oracle.equivalence_queries <= 2
        assert oracle.probability_queries <= 4 * n**2
        counts.append(oracle.probability_queries)
    assert counts == sorted(counts)


def test_probability_only_oracle_uses_bounded_search():
    target = nemo_process(0.3, 0.6)
    learned = learn_epsilon_machine_active(_ProbabilityOnly(target), ("0", "1"))
    _assert_recovers(learned, target)
    assert isinstance(_ProbabilityOnly(target), ProbabilityOracle)


def test_max_rounds_is_enforced():
    with pytest.raises(RuntimeError, match="max_rounds"):
        learn_epsilon_machine_active(ProcessOracle(nemo_process(0.3, 0.6)), ("0", "1"), max_rounds=1)


def test_max_states_is_enforced():
    with pytest.raises(MixedStateExplosionError):
        learn_epsilon_machine_active(ProcessOracle(butterfly_process()), _alphabet(butterfly_process()), max_states=3)


def test_zero_probability_history_is_rejected():
    with pytest.raises(ValueError, match="zero probability"):
        learn_epsilon_machine_active(ProcessOracle(golden_mean(0.5)), ("0", "1"), history=("1", "1"))


def _near_iid(delta):
    return from_symbol_matrices(
        ("A", "B"),
        ("0", "1"),
        {"0": np.array([[0.5, 0.0], [0.5 + delta, 0.0]]), "1": np.array([[0.0, 0.5], [0.0, 0.5 - delta]])},
    )


def test_tight_tolerance_resolves_close_states():
    target = _near_iid(0.01)
    _assert_recovers(learn_epsilon_machine_active(ProcessOracle(target), ("0", "1"), tol=1e-9), target)


def test_loose_tolerance_merges_close_states():
    target = _near_iid(0.01)
    learned = learn_epsilon_machine_active(ProcessOracle(target), ("0", "1"), tol=0.05)
    assert _n_states(learned) == 1
    assert not learned.is_equal_process(target)
    assert learned.entropy_rate() == pytest.approx(target.entropy_rate(), abs=1e-3)
