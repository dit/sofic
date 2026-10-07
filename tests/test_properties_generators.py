"""Property-based and metamorphic tests for stochastic generators.

Word distributions, block entropies, and process equivalence are checked
against the brute-force path-enumeration oracles in :mod:`tests.oracles`.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Hashable
from itertools import product
from typing import Any

import numpy as np
import pytest
from hypothesis import assume, given, note, settings
from hypothesis import strategies as st

from sofic.exceptions import MixedStateExplosionError
from sofic.generators.epsilon_machine import EpsilonMachine
from sofic.generators.lumping import is_lumpable, lump
from sofic.generators.mealy import MealyHMM
from sofic.generators.mixed_state_construction import build_mixed_state_presentation
from sofic.graph import ATTR_EMISSION, ATTR_PROB
from sofic.testing.strategies import epsilon_machines, mealy_hmms
from tests import oracles

ALPHABET = ("0", "1")
TOL = 1e-9

small_hmms = mealy_hmms(max_states=3)
small_eps = epsilon_machines(max_states=3)
# Bidirectional quantities (E, anatomy, reversal) need a finite reverse machine.
reversible_eps = small_eps.filter(lambda m: m.reverse_is_finite())
# Bidirectional constructions cost ~50 ms each; scale their example budget with the profile.
heavy = settings(max_examples=max(10, settings.default.max_examples // 4))


# --------------------------------------------------------------------------- helpers


def rebuild(
    hmm: MealyHMM,
    *,
    state_map: Callable[[Hashable], Hashable] = lambda s: s,
    symbol_map: Callable[[Any], Any] = lambda x: x,
    cls: type[MealyHMM] | None = None,
    initial: dict[Hashable, float] | None = None,
) -> MealyHMM:
    """Copy ``hmm`` with states and symbols renamed through the given maps."""
    cls = type(hmm) if cls is None else cls
    source_initial = hmm.initial_distribution if initial is None else initial
    out = cls(
        observation_alphabet=frozenset(symbol_map(x) for x in hmm.observation_alphabet),
        initial_distribution={state_map(s): float(p) for s, p in source_initial.items()},
    )
    for state in hmm.states():
        out.graph.add_state(state_map(state))
    for tr in hmm.transitions():
        out.add_transition(
            state_map(tr.source), state_map(tr.target), symbol_map(tr.data[ATTR_EMISSION]), float(tr.data[ATTR_PROB])
        )
    return out


def with_stationary_start(hmm: MealyHMM) -> MealyHMM:
    pi = hmm.stationary_distribution()
    states = hmm.reindex().states
    return rebuild(hmm, initial={s: float(p) for s, p in zip(states, pi, strict=True) if p > 0})


def add_unreachable_state(hmm: MealyHMM) -> MealyHMM:
    out = rebuild(hmm)
    out.graph.add_state("dead")
    out.add_transition("dead", "dead", "0", 0.5)
    out.add_transition("dead", next(iter(hmm.states())), "1", 0.5)
    return out


def add_unreachable_copy(hmm: MealyHMM) -> MealyHMM:
    out = rebuild(hmm)
    for state in hmm.states():
        out.graph.add_state(("copy", state))
    for tr in hmm.transitions():
        out.add_transition(("copy", tr.source), ("copy", tr.target), tr.data[ATTR_EMISSION], float(tr.data[ATTR_PROB]))
    return out


def split_states(hmm: MealyHMM, weights: list[float]) -> MealyHMM:
    """Split every state ``s`` into ``(s, 0)``/``(s, 1)``; ``{(s,0),(s,1)}`` is lumpable."""
    states = list(hmm.states())
    w = {s: weights[i % len(weights)] for i, s in enumerate(states)}
    out = MealyHMM(
        observation_alphabet=hmm.observation_alphabet,
        initial_distribution={
            (s, k): float(p) * (w[s] if k == 0 else 1 - w[s])
            for s, p in hmm.initial_distribution.items()
            for k in (0, 1)
        },
    )
    for s in states:
        for k in (0, 1):
            out.graph.add_state((s, k))
    for tr in hmm.transitions():
        p = float(tr.data[ATTR_PROB])
        wt = w[tr.target]
        for k in (0, 1):
            out.add_transition((tr.source, k), (tr.target, 0), tr.data[ATTR_EMISSION], p * wt)
            out.add_transition((tr.source, k), (tr.target, 1), tr.data[ATTR_EMISSION], p * (1 - wt))
    return out


def word_dist(hmm: MealyHMM, length: int) -> dict[tuple[Any, ...], float]:
    return hmm.word_probabilities(length, sparse=False)


def assert_same_dist(left: dict, right: dict, *, atol: float = TOL) -> None:
    for word in set(left) | set(right):
        assert left.get(word, 0.0) == pytest.approx(right.get(word, 0.0), abs=atol), word


def oracle_agrees(a: MealyHMM, b: MealyHMM, *, atol: float = 1e-9) -> bool:
    """Brute-force process equality: words up to length ``n_a + n_b - 1`` suffice."""
    depth = len(list(a.states())) + len(list(b.states())) - 1
    for length in range(depth + 1):
        da, db = oracles.word_distribution(a, length, ALPHABET), oracles.word_distribution(b, length, ALPHABET)
        if any(abs(da[w] - db[w]) > atol for w in da):
            return False
    return True


def perturb(hmm: MealyHMM, data: st.DataObject) -> MealyHMM:
    """Move probability mass between two out-edges of a state with at least two."""
    branching = [s for s in hmm.states() if len(list(hmm.graph.out_transitions(s))) >= 2]
    assume(branching)
    state = data.draw(st.sampled_from(sorted(branching)))
    edges = list(hmm.graph.out_transitions(state))
    i, j = data.draw(st.permutations(range(len(edges))))[:2]
    eps = float(edges[i].data[ATTR_PROB]) / 2
    out = MealyHMM(observation_alphabet=hmm.observation_alphabet, initial_distribution=dict(hmm.initial_distribution))
    for s in hmm.states():
        out.graph.add_state(s)
    for tr in hmm.transitions():
        p = float(tr.data[ATTR_PROB])
        if tr.source == state:
            k = edges.index(tr)
            p += -eps if k == i else eps if k == j else 0.0
        out.add_transition(tr.source, tr.target, tr.data[ATTR_EMISSION], p)
    return out


def block_entropies(hmm: MealyHMM, max_length: int) -> list[float]:
    return [oracles.block_entropy(oracles.word_distribution(hmm, n, ALPHABET)) for n in range(max_length + 1)]


# --------------------------------------------------------------------------- word distributions


@given(small_hmms, st.integers(0, 4))
def test_word_probabilities_sum_to_one_and_match_oracle(hmm, length):
    dist = word_dist(hmm, length)
    assert sum(dist.values()) == pytest.approx(1.0, abs=TOL)
    assert_same_dist(dist, oracles.word_distribution(hmm, length))
    for word, prob in dist.items():
        assert hmm.word_probability(word) == pytest.approx(prob, abs=TOL)


@given(small_hmms, st.integers(1, 4))
def test_words_of_length_is_support_of_word_distribution(hmm, length):
    sparse = hmm.words_of_length(length)
    dense = oracles.word_distribution(hmm, length)
    assert set(sparse) == {w for w, p in dense.items() if p > 0}
    assert_same_dist(sparse, dense)


@given(small_hmms, st.lists(st.sampled_from(ALPHABET), max_size=6))
def test_log_word_probability_is_log2_of_word_probability(hmm, word):
    prob = hmm.word_probability(word)
    logp = hmm.log_word_probability(word)
    if prob > 0:
        assert logp == pytest.approx(math.log2(prob), abs=1e-9)
    else:
        assert logp == -math.inf


@heavy
@given(small_eps, st.integers(0, 2**32 - 1))
def test_log_word_probability_does_not_underflow(machine, seed):
    word, _ = machine.sample(2000, np.random.default_rng(seed))
    logp = machine.log_word_probability(word)
    assert np.isfinite(logp)
    assert logp == pytest.approx(machine.log_likelihood(word), rel=1e-9, abs=1e-6)
    assert machine.word_probability(word) == pytest.approx(2.0**logp, rel=1e-6, abs=1e-300)


def test_tiny_word_probabilities_are_not_truncated():
    """Regression: words with probability below 1e-15 were reported as impossible."""
    coin = MealyHMM(observation_alphabet=frozenset(ALPHABET), initial_distribution={0: 1.0})
    coin.graph.add_state(0)
    for symbol in ALPHABET:
        coin.add_transition(0, 0, symbol, 0.5)
    word = ["0"] * 60
    assert coin.word_probability(word) == 2.0**-60
    assert coin.conditional_word_probability(["1"], word) == 0.5


@given(
    small_hmms,
    st.lists(st.sampled_from(ALPHABET), max_size=3),
    st.lists(st.sampled_from(ALPHABET), max_size=3),
)
def test_conditional_word_probability_is_joint_over_marginal(hmm, word, condition):
    marginal = oracles.word_probability(hmm, condition)
    assume(marginal > 1e-12)
    joint = oracles.word_probability(hmm, [*condition, *word])
    assert hmm.conditional_word_probability(word, condition) == pytest.approx(joint / marginal, abs=1e-9)


@given(small_hmms, st.lists(st.sampled_from(ALPHABET), max_size=3))
def test_conditional_words_of_next_symbol_sum_to_one(hmm, condition):
    assume(oracles.word_probability(hmm, condition) > 1e-12)
    total = sum(hmm.conditional_word_probability([x], condition) for x in ALPHABET)
    assert total == pytest.approx(1.0, abs=1e-9)


# --------------------------------------------------------------------------- block entropy / measures


@heavy
@given(reversible_eps)
def test_block_entropy_diagram_matches_oracle(machine):
    diagram = machine.block_entropy_diagram(4)
    assert diagram.block_entropy[0] == 0.0
    np.testing.assert_allclose(diagram.block_entropy, block_entropies(machine, 4), atol=1e-9)


@given(small_eps)
def test_entropy_rate_bounds_and_concavity(machine):
    h = machine.entropy_rate()
    entropies = block_entropies(machine, 5)
    assert entropies[0] == pytest.approx(0.0, abs=1e-12)
    deltas = np.diff(entropies)
    for length in range(1, 6):
        assert h <= entropies[length] / length + 1e-9
        assert h <= deltas[length - 1] + 1e-9
    assert np.all(np.diff(deltas) <= 1e-9)


@heavy
@given(reversible_eps)
def test_excess_entropy_bounded_by_statistical_complexity(machine):
    e, c = machine.excess_entropy(), machine.statistical_complexity()
    assert -1e-9 <= e <= c + 1e-9
    assert machine.crypticity() == pytest.approx(c - e, abs=1e-9)
    assert machine.crypticity() >= -1e-9


@heavy
@given(reversible_eps)
def test_excess_entropy_lower_bounds_block_entropy_excess(machine):
    """``H(L) - L h`` increases to ``E`` from below."""
    h, e = machine.entropy_rate(), machine.excess_entropy()
    for length, entropy in enumerate(block_entropies(machine, 5)):
        assert entropy - length * h <= e + 1e-9


@heavy
@given(reversible_eps)
def test_information_anatomy_identities(machine):
    anatomy = machine.information_anatomy()
    h, b, r, rho = (anatomy[k] for k in ("entropy_rate", "bound_mu", "ephemeral_mu", "rho_mu"))
    h1 = oracles.block_entropy(oracles.word_distribution(machine, 1, ALPHABET))
    assert h == pytest.approx(machine.entropy_rate(), abs=1e-9)
    assert h == pytest.approx(b + r, abs=1e-9)
    assert rho == pytest.approx(h1 - h, abs=1e-9)
    assert b == pytest.approx(anatomy["bound_structural"] + anatomy["bound_gauge"], abs=1e-9)
    assert r == pytest.approx(anatomy["ephemeral_structural"] + anatomy["ephemeral_gauge"], abs=1e-9)
    assert min(b, r, rho) >= -1e-9
    assert anatomy["excess_entropy"] == pytest.approx(machine.excess_entropy(), abs=1e-9)
    assert anatomy["crypticity"] == pytest.approx(machine.statistical_complexity() - machine.excess_entropy(), abs=1e-9)


@given(small_eps)
def test_cryptic_order_at_most_markov_order(machine):
    markov, cryptic = machine.markov_order(), machine.cryptic_order()
    if math.isfinite(markov):
        assert cryptic <= markov
    if cryptic == 0:
        assert machine.crypticity() == pytest.approx(0.0, abs=1e-9)


# --------------------------------------------------------------------------- constructions


@given(small_hmms, st.integers(1, 4))
def test_mixed_state_presentation_reproduces_word_distribution(hmm, length):
    try:
        msp = build_mixed_state_presentation(hmm, initial_mixed_state=hmm.initial_distribution, max_states=100)
    except MixedStateExplosionError:
        assume(False)
    assert msp.is_unifilar()
    assert_same_dist(word_dist(msp, length), oracles.word_distribution(hmm, length), atol=1e-8)


@heavy
@given(small_hmms)
def test_epsilon_machine_from_hmm_reproduces_stationary_process(hmm):
    assume(hmm.is_irreducible())
    stationary = with_stationary_start(hmm)
    if not hmm.is_unifilar():
        try:
            build_mixed_state_presentation(stationary, max_states=100)
        except MixedStateExplosionError:
            assume(False)
    machine = EpsilonMachine.from_hmm(stationary)
    note(repr(machine))
    for length in range(1, 5):
        assert_same_dist(word_dist(machine, length), oracles.word_distribution(stationary, length), atol=1e-7)


def test_epsilon_machine_from_hmm_beyond_26_causal_states():
    """Regression: the causal-state quotient crashed on more than 26 states (A-Z labels)."""
    rng = np.random.default_rng(0)
    histories = list(product(ALPHABET, repeat=5))
    chain = MealyHMM(observation_alphabet=frozenset(ALPHABET), initial_distribution=dict.fromkeys(histories, 1 / 32))
    for history in histories:
        chain.graph.add_state(history)
    for history in histories:
        p = float(rng.uniform(0.2, 0.8))
        for symbol, prob in zip(ALPHABET, (p, 1 - p), strict=True):
            chain.add_transition(history, history[1:] + (symbol,), symbol, prob)
    machine = EpsilonMachine.from_hmm(chain)
    assert len(list(machine.states())) == 32
    assert machine.markov_order() == 5
    assert machine.is_equal_process(chain)


@given(small_eps)
def test_epsilon_machine_from_hmm_is_idempotent_on_minimal_machines(machine):
    rebuilt = EpsilonMachine.from_hmm(machine)
    assert len(list(rebuilt.states())) <= len(list(machine.states()))
    assert rebuilt.is_equal_process(machine)
    assert rebuilt.statistical_complexity() <= machine.statistical_complexity() + 1e-9


@heavy
@given(reversible_eps)
def test_time_reversal_reverses_words_and_is_an_involution(machine):
    reverse = machine.reverse()
    for length in range(1, 5):
        forward = word_dist(machine, length)
        backward = word_dist(reverse, length)
        assert_same_dist(backward, {w[::-1]: p for w, p in forward.items()}, atol=1e-9)
    assert reverse.reverse().is_equal_process(machine)
    assert reverse.entropy_rate() == pytest.approx(machine.entropy_rate(), abs=1e-9)
    assert reverse.excess_entropy() == pytest.approx(machine.excess_entropy(), abs=1e-8)


@given(small_hmms, st.lists(st.sampled_from([0.25, 0.5, 0.75]), min_size=1, max_size=3))
def test_lumping_split_states_recovers_process(hmm, weights):
    split = split_states(hmm, weights)
    partition = [{(s, 0), (s, 1)} for s in hmm.states()]
    assert is_lumpable(split, partition)
    lumped = lump(split, partition)
    assert len(list(lumped.states())) == len(list(hmm.states()))
    for length in range(4):
        expected = oracles.word_distribution(hmm, length)
        assert_same_dist(word_dist(split, length), expected)
        assert_same_dist(word_dist(lumped, length), expected)


# --------------------------------------------------------------------------- process equivalence


@given(small_hmms)
def test_is_equal_process_reflexive(hmm):
    assert hmm.is_equal_process(hmm)
    assert hmm.is_equal_process(rebuild(hmm))


@given(small_hmms, small_hmms)
def test_is_equal_process_symmetric_and_matches_oracle(a, b):
    forward = a.is_equal_process(b)
    assert forward == b.is_equal_process(a)
    assert forward == oracle_agrees(a, b)


@given(small_hmms, st.data())
def test_is_equal_process_detects_perturbations(hmm, data):
    perturbed = perturb(hmm, data)
    assert hmm.is_equal_process(perturbed) == oracle_agrees(hmm, perturbed)


# --------------------------------------------------------------------------- metamorphic


@given(small_hmms, st.data())
def test_state_relabeling_preserves_process(hmm, data):
    states = list(hmm.states())
    names = data.draw(st.permutations([f"s{i}" for i in range(len(states))]))
    mapping = dict(zip(states, names, strict=True))
    relabeled = rebuild(hmm, state_map=mapping.__getitem__)
    word = data.draw(st.lists(st.sampled_from(ALPHABET), max_size=6))
    for length in range(4):
        assert_same_dist(word_dist(relabeled, length), word_dist(hmm, length))
    assert relabeled.log_likelihood(word) == pytest.approx(hmm.log_likelihood(word), abs=1e-9) or (
        hmm.log_likelihood(word) == relabeled.log_likelihood(word) == -math.inf
    )
    assert relabeled.is_equal_process(hmm)


@heavy
@given(reversible_eps, st.data())
def test_state_relabeling_preserves_measures(machine, data):
    states = list(machine.states())
    names = data.draw(st.permutations(range(len(states))))
    mapping = dict(zip(states, names, strict=True))
    relabeled = rebuild(machine, state_map=mapping.__getitem__)
    assert relabeled.entropy_rate() == pytest.approx(machine.entropy_rate(), abs=1e-9)
    assert relabeled.statistical_complexity() == pytest.approx(machine.statistical_complexity(), abs=1e-9)
    assert relabeled.excess_entropy() == pytest.approx(machine.excess_entropy(), abs=1e-9)


@heavy
@given(reversible_eps, st.sampled_from([{"0": "1", "1": "0"}, {"0": "a", "1": "b"}]))
def test_symbol_relabeling_preserves_measures_and_maps_words(machine, mapping):
    relabeled = rebuild(machine, symbol_map=mapping.__getitem__)
    assert relabeled.entropy_rate() == pytest.approx(machine.entropy_rate(), abs=1e-9)
    assert relabeled.statistical_complexity() == pytest.approx(machine.statistical_complexity(), abs=1e-9)
    assert relabeled.excess_entropy() == pytest.approx(machine.excess_entropy(), abs=1e-9)
    assert relabeled.crypticity() == pytest.approx(machine.crypticity(), abs=1e-9)
    for length in range(4):
        mapped = {tuple(mapping[x] for x in w): p for w, p in word_dist(machine, length).items()}
        assert_same_dist(word_dist(relabeled, length), mapped)


@given(small_hmms, st.sampled_from([add_unreachable_state, add_unreachable_copy]))
def test_unreachable_states_leave_process_unchanged(hmm, extend):
    extended = extend(hmm)
    for length in range(4):
        assert_same_dist(word_dist(extended, length), word_dist(hmm, length))
    assert extended.is_equal_process(hmm)
    assert hmm.is_equal_process(extended)


@heavy
@given(reversible_eps)
def test_epsilon_machine_ignores_unreachable_copy(machine):
    extended = rebuild(add_unreachable_copy(machine), cls=MealyHMM)
    rebuilt = EpsilonMachine.from_hmm(extended)
    assert rebuilt.is_equal_process(machine)
    assert rebuilt.statistical_complexity() == pytest.approx(machine.statistical_complexity(), abs=1e-9)
