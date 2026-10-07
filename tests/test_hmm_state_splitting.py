"""Process-preserving state splitting and amalgamation of Mealy HMMs.

Metamorphic checks: a split generates the same word distributions (brute force
via :mod:`tests.oracles`), entropy rate, excess entropy and ε-machine, and
amalgamating the copies recovers the original presentation.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import pytest
import sympy
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from sofic.examples import golden_mean
from sofic.exceptions import LumpabilityError, MixedStateExplosionError
from sofic.generators.epsilon_machine import EpsilonMachine
from sofic.generators.mealy import MealyHMM
from sofic.generators.measures import entropy_rate_bounds
from sofic.generators.state_splitting import amalgamate, in_edges, out_edges, split_state
from sofic.graph import ATTR_EMISSION, ATTR_PROB
from sofic.testing.strategies import epsilon_machines, mealy_hmms
from tests import oracles

ALPHABET = ("0", "1")
TOL = 1e-9

small_hmms = mealy_hmms(alphabet=ALPHABET, max_states=3)
small_eps = epsilon_machines(alphabet=ALPHABET, max_states=3)
reversible_eps = small_eps.filter(lambda m: m.reverse_is_finite())
heavy = settings(max_examples=max(10, settings.default.max_examples // 4))


@st.composite
def splits(draw: Any, models: Any, kind: str) -> tuple[MealyHMM, Any, list[list[Any]]]:
    """Draw ``(hmm, state, partition)`` with at least two parts."""
    hmm = draw(models)
    state = draw(st.sampled_from(sorted(hmm.states(), key=repr)))
    edges = out_edges(hmm, state) if kind == "out" else in_edges(hmm, state)
    assume(len(edges) >= 2)
    parts = draw(st.integers(2, len(edges)))
    labels = list(range(parts)) + draw(
        st.lists(st.integers(0, parts - 1), min_size=len(edges) - parts, max_size=len(edges) - parts)
    )
    order = draw(st.permutations(range(len(edges))))
    partition: list[list[Any]] = [[] for _ in range(parts)]
    for index, label in zip(order, labels, strict=True):
        partition[label].append(edges[index])
    return hmm, state, partition


def edge_table(hmm: MealyHMM) -> dict[tuple[Any, Any, Any], float]:
    table: dict[tuple[Any, Any, Any], float] = defaultdict(float)
    for t in hmm.transitions():
        table[(t.source, t.target, t.data[ATTR_EMISSION])] += float(t.data[ATTR_PROB])
    return dict(table)


def assert_same_presentation(left: MealyHMM, right: MealyHMM) -> None:
    assert set(left.states()) == set(right.states())
    left_edges, right_edges = edge_table(left), edge_table(right)
    assert set(left_edges) == set(right_edges)
    for key, prob in left_edges.items():
        assert right_edges[key] == pytest.approx(prob, abs=TOL)
    states = set(left.initial_distribution) | set(right.initial_distribution)
    for state in states:
        assert float(left.initial_distribution.get(state, 0.0)) == pytest.approx(
            float(right.initial_distribution.get(state, 0.0)), abs=TOL
        )


def assert_same_words(left: MealyHMM, right: MealyHMM, max_length: int = 3) -> None:
    for n in range(max_length + 1):
        expected = oracles.word_distribution(left, n, ALPHABET)
        observed = oracles.word_distribution(right, n, ALPHABET)
        for word, prob in expected.items():
            assert observed[word] == pytest.approx(prob, abs=TOL)


def copies_of(state: Any, partition: list[list[Any]]) -> frozenset[Any]:
    return frozenset((state, index) for index in range(len(partition)))


def has_inflow(hmm: MealyHMM, state: Any) -> bool:
    """Whether out-split weights of ``state`` are recorded in its copies' inflow."""
    return bool(in_edges(hmm, state)) or float(hmm.initial_distribution.get(state, 0.0)) > 0.0


def rebuild_epsilon_machine(split: MealyHMM) -> EpsilonMachine:
    """ε-machine of a split; out-splits are non-unifilar and may have infinitely many mixed states."""
    try:
        return EpsilonMachine.from_hmm(split, max_states=200)
    except MixedStateExplosionError:
        assume(False)
        raise


def merge_back(split: MealyHMM, state: Any, partition: list[list[Any]]) -> MealyHMM:
    block = copies_of(state, partition)
    blocks = [block, *({other} for other in split.states() if other not in block)]
    return amalgamate(split, blocks, labels={block: state})


# --------------------------------------------------------------------------- word distributions


@pytest.mark.parametrize("kind", ["out", "in"])
@given(data=st.data())
def test_split_preserves_word_distributions(kind, data):
    hmm, state, partition = data.draw(splits(small_hmms, kind))
    split = split_state(hmm, state, partition, kind=kind)
    split.validate()
    assert_same_words(hmm, split)
    assert split.is_equal_process(hmm)


@pytest.mark.parametrize("kind", ["out", "in"])
@given(data=st.data())
def test_amalgamating_a_split_recovers_the_original(kind, data):
    hmm, state, partition = data.draw(splits(small_hmms, kind))
    assume(kind == "in" or has_inflow(hmm, state))
    split = split_state(hmm, state, partition, kind=kind)
    assert_same_presentation(merge_back(split, state, partition), hmm)


@given(data=st.data())
def test_amalgamating_an_unreachable_out_split_preserves_the_process(data):
    hmm, state, partition = data.draw(splits(small_hmms, "out"))
    split = split_state(hmm, state, partition)
    assert merge_back(split, state, partition).is_equal_process(hmm)


@given(data=st.data())
def test_in_then_out_split_amalgamates_in_one_call(data):
    hmm, first, in_partition = data.draw(splits(small_hmms, "in"))
    split = split_state(hmm, first, in_partition, kind="in")
    second = data.draw(st.sampled_from(sorted(set(hmm.states()) - {first}, key=repr) or [first]))
    assume(second != first and len(out_edges(split, second)) >= 2 and has_inflow(split, second))
    edges = out_edges(split, second)
    out_partition = [edges[:1], edges[1:]]
    twice = split_state(split, second, out_partition, kind="out")
    assert_same_words(hmm, twice, max_length=2)

    in_block, out_block = copies_of(first, in_partition), copies_of(second, out_partition)
    rest = [{state} for state in twice.states() if state not in in_block | out_block]
    merged = amalgamate(twice, [in_block, out_block, *rest], labels={in_block: first, out_block: second})
    assert_same_presentation(merged, hmm)


# --------------------------------------------------------------------------- information measures


@pytest.mark.parametrize("kind", ["out", "in"])
@heavy
@given(data=st.data())
def test_split_preserves_entropy_rate_and_epsilon_machine(kind, data):
    machine, state, partition = data.draw(splits(small_eps, kind))
    split = split_state(machine, state, partition, kind=kind)
    rebuilt = rebuild_epsilon_machine(split)
    assert rebuilt.is_equal_process(machine)
    assert len(list(rebuilt.states())) == len(list(machine.states()))
    assert float(rebuilt.statistical_complexity()) == pytest.approx(float(machine.statistical_complexity()), abs=1e-9)
    assert float(rebuilt.entropy_rate()) == pytest.approx(float(machine.entropy_rate()), abs=1e-9)
    lower, upper = entropy_rate_bounds(split, 6)
    assert lower - TOL <= float(machine.entropy_rate()) <= upper + TOL


@pytest.mark.parametrize("kind", ["out", "in"])
@heavy
@given(data=st.data())
def test_split_preserves_excess_entropy(kind, data):
    machine, state, partition = data.draw(splits(reversible_eps, kind))
    rebuilt = rebuild_epsilon_machine(split_state(machine, state, partition, kind=kind))
    assert float(rebuilt.excess_entropy()) == pytest.approx(float(machine.excess_entropy()), abs=1e-6)


@pytest.mark.parametrize("kind", ["out", "in"])
@given(data=st.data())
def test_stationary_initial_stays_stationary(kind, data):
    machine, state, partition = data.draw(splits(small_eps, kind))
    assert split_state(machine, state, partition, kind=kind).is_stationary()


def test_in_split_preserves_unifilarity():
    machine = golden_mean(0.5)
    edges = in_edges(machine, "A")
    split = split_state(machine, "A", [edges[:1], edges[1:]], kind="in")
    assert split.is_unifilar()
    assert split.is_equal_process(machine)


def test_out_split_is_exact_for_symbolic_probabilities():
    p = sympy.Symbol("p", positive=True)
    hmm = MealyHMM(initial_distribution={"A": 1 / (1 + p), "B": p / (1 + p)})
    hmm.graph.add_state("A")
    hmm.graph.add_state("B")
    hmm.add_transition("A", "A", "0", 1 - p)
    hmm.add_transition("A", "B", "1", p)
    hmm.add_transition("B", "A", "0", 1)
    edges = out_edges(hmm, "A")
    split = split_state(hmm, "A", [edges[:1], edges[1:]])
    for state in split.states():
        row = sum(t.data[ATTR_PROB] for t in split.graph.out_transitions(state))
        assert sympy.simplify(row - 1) == 0
    assert sympy.simplify(sum(split.initial_distribution.values()) - 1) == 0

    value = sympy.Rational(1, 3)
    numeric_split, numeric_hmm = substitute(split, p, value), substitute(hmm, p, value)
    assert numeric_split.is_equal_process(numeric_hmm)


def substitute(hmm: MealyHMM, symbol: sympy.Symbol, value: Any) -> MealyHMM:
    numeric = MealyHMM(
        observation_alphabet=hmm.observation_alphabet,
        initial_distribution={
            s: float(sympy.sympify(m).subs(symbol, value)) for s, m in hmm.initial_distribution.items()
        },
    )
    for state in hmm.states():
        numeric.graph.add_state(state)
    for t in hmm.transitions():
        prob = float(sympy.sympify(t.data[ATTR_PROB]).subs(symbol, value))
        numeric.add_transition(t.source, t.target, t.data[ATTR_EMISSION], prob)
    return numeric


# --------------------------------------------------------------------------- errors


def test_split_rejects_bad_partitions():
    machine = golden_mean(0.5)
    edges = out_edges(machine, "A")
    with pytest.raises(ValueError, match="unknown state"):
        split_state(machine, "Z", [edges])
    with pytest.raises(ValueError, match="empty part"):
        split_state(machine, "A", [edges, []])
    with pytest.raises(ValueError, match="do not cover"):
        split_state(machine, "A", [edges[:1]])
    with pytest.raises(ValueError, match="disjoint"):
        split_state(machine, "A", [edges, edges[:1]])
    with pytest.raises(ValueError, match="kind"):
        split_state(machine, "A", [edges], kind="sideways")


def test_amalgamate_rejects_inequivalent_states():
    machine = golden_mean(0.5)
    with pytest.raises(LumpabilityError):
        amalgamate(machine, [{"A", "B"}])
