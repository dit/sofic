"""Tests for the coarsest probabilistic bisimulation (coarsest strong lumping)."""

from fractions import Fraction

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from sofic.generators.epsilon_machine import EpsilonMachine
from sofic.generators.lumping import bisimulation_partition, coarsest_lumping, is_lumpable, lump
from sofic.generators.markov import MarkovChain
from sofic.generators.mealy import MealyHMM
from sofic.generators.moore import MooreHMM
from sofic.graph import ATTR_EMISSION, ATTR_PROB
from sofic.testing.strategies import epsilon_machines, mealy_hmms
from tests import oracles
from tests.test_lumping import _moore, _symmetric_chain

ALPHABET = (0, 1)


def _golden_mean() -> MealyHMM:
    hmm = MealyHMM(initial_distribution={0: 2 / 3, 1: 1 / 3})
    hmm.add_transition(0, 0, 0, 0.5)
    hmm.add_transition(0, 1, 1, 0.5)
    hmm.add_transition(1, 0, 0, 1.0)
    return hmm


def _set_partitions(items: list) -> list[list[list]]:
    if not items:
        return [[]]
    first, rest = items[0], items[1:]
    partitions = []
    for partition in _set_partitions(rest):
        partitions.append([[first], *partition])
        for i in range(len(partition)):
            partitions.append([*partition[:i], [first, *partition[i]], *partition[i + 1 :]])
    return partitions


def _refines(fine: list, coarse: list[frozenset]) -> bool:
    return all(any(set(block) <= other for other in coarse) for block in fine)


def _clone_state(hmm: MealyHMM, state: int, route: list[float]) -> MealyHMM:
    """Add a copy of ``state``; the ``i``-th edge into ``state`` sends ``route[i]`` of its mass to the original."""
    clone = max(hmm.states()) + 1
    out = MealyHMM(observation_alphabet=hmm.observation_alphabet)
    for s in [*hmm.states(), clone]:
        out.graph.add_state(s)
    incoming = 0
    for transition in hmm.transitions():
        symbol, prob = transition.data[ATTR_EMISSION], transition.data[ATTR_PROB]
        sources = [transition.source, clone] if transition.source == state else [transition.source]
        for source in sources:
            if transition.target != state:
                out.add_transition(source, transition.target, symbol, prob)
                continue
            share = route[incoming % len(route)]
            if share > 0:
                out.add_transition(source, state, symbol, share * prob)
            if share < 1:
                out.add_transition(source, clone, symbol, (1 - share) * prob)
        incoming += transition.target == state
    initial = dict(hmm.initial_distribution)
    mass = initial.pop(state, 0.0)
    out.initial_distribution = {**initial, state: mass / 2, clone: mass / 2}
    return out


@st.composite
def redundant_hmms(draw, *, max_states=4, unifilar_routes=False):
    hmm = draw(mealy_hmms(alphabet=ALPHABET, max_states=max_states))
    state = draw(st.sampled_from(sorted(hmm.states())))
    shares = st.sampled_from([0.0, 1.0]) if unifilar_routes else st.sampled_from([0.0, 0.25, 0.5, 1.0])
    return _clone_state(hmm, state, draw(st.lists(shares, min_size=1, max_size=4)))


any_small_hmm = st.one_of(mealy_hmms(alphabet=ALPHABET, max_states=5), redundant_hmms())


@settings(max_examples=80, deadline=None)
@given(any_small_hmm)
def test_partition_is_lumpable(hmm):
    assert is_lumpable(hmm, bisimulation_partition(hmm))


@settings(max_examples=60, deadline=None)
@given(any_small_hmm)
def test_partition_is_coarsest(hmm):
    blocks = bisimulation_partition(hmm)
    for candidate in _set_partitions(sorted(hmm.states())):
        if is_lumpable(hmm, candidate):
            assert _refines(candidate, blocks)


@settings(max_examples=60, deadline=None)
@given(any_small_hmm)
def test_lumped_model_generates_same_process(hmm):
    lumped = coarsest_lumping(hmm)
    assert len(list(lumped.states())) == len(bisimulation_partition(hmm))
    for word in oracles.words(ALPHABET, 4):
        assert oracles.word_probability(lumped, word) == pytest.approx(oracles.word_probability(hmm, word), abs=1e-12)


@settings(max_examples=60, deadline=None)
@given(epsilon_machines(alphabet=ALPHABET, max_states=4))
def test_epsilon_machine_is_bisimulation_minimal(machine):
    assert all(len(block) == 1 for block in bisimulation_partition(machine))


@settings(max_examples=60, deadline=None)
@given(mealy_hmms(alphabet=ALPHABET, max_states=4), st.data())
def test_duplicated_state_collapses(hmm, data):
    state = data.draw(st.sampled_from(sorted(hmm.states())))
    redundant = _clone_state(hmm, state, [0.5])
    clone = max(redundant.states())
    blocks = bisimulation_partition(redundant)
    assert any({state, clone} <= block for block in blocks)
    assert len(blocks) == len(bisimulation_partition(hmm))


@settings(max_examples=60, deadline=None)
@given(epsilon_machines(alphabet=ALPHABET, max_states=4), st.data())
def test_unifilar_quotient_matches_epsilon_machine(machine, data):
    state = data.draw(st.sampled_from(sorted(machine.states())))
    routes = data.draw(st.lists(st.sampled_from([0.0, 1.0]), min_size=1, max_size=4))
    redundant = _clone_state(machine, state, routes)
    assert redundant.is_unifilar()
    blocks = bisimulation_partition(redundant)
    assert len(blocks) == len(list(EpsilonMachine.from_hmm(redundant).states()))
    assert len(blocks) == len(list(machine.states()))


def test_lump_without_partition_uses_coarsest():
    hmm = _clone_state(_golden_mean(), 1, [0.5])
    assert bisimulation_partition(hmm) == [frozenset({0}), frozenset({1, 2})]
    assert sorted(lump(hmm).states(), key=str) == [0, "1+2"]


def test_markov_chain_without_labels_lumps_to_one_block():
    assert bisimulation_partition(_symmetric_chain()) == [frozenset({"A", "B", "C"})]


def test_markov_chain_refines_initial_partition():
    chain = _symmetric_chain()
    blocks = bisimulation_partition(chain, initial={"A": 0, "B": 1, "C": 1})
    assert blocks == [frozenset({"A"}), frozenset({"B", "C"})]
    assert bisimulation_partition(chain, initial=[{"A"}, {"B"}, {"C"}]) == [
        frozenset({"A"}),
        frozenset({"B"}),
        frozenset({"C"}),
    ]


def test_moore_partition_respects_emissions():
    assert bisimulation_partition(_moore()) == [frozenset({"A"}), frozenset({"B", "C"})]
    assert isinstance(coarsest_lumping(_moore()), MooreHMM)


def test_markov_chain_quotient_type():
    lumped = coarsest_lumping(_symmetric_chain(), initial=[{"A"}, {"B", "C"}])
    assert isinstance(lumped, MarkovChain)
    assert sorted(lumped.states()) == ["A", "B+C"]


def test_symbolic_probabilities_compare_exactly():
    sp = pytest.importorskip("sympy")
    a = sp.Symbol("a", positive=True)
    hmm = MealyHMM(initial_distribution={0: sp.Integer(1)})
    hmm.add_transition(0, 1, 0, a / 2)
    hmm.add_transition(0, 2, 0, a / 2)
    hmm.add_transition(0, 0, 1, 1 - a)
    hmm.add_transition(1, 0, 1, sp.Integer(1))
    hmm.add_transition(2, 0, 1, sp.Integer(1))
    assert bisimulation_partition(hmm) == [frozenset({0}), frozenset({1, 2})]


def test_fraction_probabilities_compare_exactly():
    chain = MarkovChain(initial_distribution={0: 1.0})
    for state in range(3):
        chain.graph.add_state(state)
    tiny = Fraction(1, 10**20)
    chain.graph.add_transition(0, 1, **{ATTR_PROB: Fraction(1, 2)})
    chain.graph.add_transition(0, 2, **{ATTR_PROB: Fraction(1, 2)})
    chain.graph.add_transition(1, 0, **{ATTR_PROB: Fraction(1, 2) + tiny})
    chain.graph.add_transition(1, 1, **{ATTR_PROB: Fraction(1, 2) - tiny})
    chain.graph.add_transition(2, 0, **{ATTR_PROB: Fraction(1, 2)})
    chain.graph.add_transition(2, 2, **{ATTR_PROB: Fraction(1, 2)})
    blocks = bisimulation_partition(chain, initial=[{0}, {1, 2}])
    assert blocks == [frozenset({0}), frozenset({1}), frozenset({2})]


def test_float_tolerance_merges_rounding_noise():
    chain = MarkovChain(initial_distribution={0: 1.0})
    for state in range(3):
        chain.graph.add_state(state)
    chain.add_transition(0, 1, 0.5)
    chain.add_transition(0, 2, 0.5)
    chain.add_transition(1, 0, 0.1 + 0.2)
    chain.add_transition(1, 1, 0.7)
    chain.add_transition(2, 0, 0.3)
    chain.add_transition(2, 2, 0.7)
    assert bisimulation_partition(chain, initial=[{0}, {1, 2}]) == [frozenset({0}), frozenset({1, 2})]


def test_unsupported_model_raises():
    from sofic.automata.dfa import DFA

    with pytest.raises(TypeError):
        bisimulation_partition(DFA())
