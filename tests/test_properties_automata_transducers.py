"""Property-based tests for transducers, transducer/generator operations, and unifilar automata."""

from __future__ import annotations

import math
from collections.abc import Hashable
from itertools import product
from typing import Any

import pytest
from hypothesis import assume, given
from hypothesis import strategies as st

from sofic.automata import (
    MealyMachine,
    SubsequentialTransducer,
    UnifilarAutomaton,
    WeightedFiniteStateTransducer,
    compose_transducer_generator,
    compose_transducers,
    generator_product,
    transduce_generator,
    transducer_product,
)
from sofic.generators.mealy import MealyHMM
from sofic.graph import ATTR_OUTPUT, ATTR_PROB, ATTR_SYMBOL, EPSILON
from sofic.testing import dfas, mealy_hmms, mealy_transducers
from tests import oracles

AB = ("0", "1")
OUT = ("a", "b")
FINAL = ("x", "y")

Word = tuple[Any, ...]


@st.composite
def stochastic_transducers(
    draw: Any, inputs: tuple[str, ...] = AB, outputs: tuple[str, ...] = OUT, max_states: int = 2
) -> MealyMachine:
    """Complete, epsilon-free Mealy transducers with ``P(. | state, input)`` summing to one."""
    n = draw(st.integers(1, max_states))
    machine = MealyMachine(
        input_alphabet=frozenset(inputs),
        output_alphabet=frozenset(outputs),
        initial_states=draw(st.frozensets(st.integers(0, n - 1), min_size=1)),
    )
    for state in range(n):
        machine.graph.add_state(state)
    edge = st.tuples(st.integers(0, n - 1), st.sampled_from(outputs))
    for state in range(n):
        for symbol in inputs:
            chosen = draw(st.lists(edge, min_size=1, max_size=2, unique=True))
            weights = draw(st.lists(st.integers(1, 5), min_size=len(chosen), max_size=len(chosen)))
            for (target, output), weight in zip(chosen, weights, strict=True):
                machine.add_transition(state, target, symbol, output, prob=weight / sum(weights))
    return machine


def conditional(machine: MealyMachine, x: Word, y: Word) -> float:
    """``P(y | x)`` by path enumeration, starting uniformly over the initial states."""

    def paths(state: Hashable, position: int) -> float:
        if position == len(x):
            return 1.0
        total = 0.0
        for t in machine.graph.out_transitions(state):
            if t.data.get(ATTR_SYMBOL) == x[position] and t.data.get(ATTR_OUTPUT) == y[position]:
                total += float(t.data.get(ATTR_PROB, 1.0)) * paths(t.target, position + 1)
        return total

    starts = tuple(machine.initial_states)
    return sum(paths(state, 0) for state in starts) / len(starts)


def path_weights(machine: MealyMachine, x: Word, y: Word) -> list[float]:
    """Probability of every complete path reading ``x`` and writing ``y``."""
    found: list[float] = []

    def walk(state: Hashable, position: int, weight: float) -> None:
        if position == len(x):
            found.append(weight)
            return
        for t in machine.graph.out_transitions(state):
            if t.data.get(ATTR_SYMBOL) == x[position] and t.data.get(ATTR_OUTPUT) == y[position]:
                walk(t.target, position + 1, weight * float(t.data.get(ATTR_PROB, 1.0)))

    for state in machine.initial_states:
        walk(state, 0, 1.0)
    return found


def close(left: float, right: float) -> bool:
    return math.isclose(left, right, rel_tol=1e-9, abs_tol=1e-12)


# --------------------------------------------------------------------------- relations


@given(mealy_transducers(), st.lists(st.sampled_from(AB), max_size=5).map(tuple))
def test_transduce_matches_oracle(machine: MealyMachine, x: Word) -> None:
    assert machine.transduce(x) == oracles.transducer_outputs(machine, x)


@given(
    mealy_transducers(),
    mealy_transducers(input_alphabet=OUT, output_alphabet=FINAL),
)
def test_compose_transducers_is_relational_composition(first: MealyMachine, second: MealyMachine) -> None:
    inputs = oracles.words(AB, 4)
    composed = compose_transducers((first, second), complete=False)
    expected = oracles.compose_relations(
        oracles.transducer_relation(first, inputs), lambda y: oracles.transducer_outputs(second, y)
    )
    assert oracles.transducer_relation(composed, inputs) == expected


@given(
    mealy_transducers(allow_epsilon_output=False),
    mealy_transducers(input_alphabet=("p", "q"), output_alphabet=FINAL, allow_epsilon_output=False),
)
def test_transducer_product_is_componentwise(first: MealyMachine, second: MealyMachine) -> None:
    joint = transducer_product((first, second))
    for n in range(4):
        for x1, x2 in product(oracles.words(AB, n, min_length=n), oracles.words(("p", "q"), n, min_length=n)):
            expected = {
                tuple(zip(y1, y2, strict=True))
                for y1 in oracles.transducer_outputs(first, x1)
                for y2 in oracles.transducer_outputs(second, x2)
            }
            assert joint.transduce(tuple(zip(x1, x2, strict=True))) == expected


@given(stochastic_transducers(), stochastic_transducers(inputs=OUT, outputs=FINAL))
def test_compose_transducers_multiplies_conditionals(first: MealyMachine, second: MealyMachine) -> None:
    composed = compose_transducers((first, second))
    for n in range(4):
        middles = oracles.words(OUT, n, min_length=n)
        for x in oracles.words(AB, n, min_length=n):
            for z in oracles.words(FINAL, n, min_length=n):
                expected = sum(conditional(first, x, y) * conditional(second, y, z) for y in middles)
                assert close(conditional(composed, x, z), expected)


# --------------------------------------------------------------------------- generator pushforwards


@given(stochastic_transducers(), mealy_hmms(max_states=2), st.integers(0, 3))
def test_transduce_generator_is_pushforward(machine: MealyMachine, generator: MealyHMM, n: int) -> None:
    output = transduce_generator(machine, generator)
    inputs = oracles.words(AB, n, min_length=n)
    for y in oracles.words(OUT, n, min_length=n):
        expected = sum(oracles.word_probability(generator, x) * conditional(machine, x, y) for x in inputs)
        assert close(oracles.word_probability(output, y), expected)


@given(stochastic_transducers(), mealy_hmms(max_states=2), st.integers(0, 3))
def test_compose_transducer_generator_is_joint_law(machine: MealyMachine, generator: MealyHMM, n: int) -> None:
    joint = compose_transducer_generator(machine, generator)
    for x in oracles.words(AB, n, min_length=n):
        px = oracles.word_probability(generator, x)
        for y in oracles.words(OUT, n, min_length=n):
            expected = px * conditional(machine, x, y)
            assert close(oracles.word_probability(joint, tuple(zip(x, y, strict=True))), expected)


@given(mealy_hmms(max_states=2), mealy_hmms(alphabet=OUT, max_states=2), st.integers(0, 3))
def test_generator_product_is_independent_coupling(left: MealyHMM, right: MealyHMM, n: int) -> None:
    joint = generator_product((left, right))
    for a in oracles.words(AB, n, min_length=n):
        pa = oracles.word_probability(left, a)
        for b in oracles.words(OUT, n, min_length=n):
            expected = pa * oracles.word_probability(right, b)
            assert close(oracles.word_probability(joint, tuple(zip(a, b, strict=True))), expected)


@given(
    stochastic_transducers(max_states=1),
    stochastic_transducers(inputs=OUT, outputs=FINAL, max_states=2),
    mealy_hmms(max_states=2),
)
def test_transducing_twice_equals_transducing_by_composition(
    first: MealyMachine, second: MealyMachine, generator: MealyHMM
) -> None:
    staged = transduce_generator(second, transduce_generator(first, generator))
    direct = transduce_generator(compose_transducers((first, second)), generator)
    for z in oracles.words(FINAL, 2, min_length=2):
        assert close(oracles.word_probability(staged, z), oracles.word_probability(direct, z))


# --------------------------------------------------------------------------- subsequential and weighted


@st.composite
def subsequential_transducers(draw: Any) -> SubsequentialTransducer:
    n = draw(st.integers(1, 3))
    output_word = st.lists(st.sampled_from(OUT), max_size=2).map(tuple)
    machine = SubsequentialTransducer(
        input_alphabet=frozenset(AB),
        output_alphabet=frozenset(OUT),
        initial_states=frozenset({0}),
        final_output=draw(st.dictionaries(st.integers(0, n - 1), output_word)),
    )
    for state in range(n):
        machine.graph.add_state(state)
    for state in range(n):
        for symbol in AB:
            if draw(st.booleans()):
                target = draw(st.integers(0, n - 1))
                machine.add_transition(state, target, symbol, draw(st.sampled_from((*OUT, EPSILON))))
    return machine


@given(subsequential_transducers(), st.lists(st.sampled_from(AB), max_size=5).map(tuple))
def test_subsequential_transduce_appends_final_output(machine: SubsequentialTransducer, x: Word) -> None:
    machine.validate()
    assert machine.is_subsequential()
    state: Hashable | None = 0
    out: Word = ()
    for symbol in x:
        step = [t for t in machine.graph.out_transitions(state) if t.data.get(ATTR_SYMBOL) == symbol]
        if not step:
            state = None
            break
        emitted = step[0].data.get(ATTR_OUTPUT)
        out += () if emitted is EPSILON else (emitted,)
        state = step[0].target
    expected = set() if state is None else {out + machine.final_output.get(state, ())}
    assert machine.transduce(x) == expected


@given(stochastic_transducers(), st.integers(0, 4), st.data())
def test_wfst_weights_match_path_sums(machine: MealyMachine, n: int, data: Any) -> None:
    x = data.draw(st.lists(st.sampled_from(AB), min_size=n, max_size=n).map(tuple))
    y = data.draw(st.lists(st.sampled_from(OUT), min_size=n, max_size=n).map(tuple))
    found = path_weights(machine, x, y)
    probability = WeightedFiniteStateTransducer.from_transducer(machine)
    tropical = WeightedFiniteStateTransducer.from_transducer(machine, semiring="tropical")
    assert close(probability.weight(x, y), sum(found))
    expected_cost = -math.log(max(found)) if found else math.inf
    assert tropical.weight(x, y) == pytest.approx(expected_cost)


# --------------------------------------------------------------------------- unifilar automata


@st.composite
def unifilar_automata(draw: Any) -> UnifilarAutomaton:
    dfa = draw(dfas())
    result = UnifilarAutomaton(
        input_alphabet=frozenset(AB),
        initial_states=frozenset(dfa.initial_states),
        accepting_states=frozenset(dfa.accepting_states),
    )
    for state in dfa.states():
        result.graph.add_state(state)
    for t in dfa.transitions():
        if draw(st.integers(0, 3)):
            result.graph.add_transition(t.source, t.target, **{ATTR_SYMBOL: t.data.get(ATTR_SYMBOL)})
    return result


def delta_set(aut: UnifilarAutomaton, states: frozenset[Hashable], w: Word) -> frozenset[Hashable]:
    for symbol in w:
        states = oracles._symbol_step(aut, states, symbol)
    return states


def brute_reset_threshold(aut: UnifilarAutomaton, limit: int = 8) -> float:
    full = frozenset(aut.states())
    for length in range(limit + 1):
        if any(len(delta_set(aut, full, w)) == 1 for w in oracles.words(AB, length, min_length=length)):
            return length
    return math.inf


def brute_markov_order(aut: UnifilarAutomaton, limit: int = 4) -> float:
    full = frozenset(aut.states())
    for length in range(limit + 1):
        images = [delta_set(aut, full, w) for w in oracles.words(AB, length, min_length=length)]
        if all(len(image) <= 1 for image in images):
            return length
    return math.inf


@given(unifilar_automata())
def test_unifilar_recognizes_matches_oracle(aut: UnifilarAutomaton) -> None:
    aut.validate()
    assert aut.is_unifilar()
    for w in oracles.words(AB, 5):
        assert aut.recognizes(w) == oracles.nfa_accepts(aut, w)


@given(unifilar_automata())
def test_synchronizing_word_is_shortest_reset_word(aut: UnifilarAutomaton) -> None:
    threshold = aut.reset_threshold()
    assert threshold == brute_reset_threshold(aut)
    assert aut.is_exactly_synchronizable() == (threshold != math.inf)
    word = aut.synchronizing_word()
    if word is None:
        assert threshold == math.inf
    else:
        assert len(word) == threshold
        assert len(delta_set(aut, frozenset(aut.states()), tuple(word))) == 1


def futures(aut: UnifilarAutomaton, state: Hashable, n: int = 3) -> frozenset[Word]:
    return frozenset(w for w in oracles.words(AB, n) if delta_set(aut, frozenset({state}), w))


@given(unifilar_automata())
def test_markov_order_matches_brute_force(aut: UnifilarAutomaton) -> None:
    # markov_order ignores subsets that can never synchronize, which only exist
    # when two states share their futures, so require a minimal presentation.
    assume(all(any(True for _ in aut.graph.out_transitions(state)) for state in aut.states()))
    assume(len({futures(aut, state) for state in aut.states()}) == len(list(aut.states())))
    order = aut.markov_order()
    assert order == brute_markov_order(aut)
    assert aut.is_definite() == (order != math.inf)
