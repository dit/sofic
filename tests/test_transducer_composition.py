"""Tests for transducer composition helpers."""

from __future__ import annotations

import pytest

import sofic.examples.processes as processes
from sofic.automata.transducer_operations import compose_transducer_generator, compose_transducers, transduce_generator
from sofic.examples import bernoulli


def test_bitflip_composed_with_bitflip_is_identity():
    composed = compose_transducers((processes.bit_flip(), processes.bit_flip()))

    composed.validate()
    composed.validate_stochastic()
    assert composed.transduce(("0", "1", "1", "0")) == {("0", "1", "1", "0")}


def test_serial_composition_passes_outputs_to_next_transducer():
    composed = compose_transducers((processes.gm_to_even(), processes.bit_flip()))

    assert composed.transduce(("1", "0")) == {("0", "0")}
    assert composed.transduce(("0", "1", "0")) == {("1", "0", "0")}


def test_transducer_completion_emits_error_symbol_for_missing_input():
    completed = processes.gm_to_even().complete(frozenset({"0", "1"}))

    completed.validate()
    assert completed.transduce(("1", "1")) == {("1", "?")}


def test_compose_tg_keeps_joint_input_output_emissions():
    joint = compose_transducer_generator(processes.gm_to_even(), processes.golden_mean(0.5))

    assert joint.word_probability((("1", "1"), ("0", "1"))) == pytest.approx(1 / 3)
    assert joint.word_probability((("1", "1"), ("1", "1"))) == pytest.approx(0.0)


def test_golden_mean_through_gm_to_even_generator():
    output = transduce_generator(processes.gm_to_even(), processes.golden_mean(0.5))

    assert output.word_probability(("1", "1")) == pytest.approx(1 / 3)
    assert output.word_probability(("1", "0")) == pytest.approx(0.0)


def test_binary_channel_preserves_output_probabilities():
    channel = processes.binary_channel(p=0.25, q=0.5)
    output = transduce_generator(channel, bernoulli())

    assert output.word_probability(("1",)) == pytest.approx(0.375)
    assert output.word_probability(("0",)) == pytest.approx(0.625)


def _random_mealy(rng, inputs, outputs, *, epsilon_inputs=True):
    import itertools

    from sofic.automata.transducers import MealyMachine
    from sofic.graph import EPSILON

    n = rng.randint(1, 3)
    machine = MealyMachine(
        input_alphabet=frozenset(inputs), output_alphabet=frozenset(outputs), initial_states=frozenset({0})
    )
    for state in range(n):
        machine.graph.add_state(state)
    for source, target in itertools.product(range(n), repeat=2):
        for symbol in inputs:
            if rng.random() < 0.3:
                machine.add_transition(source, target, symbol, rng.choice([*outputs, EPSILON]), prob=rng.random())
        # Epsilon-input edges only go forward, so runs stay finite.
        if epsilon_inputs and target > source and rng.random() < 0.3:
            machine.add_transition(source, target, EPSILON, rng.choice([*outputs, EPSILON]), prob=rng.random())
    return machine


def _path_weights(machine, word):
    """Sum of path probabilities per output for ``word`` (brute-force run enumeration)."""
    from collections import defaultdict

    from sofic.graph import ATTR_OUTPUT, ATTR_PROB, ATTR_SYMBOL, EPSILON

    weights = defaultdict(float)

    def walk(state, position, output, weight):
        if position == len(word):
            weights[output] += weight
        for edge in machine.graph.out_transitions(state):
            symbol = edge.data.get(ATTR_SYMBOL, EPSILON)
            emitted = edge.data.get(ATTR_OUTPUT, EPSILON)
            extra = () if emitted is EPSILON else (emitted,)
            step = weight * edge.data.get(ATTR_PROB, 1.0)
            if symbol is EPSILON:
                walk(edge.target, position, output + extra, step)
            elif position < len(word) and symbol == word[position]:
                walk(edge.target, position + 1, output + extra, step)

    for start in machine.initial_states:
        walk(start, 0, (), 1.0)
    return weights


def test_composition_weights_count_each_path_pair_once():
    import itertools
    import random

    rng = random.Random(4)
    for _trial in range(150):
        left, right = _random_mealy(rng, "ab", "xy"), _random_mealy(rng, "xy", "uv")
        composed = compose_transducers((left, right), complete=False, normalize=False)
        for word in (w for n in range(4) for w in itertools.product("ab", repeat=n)):
            expected = {}
            for middle, weight in _path_weights(left, word).items():
                for output, inner in _path_weights(right, middle).items():
                    expected[output] = expected.get(output, 0.0) + weight * inner
            got = _path_weights(composed, word)
            assert set(got) == set(expected)
            for output, weight in expected.items():
                assert got[output] == pytest.approx(weight)
            assert composed.transduce(word) == set(expected)


def test_composition_epsilon_interleavings_are_not_double_counted():
    from sofic.automata.transducers import MealyMachine
    from sofic.graph import EPSILON

    left = MealyMachine(input_alphabet=frozenset("a"), output_alphabet=frozenset("x"), initial_states=frozenset({0}))
    left.graph.add_state(0)
    left.graph.add_state(1)
    left.add_transition(0, 1, "a", EPSILON, prob=1.0)
    right = MealyMachine(input_alphabet=frozenset("x"), output_alphabet=frozenset("u"), initial_states=frozenset({0}))
    right.graph.add_state(0)
    right.graph.add_state(1)
    right.add_transition(0, 1, EPSILON, "u", prob=1.0)
    composed = compose_transducers((left, right), complete=False, normalize=False)
    assert dict(_path_weights(composed, ("a",))) == {(): 1.0, ("u",): 1.0}


def test_transducer_product_is_the_product_of_transductions():
    import itertools
    import random

    from sofic.automata.transducer_operations import transducer_product
    from sofic.exceptions import InfiniteTransductionError
    from sofic.graph import EPSILON

    rng = random.Random(8)
    for _trial in range(150):
        first, second = _random_mealy(rng, "ab", "xy"), _random_mealy(rng, "ab", "uv")
        joint = transducer_product((first, second), normalize=False)
        for length in range(4):
            for left_word, right_word in itertools.product(itertools.product("ab", repeat=length), repeat=2):
                try:
                    expected = {(a, b) for a in first.transduce(left_word) for b in second.transduce(right_word)}
                except InfiniteTransductionError:
                    continue
                outputs = joint.transduce(tuple(zip(left_word, right_word, strict=True)))
                projected = {
                    tuple(tuple(part[i] for part in output if part[i] is not EPSILON) for i in range(2))
                    for output in outputs
                }
                assert projected == expected
