"""Tests for cmpy-compatible process constructors."""

from __future__ import annotations

import pytest

import sofic.examples.processes as processes
from sofic.automata.transducers import MealyMachine
from sofic.examples import bernoulli, golden_mean
from sofic.generators.base import HiddenMarkovModel
from sofic.graph import EPSILON


def test_cmpy_process_constructor_names_are_exported():
    expected = {
        "afc",
        "afc2",
        "band_merging",
        "beads_on_necklace",
        "before_after",
        "binary_markov_chain",
        "cantor",
        "coupled_gmps",
        "ehrenfest",
        "golden_mean_ghmm",
        "logic_machine",
        "odd",
        "period",
        "periodic",
        "perturbed_coin",
        "random_even",
        "random_golden_mean",
        "rip",
        "sns",
        "uncoupled_gmps",
        "uniform_mealyhmm",
        "uniform_mealymc",
        "gm_to_even",
        "bit_flip",
        "binary_channel",
        "parity",
    }
    assert expected <= set(processes.__all__)
    for name in expected:
        assert hasattr(processes, name)


@pytest.mark.parametrize(
    "constructor",
    [
        processes.band_merging,
        processes.beads_on_necklace,
        processes.before_after,
        processes.binary_markov_chain,
        processes.cantor,
        processes.ehrenfest,
        processes.odd,
        processes.perturbed_coin,
        processes.rip,
        processes.rn1c,
        processes.rn1n,
        processes.rrx,
        processes.rrxro,
        processes.sns,
        processes.three_hundred,
    ],
)
def test_default_process_constructors_validate(constructor):
    machine = constructor()
    assert isinstance(machine, HiddenMarkovModel)
    machine.validate()


def test_golden_mean_cmpy_topology():
    gm = golden_mean(0.75)
    edges = {(t.source, t.data["emission"], t.target): t.data["prob"] for t in gm.transitions()}
    assert edges[("A", "0", "A")] == pytest.approx(0.75)
    assert edges[("A", "1", "B")] == pytest.approx(0.25)
    assert edges[("B", "0", "A")] == pytest.approx(1.0)


@pytest.mark.parametrize(
    "factory",
    [
        lambda: processes.stretched_gm(1),
        lambda: processes.rk_gm(1, 1),
        lambda: processes.rn_gm(1, 1),
        processes.nonunifilar_golden_mean,
    ],
)
def test_golden_mean_family_base_cases_equal_golden_mean(factory):
    machine = factory()
    assert machine.word_probability(("1", "1")) == 0.0
    assert machine.word_probability(("0", "0")) > 0.0
    assert machine.is_equal_process(golden_mean())


@pytest.mark.parametrize(
    ("factory", "allowed", "forbidden"),
    [
        (lambda: processes.stretched_gm(3), "011101", "0110"),
        (lambda: processes.rk_gm(3, 2), "0111001", "11101"),
        (lambda: processes.rn_gm(3, 2), "0110001", "11001"),
        (processes.random_golden_mean, "01010", "11"),
    ],
)
def test_golden_mean_family_block_structure(factory, allowed, forbidden):
    machine = factory()
    assert machine.word_probability(tuple(allowed)) > 0.0
    assert machine.word_probability(tuple(forbidden)) == 0.0


def test_nonunifilar_golden_mean_bias_is_the_probability_of_one():
    assert processes.nonunifilar_golden_mean(0.3, 0.5).is_equal_process(golden_mean(0.7))


def test_golden_mean_ghmm_matches_golden_mean_word_probabilities():
    import itertools

    ghmm = processes.golden_mean_ghmm()
    gm = golden_mean()
    for n in range(1, 7):
        for word in itertools.product("01", repeat=n):
            assert ghmm.word_probability(word) == pytest.approx(gm.word_probability(word), abs=1e-12)


def test_gm_to_even_reads_golden_mean_words():
    transducer = processes.gm_to_even()
    assert transducer.transduce(("0", "1", "0", "0")) == {("0", "1", "1", "0")}
    assert transducer.transduce(("1", "1")) == set()


def test_all_transducer_constructors_return_mealy_machine():
    for name in processes.transducers:
        machine = getattr(processes, name)()
        assert isinstance(machine, MealyMachine)
        machine.validate()


def test_all_transducer_constructors_have_cmpy_style_alphabets_and_rows():
    for name in processes.transducers:
        machine = getattr(processes, name)()
        inputs, outputs = machine.alphabets()

        assert EPSILON not in inputs
        assert EPSILON not in outputs
        assert inputs <= machine.input_alphabet
        assert outputs <= machine.output_alphabet
        machine.validate_stochastic()


def test_delay_transducer_delays_symbols():
    delay = processes.delay(length=2, symbols=["0", "1"])
    assert delay.transduce(("1", "0")) == {("0", "0")}


def test_cmpy_style_instance_composition_and_generator_transduction():
    transducer = processes.bit_flip().compose(processes.bit_flip())
    assert transducer.transduce(("0", "1")) == {("0", "1")}

    output = processes.binary_channel(p=0.25, q=0.5).transduce_generator(bernoulli())
    assert output.word_probability(("1",)) == pytest.approx(0.375)
