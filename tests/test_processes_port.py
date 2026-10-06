"""Tests for cmpy-compatible process constructors."""

from __future__ import annotations

import pytest

import sofic.examples.processes as processes
from sofic.automata.transducers import MealyMachine
from sofic.examples import fair_coin
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
        "butterfly_two_branch",
        "cantor",
        "coupled_gmps",
        "ehrenfest",
        "golden_mean_forbid_00",
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
        processes.butterfly_two_branch,
        processes.cantor,
        processes.ehrenfest,
        processes.golden_mean_forbid_00,
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
    gm = processes.golden_mean_forbid_00(bias=0.25)
    edges = {(t.source, t.data["emission"], t.target): t.data["prob"] for t in gm.transitions()}
    assert edges[("A", "1", "A")] == pytest.approx(0.75)
    assert edges[("A", "0", "B")] == pytest.approx(0.25)
    assert edges[("B", "1", "A")] == pytest.approx(1.0)


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

    output = processes.binary_channel(p=0.25, q=0.5).transduce_generator(fair_coin())
    assert output.word_probability(("1",)) == pytest.approx(0.375)
