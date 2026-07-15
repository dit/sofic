"""Tests for cmpy-compatible process constructors."""

from __future__ import annotations

import pytest

import sofic.examples.processes as processes
from sofic.automata.transducers import MealyMachine
from sofic.generators.base import HiddenMarkovModel
from sofic.graph import EPSILON


def test_cmpy_process_constructor_names_are_exported():
    expected = {
        "ABC",
        "AFC",
        "AFC2",
        "BandMerging",
        "BeadsOnNecklace",
        "BeforeAfter",
        "BinaryMarkovChain",
        "Butterfly",
        "Cantor",
        "CoupledGMPs",
        "Ehrenfest",
        "Even",
        "FairCoin",
        "GoldenMean",
        "GoldenMeanGHMM",
        "LogicMachine",
        "Nemo",
        "Odd",
        "Period",
        "Periodic",
        "PerturbedCoin",
        "RandomEven",
        "RandomGoldenMean",
        "RIP",
        "SNS",
        "UncoupledGMPs",
        "uniform_mealyhmm",
        "uniform_mealymc",
        "GMtoEven",
        "BitFlip",
        "BinaryChannel",
        "Parity",
    }
    assert expected <= set(processes.__all__)
    for name in expected:
        assert hasattr(processes, name)


@pytest.mark.parametrize(
    "constructor",
    [
        processes.ABC,
        processes.BandMerging,
        processes.BeadsOnNecklace,
        processes.BeforeAfter,
        processes.BinaryMarkovChain,
        processes.Butterfly,
        processes.Cantor,
        processes.Ehrenfest,
        processes.Even,
        processes.FairCoin,
        processes.GoldenMean,
        processes.Nemo,
        processes.Odd,
        processes.PerturbedCoin,
        processes.RIP,
        processes.Rn1C,
        processes.Rn1N,
        processes.RRX,
        processes.RRXRO,
        processes.SNS,
        processes.ThreeHundred,
    ],
)
def test_default_process_constructors_validate(constructor):
    machine = constructor()
    assert isinstance(machine, HiddenMarkovModel)
    machine.validate()


def test_golden_mean_cmpy_topology():
    gm = processes.GoldenMean(bias=0.25)
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
    delay = processes.Delay(length=2, symbols=["0", "1"])
    assert delay.transduce(("1", "0")) == {("0", "0")}


def test_cmpy_style_instance_composition_and_generator_transduction():
    transducer = processes.BitFlip().compose(processes.BitFlip())
    assert transducer.transduce(("0", "1")) == {("0", "1")}

    output = processes.BinaryChannel(p=0.25, q=0.5).transduce_generator(processes.FairCoin())
    assert output.word_probability(("1",)) == pytest.approx(0.375)
