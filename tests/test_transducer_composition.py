"""Tests for transducer composition helpers."""

from __future__ import annotations

import pytest

import pensive.examples.processes as processes
from pensive.automata.transducer_operations import compose_tg, compose_tt, transduce_generator


def test_bitflip_composed_with_bitflip_is_identity():
    composed = compose_tt((processes.BitFlip(), processes.BitFlip()))

    composed.validate()
    composed.validate_stochastic()
    assert composed.transduce(("0", "1", "1", "0")) == {("0", "1", "1", "0")}


def test_serial_composition_passes_outputs_to_next_transducer():
    composed = compose_tt((processes.GMtoEven(), processes.BitFlip()))

    assert composed.transduce(("0", "1")) == {("0", "0")}
    assert composed.transduce(("1", "0", "1")) == {("1", "0", "0")}


def test_transducer_completion_emits_error_symbol_for_missing_input():
    completed = processes.GMtoEven().complete(frozenset({"0", "1"}))

    completed.validate()
    assert completed.transduce(("0", "0")) == {("1", "?")}


def test_compose_tg_keeps_joint_input_output_emissions():
    joint = compose_tg(processes.GMtoEven(), processes.GoldenMean(0.5))

    assert joint.word_probability((("0", "1"), ("1", "1"))) == pytest.approx(1 / 3)
    assert joint.word_probability((("0", "1"), ("0", "1"))) == pytest.approx(0.0)


def test_golden_mean_through_gm_to_even_generator():
    output = transduce_generator(processes.GMtoEven(), processes.GoldenMean(0.5))

    assert output.word_probability(("1", "1")) == pytest.approx(1 / 3)
    assert output.word_probability(("1", "0")) == pytest.approx(0.0)


def test_binary_channel_preserves_output_probabilities():
    channel = processes.BinaryChannel(p=0.25, q=0.5)
    output = transduce_generator(channel, processes.FairCoin())

    assert output.word_probability(("1",)) == pytest.approx(0.375)
    assert output.word_probability(("0",)) == pytest.approx(0.625)
