"""Tests for textile systems (Nasu 1995)."""

import pytest

from sofic import SoficShift, TextileSystem
from sofic.examples.processes import BinaryChannel, BitFlip, SlidingNOR
from sofic.shifts.sliding_block_code import SlidingBlockCode


def test_induced_code_recovers_memory():
    textile = TextileSystem.from_transducer(SlidingNOR())
    code = textile.induced_code()
    assert isinstance(code, SlidingBlockCode)
    assert code.memory == 1


def test_induced_code_memoryless():
    textile = TextileSystem.from_transducer(BitFlip())
    code = textile.induced_code()
    assert code.memory == 0
    assert code.apply_word(["0"]) == ("1",)


def test_input_output_shifts():
    textile = TextileSystem.from_transducer(SlidingNOR())
    assert isinstance(textile.input_shift(), SoficShift)
    assert isinstance(textile.output_shift(), SoficShift)


def test_to_transducer_round_trip():
    textile = TextileSystem.from_transducer(SlidingNOR())
    machine = textile.to_transducer()
    assert len(list(machine.states())) == len(list(SlidingNOR().states()))


def test_stochastic_channel_has_no_induced_code():
    textile = TextileSystem.from_transducer(BinaryChannel(0.1, 0.2))
    with pytest.raises(ValueError):
        textile.induced_code()
