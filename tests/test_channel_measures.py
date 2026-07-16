"""Tests for channel information measures on transducers."""

import pytest

from sofic import EpsilonTransducer, MealyHMM
from sofic.examples.processes import BinaryChannel, Delay
from sofic.generators.channel_measures import (
    channel_statistical_complexity,
    directed_information,
    driven_entropy_rate,
    transfer_entropy,
)


def _iid_input() -> MealyHMM:
    inp = MealyHMM(observation_alphabet=frozenset({"0", "1"}), initial_distribution={"S": 1.0})
    inp.graph.add_state("S")
    inp.add_transition("S", "S", "0", 0.5)
    inp.add_transition("S", "S", "1", 0.5)
    inp.validate()
    return inp


def test_memoryless_channel_zero_complexity():
    eps = EpsilonTransducer.from_channel(BinaryChannel(0.1, 0.2))
    assert channel_statistical_complexity(eps, _iid_input()) == pytest.approx(0.0, abs=1e-9)


def test_memoryless_channel_zero_transfer_entropy():
    eps = EpsilonTransducer.from_channel(BinaryChannel(0.1, 0.2))
    assert transfer_entropy(eps, _iid_input()) == pytest.approx(0.0, abs=1e-9)


def test_memory_channel_positive_complexity():
    eps = EpsilonTransducer.from_channel(Delay(1))
    assert channel_statistical_complexity(eps, _iid_input()) > 0.5


def test_driven_entropy_rate_bounds():
    eps = EpsilonTransducer.from_channel(BinaryChannel(0.1, 0.2))
    rate = driven_entropy_rate(eps, _iid_input())
    assert 0.0 <= rate <= 1.0 + 1e-9


def test_directed_information_positive():
    eps = EpsilonTransducer.from_channel(BinaryChannel(0.1, 0.2))
    assert directed_information(eps, _iid_input(), length=2) > 0.0


def test_method_dispatch_matches_functions():
    eps = EpsilonTransducer.from_channel(BinaryChannel(0.1, 0.2))
    inp = _iid_input()
    assert eps.statistical_complexity(inp) == pytest.approx(channel_statistical_complexity(eps, inp))
    assert eps.transfer_entropy(inp) == pytest.approx(transfer_entropy(eps, inp))
