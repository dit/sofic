"""Tests for quasi-realization conversions."""

import numpy as np
import pytest

from pensive.generators.nmachine import NMachine
from pensive.generators.quasi_realization import QuasiRealization
from pensive.graph import ATTR_EMISSION, ATTR_QUASIPROB


def _nmachine() -> NMachine:
    nm = NMachine(
        initial_quasidistribution={"q0": 1.0},
        observation_alphabet=frozenset({"0"}),
    )
    nm.graph.add_state("q0")
    nm.graph.add_transition("q0", "q0", **{ATTR_QUASIPROB: 1.0, ATTR_EMISSION: "0"})
    return nm


def test_from_nmachine_round_trip():
    qr = QuasiRealization.from_nmachine(_nmachine())
    rebuilt = qr.to_nmachine()
    assert rebuilt.word_probability(("0",)) == pytest.approx(1.0)


def test_word_probability_via_matrices():
    qr = QuasiRealization(
        pi=np.array([1.0]),
        tau=np.array([1.0]),
        symbol_maps={"0": np.array([[1.0]])},
    )
    assert qr.word_probability(("0", "0")) == pytest.approx(1.0)
