"""Tests for topological Markov chains."""

import pytest

from sofic.graph import ATTR_MULTIPLICITY, ATTR_SYMBOL
from sofic.shifts.tmc import TopologicalMarkovChain


def _tmc() -> TopologicalMarkovChain:
    tmc = TopologicalMarkovChain(symbol_alphabet=frozenset({"a"}))
    tmc.graph.add_state("s")
    tmc.graph.add_transition("s", "s", **{ATTR_SYMBOL: "a", ATTR_MULTIPLICITY: 2})
    return tmc


def test_validate_multiplicity():
    _tmc().validate()


def test_from_adjacency():
    import numpy as np

    tmc = TopologicalMarkovChain.from_adjacency(np.array([[0, 1], [1, 1]]), symbol_alphabet=frozenset({"a", "b"}))
    tmc.validate()
    assert tmc.topological_entropy() > 0


def test_parry_measure():
    import numpy as np

    tmc = TopologicalMarkovChain.from_adjacency(np.array([[0, 1], [1, 1]]), symbol_alphabet=frozenset({"0", "1"}))
    parry = tmc.parry_measure()
    parry.validate()
    assert parry.entropy_rate() == pytest.approx(tmc.topological_entropy() / np.log(2), rel=0.1)
