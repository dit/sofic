"""Tests for reusable Hypothesis strategies."""

from __future__ import annotations

import pytest
from hypothesis import given, settings

from pensive.automata.icdfa import dfa_to_icdfa_string
from pensive.testing.strategies import dfas, epsilon_machines


@given(dfa=dfas(max_states=3))
@settings(max_examples=25)
def test_dfas_strategy_generates_valid_complete_icdfas(dfa):
    dfa.validate()

    assert dfa.is_deterministic()
    assert len(dfa.initial_states) == 1
    for state in dfa.states():
        for symbol in dfa.input_alphabet:
            assert len(dfa.delta(state, symbol)) == 1

    encoded = dfa_to_icdfa_string(dfa, symbol_order=tuple(sorted(dfa.input_alphabet)))
    assert encoded.n == len(list(dfa.states()))
    assert encoded.k == len(dfa.input_alphabet)


@given(machine=epsilon_machines(max_states=3))
@settings(max_examples=25)
def test_epsilon_machines_strategy_generates_valid_models(machine):
    machine.validate()

    assert machine.is_unifilar()
    assert machine.is_irreducible()
    assert machine.is_stationary()
    assert machine.observation_alphabet == frozenset({"0", "1"})
    assert sum(machine.initial_distribution.values()) == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"alphabet": ()}, "alphabet must be non-empty"),
        ({"alphabet": ("0", "0")}, "alphabet symbols must be unique"),
        ({"min_states": 0}, "min_states must be positive"),
        ({"min_states": 3, "max_states": 2}, "min_states must be less than or equal to max_states"),
    ],
)
def test_dfas_strategy_validates_arguments(kwargs, message):
    with pytest.raises(ValueError, match=message):
        dfas(**kwargs)


def test_epsilon_machines_strategy_rejects_empty_enumeration():
    with pytest.raises(ValueError, match="no topological epsilon-machine strings"):
        epsilon_machines(alphabet=("0",), min_states=2, max_states=2)
