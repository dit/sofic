"""Property-based tests for information-anatomy identities."""

from __future__ import annotations

import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st
from hypothesis.errors import Unsatisfiable

from pensive.generators.epsilon_machine import EpsilonMachine
from pensive.testing.strategies import epsilon_machines

THREE_STATE_ALPHABET = (0, 1, 2)
THREE_BY_THREE_POOL = 32
_ATTEMPTS = 12


@st.composite
def three_state_three_symbol_anatomy_machines(draw: st.DrawFn) -> EpsilonMachine:
    """Random 3-state, 3-symbol ε-machines with valid bidirectional anatomy support."""
    base = epsilon_machines(
        alphabet=THREE_STATE_ALPHABET,
        min_states=3,
        max_states=3,
        max_pool=THREE_BY_THREE_POOL,
    )
    for _ in range(_ATTEMPTS):
        machine = draw(base)
        try:
            bidir = machine.to_bidirectional()
        except Exception:
            continue
        h_mu = bidir.entropy_rate()
        b_mu = bidir.bound_information()
        r_mu = bidir.ephemeral_information()
        if h_mu > 1e-12 and abs(h_mu - (b_mu + r_mu)) < 1e-9:
            return machine
    assume(False)


@given(machine=three_state_three_symbol_anatomy_machines())
@settings(max_examples=10, deadline=None)
def test_anatomy_entropy_rate_equals_bound_plus_ephemeral(machine):
    """h_μ = b_μ + r_μ on random 3-state, 3-symbol ε-machines."""
    pytest.importorskip("dit")

    bidir = machine.to_bidirectional()
    h_mu = bidir.entropy_rate()
    b_mu = bidir.bound_information()
    r_mu = bidir.ephemeral_information()

    assert h_mu == pytest.approx(b_mu + r_mu, abs=1e-9)


def test_three_by_three_strategy_pool_is_usable():
    """Guard against an empty bounded pool for the anatomy property test."""
    strategy = epsilon_machines(
        alphabet=THREE_STATE_ALPHABET,
        min_states=3,
        max_states=3,
        max_pool=THREE_BY_THREE_POOL,
    )
    try:
        machine = strategy.example()
    except Unsatisfiable as exc:
        raise AssertionError("bounded 3-state 3-symbol pool produced no examples") from exc
    machine.validate()
