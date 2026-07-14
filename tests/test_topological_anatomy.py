"""Tests for the topological information anatomy of sofic shifts (MME split of h_top)."""

from __future__ import annotations

from math import log2

import numpy as np
import pytest

from pensive.exceptions import UnifilarityError
from pensive.generators.epsilon_machine import EpsilonMachine
from pensive.shifts.sofic import SoficShift
from pensive.shifts.tmc import TopologicalMarkovChain
from pensive.shifts.topological_anatomy import _right_resolving

PHI = (1.0 + 5.0**0.5) / 2.0
LOG2_PHI = log2(PHI)


def full_shift() -> SoficShift:
    """Full 2-shift: one state, two self-loops."""
    shift = SoficShift(symbol_alphabet=frozenset({0, 1}))
    shift.graph.add_state("S")
    shift.add_transition("S", "S", 0)
    shift.add_transition("S", "S", 1)
    return shift


def golden_mean_shift() -> SoficShift:
    """Golden-mean SFT (forbid ``11``), right-resolving presentation."""
    shift = SoficShift(symbol_alphabet=frozenset({0, 1}))
    for state in ("A", "B"):
        shift.graph.add_state(state)
    shift.add_transition("A", "A", 0)
    shift.add_transition("A", "B", 1)
    shift.add_transition("B", "A", 0)
    return shift


def even_shift() -> SoficShift:
    """Even shift (even runs of 0 between 1s): the canonical sofic, non-SFT shift."""
    shift = SoficShift(symbol_alphabet=frozenset({0, 1}))
    for state in ("E", "O"):
        shift.graph.add_state(state)
    shift.add_transition("E", "E", 1)
    shift.add_transition("E", "O", 0)
    shift.add_transition("O", "E", 0)
    return shift


def test_full_shift_is_all_ephemeral():
    anatomy = full_shift().topological_anatomy()
    assert anatomy["h_top"] == pytest.approx(1.0, abs=1e-9)
    assert anatomy["r_top"] == pytest.approx(1.0, abs=1e-9)
    assert anatomy["b_top"] == pytest.approx(0.0, abs=1e-9)
    assert anatomy["excess_entropy"] == pytest.approx(0.0, abs=1e-9)


def test_golden_mean_anatomy_splits_h_top():
    anatomy = golden_mean_shift().topological_anatomy()
    assert anatomy["h_top"] == pytest.approx(LOG2_PHI, abs=1e-9)
    assert anatomy["b_top"] + anatomy["r_top"] == pytest.approx(anatomy["h_top"], abs=1e-9)
    # Genuine memory: both parts are strictly positive.
    assert anatomy["b_top"] > 1e-6
    assert anatomy["r_top"] > 1e-6


def test_even_shift_is_sofic_and_all_bound():
    shift = even_shift()
    # The even shift is sofic but not a shift of finite type; its MME entropy is log2 phi.
    anatomy = shift.topological_anatomy()
    assert anatomy["h_top"] == pytest.approx(LOG2_PHI, abs=1e-9)
    assert anatomy["b_top"] + anatomy["r_top"] == pytest.approx(anatomy["h_top"], abs=1e-9)
    # Past + future determine the present symbol: nothing is ephemeral.
    assert anatomy["r_top"] == pytest.approx(0.0, abs=1e-9)
    assert anatomy["b_top"] == pytest.approx(LOG2_PHI, abs=1e-9)


def test_split_refines_h_top():
    """Golden-mean and even shift share h_top but have different anatomies."""
    golden = golden_mean_shift().topological_anatomy()
    even = even_shift().topological_anatomy()
    assert golden["h_top"] == pytest.approx(even["h_top"], abs=1e-9)
    assert abs(golden["b_top"] - even["b_top"]) > 0.1


@pytest.mark.parametrize("builder", [full_shift, golden_mean_shift, even_shift])
def test_h_top_matches_topological_entropy_and_is_additive(builder):
    shift = builder()
    anatomy = shift.topological_anatomy()
    assert anatomy["h_top"] == pytest.approx(shift.topological_entropy() / np.log(2), abs=1e-9)
    assert anatomy["h_top"] == pytest.approx(anatomy["b_top"] + anatomy["r_top"], abs=1e-9)


def test_matches_tmc_parry_pipeline():
    """The sofic path equals the TMC -> Parry -> epsilon-machine path on the same object."""
    tmc = TopologicalMarkovChain.from_adjacency(
        np.array([[1, 1], [1, 0]], dtype=float), symbol_alphabet=frozenset({0, 1})
    )
    mine = tmc.to_sofic_shift().topological_anatomy()
    reference = EpsilonMachine.from_hmm(tmc.parry_measure()).information_anatomy()
    assert mine["h_top"] == pytest.approx(reference["entropy_rate"], abs=1e-12)
    assert mine["b_top"] == pytest.approx(reference["bound_mu"], abs=1e-12)
    assert mine["r_top"] == pytest.approx(reference["ephemeral_mu"], abs=1e-12)
    assert mine["excess_entropy"] == pytest.approx(reference["excess_entropy"], abs=1e-12)


def test_symbol_relabel_invariance():
    """Relabeling the alphabet leaves the anatomy unchanged (it is entropy-based)."""
    relabeled = SoficShift(symbol_alphabet=frozenset({"x", "y"}))
    for state in ("A", "B"):
        relabeled.graph.add_state(state)
    relabeled.add_transition("A", "A", "x")
    relabeled.add_transition("A", "B", "y")
    relabeled.add_transition("B", "A", "x")

    base = golden_mean_shift().topological_anatomy()
    other = relabeled.topological_anatomy()
    for key in ("h_top", "b_top", "r_top", "excess_entropy"):
        assert base[key] == pytest.approx(other[key], abs=1e-12)


def test_right_resolving_returns_unifilar_input_unchanged():
    shift = golden_mean_shift()
    assert shift.is_unifilar()
    assert _right_resolving(shift) is shift


def test_determinizes_nondeterministic_full_shift():
    """A nondeterministic full-shift presentation is determinized via the Fischer cover."""
    shift = SoficShift(symbol_alphabet=frozenset({0, 1}))
    for state in ("P", "Q"):
        shift.graph.add_state(state)
    for source in ("P", "Q"):
        for target in ("P", "Q"):
            for symbol in (0, 1):
                shift.add_transition(source, target, symbol)
    assert not shift.is_unifilar()
    anatomy = shift.topological_anatomy()
    assert anatomy["h_top"] == pytest.approx(1.0, abs=1e-9)
    assert anatomy["r_top"] == pytest.approx(1.0, abs=1e-9)
    assert anatomy["b_top"] == pytest.approx(0.0, abs=1e-9)


def test_non_auto_determinizable_presentation_raises():
    """A nondeterministic presentation the bounded Fischer cover cannot resolve raises."""
    shift = SoficShift(symbol_alphabet=frozenset({0, 1}))
    for state in ("u", "v"):
        shift.graph.add_state(state)
    shift.add_transition("u", "u", 0)
    shift.add_transition("u", "v", 0)  # nondeterministic on symbol 0
    shift.add_transition("u", "v", 1)
    shift.add_transition("v", "u", 0)
    assert not shift.is_unifilar()
    with pytest.raises(UnifilarityError):
        shift.topological_anatomy()
