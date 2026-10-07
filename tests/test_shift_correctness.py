"""Regression tests for shift-presentation correctness fixes."""

from __future__ import annotations

import numpy as np
import pytest

from sofic.examples.shifts import dyck_shift_order, sofic_dyck_fig1_shift, sofic_dyck_nondeterminizable_shift
from sofic.generators.moore import MooreHMM
from sofic.graph import ATTR_EMISSION_DIST, ATTR_PROB
from sofic.shifts.markov_dyck import MarkovDyckShift
from sofic.shifts.sft import ShiftOfFiniteType
from sofic.shifts.sofic import SoficShift
from sofic.shifts.tmc import TopologicalMarkovChain


def test_sofic_entropy_ignores_parallel_same_label_edges():
    shift = SoficShift(symbol_alphabet=frozenset({"0"}))
    shift.graph.add_state("s")
    shift.add_transition("s", "s", "0")
    shift.add_transition("s", "s", "0")
    assert shift.topological_entropy() == pytest.approx(0.0, abs=1e-12)


def test_sofic_entropy_uses_right_resolving_presentation():
    # Two states both emitting 0 and 1 into each other: the full 2-shift, but 4 paths per step.
    shift = SoficShift(symbol_alphabet=frozenset({"0", "1"}))
    for source in "ab":
        for target in "ab":
            shift.add_transition(source, target, "0" if target == "a" else "1")
            shift.add_transition(source, target, "1" if target == "a" else "0")
    assert shift.topological_entropy() == pytest.approx(1.0)


def test_tmc_keeps_edge_shift_entropy():
    tmc = TopologicalMarkovChain.from_adjacency(np.array([[2]]), symbol_alphabet=frozenset({"0"}))
    assert tmc.topological_entropy() == pytest.approx(1.0)
    assert tmc.to_sofic_shift().topological_entropy() == pytest.approx(0.0, abs=1e-12)


def test_sft_entropy_from_forbidden_words():
    sft = ShiftOfFiniteType.from_forbidden_words({("1", "1")}, frozenset({"0", "1"}))
    assert sft.topological_entropy() == pytest.approx(np.log2((1 + np.sqrt(5)) / 2))


def test_tmc_parry_measure_with_multiplicity_attains_h_top():
    tmc = TopologicalMarkovChain.from_adjacency(np.array([[2, 1], [1, 0]]), symbol_alphabet=frozenset("ab"))
    parry = tmc.parry_measure()
    assert parry.entropy_rate() == pytest.approx(tmc.topological_entropy(), abs=1e-9)
    labels = {t.data["emission"] for t in parry.transitions()}
    assert labels == {("a", 0), ("a", 1), ("b", 0)}


def test_tmc_from_adjacency_symbols_are_sorted_and_cls_is_honored():
    class Sub(TopologicalMarkovChain):
        pass

    tmc = Sub.from_adjacency(np.array([[1, 1], [1, 0]]), symbol_alphabet=frozenset({"y", "x"}))
    assert type(tmc) is Sub
    labels = {(t.source, t.target): t.data["symbol"] for t in tmc.transitions()}
    targets = sorted({t for _, t in labels})
    assert {labels[key] for key in labels if key[1] == targets[0]} == {"x"}
    assert {labels[key] for key in labels if key[1] == targets[1]} == {"y"}


def test_factor_language_excludes_non_extendable_words():
    sft = ShiftOfFiniteType.from_forbidden_words({("0", "0"), ("0", "1")}, frozenset({"0", "1"}))
    assert set(sft.factor_language(1)) == {("1",)}
    assert set(sft.factor_language(3)) == {("1", "1", "1")}
    assert sft.forbidden_words(length=1) == frozenset({("0",)})


def test_trim_transient_removes_states_off_bi_infinite_paths():
    shift = SoficShift(symbol_alphabet=frozenset({"a"}))
    shift.add_transition("source", "loop", "a")
    shift.add_transition("loop", "loop", "a")
    shift.add_transition("loop", "sink", "a")
    assert set(shift.trim_transient().states()) == {"loop"}


@pytest.mark.parametrize(
    "shift",
    [
        MarkovDyckShift.from_adjacency(np.array([[1, 1], [1, 0]])),
        dyck_shift_order(2),
        sofic_dyck_fig1_shift(),
        sofic_dyck_nondeterminizable_shift(),
    ],
)
def test_dyck_reverse_is_mirror_language(shift):
    reversed_shift = shift.reverse()
    reversed_shift.validate()
    assert type(reversed_shift) is type(shift)
    assert reversed_shift.call_alphabet == shift.return_alphabet
    assert reversed_shift.return_alphabet == shift.call_alphabet
    for length in range(1, 6):
        mirrored = {word[::-1] for word in reversed_shift.factor_language(length)}
        assert mirrored == set(shift.factor_language(length))


def test_dyck_shift_reverse_rejects_mirrored_mismatch():
    reversed_shift = dyck_shift_order(2).reverse()
    assert not reversed_shift.is_admissible_word(("b2", "a1"))
    assert reversed_shift.is_admissible_word(("b1", "a1"))
    assert reversed_shift.is_admissible_word(("a1", "b2"))


def test_copy_deep_copies_attribute_dicts():
    moore = MooreHMM(initial_distribution={"A": 1.0}, observation_alphabet=frozenset({0, 1}))
    moore.graph.add_state("A", **{ATTR_EMISSION_DIST: {0: 0.5, 1: 0.5}})
    moore.graph.add_transition("A", "A", **{ATTR_PROB: 1.0, "meta": {"tag": 1}})
    clone = moore.copy()
    clone.graph.nx.nodes["A"][ATTR_EMISSION_DIST][0] = 0.9
    next(iter(clone.graph.nx.edges(data=True)))[2]["meta"]["tag"] = 2
    assert moore.graph.state_attrs("A")[ATTR_EMISSION_DIST][0] == 0.5
    assert next(iter(moore.graph.nx.edges(data=True)))[2]["meta"]["tag"] == 1


def test_copy_preserves_epsilon_sentinel():
    from sofic.graph import ATTR_SYMBOL, EPSILON

    shift = SoficShift(symbol_alphabet=frozenset({"a"}))
    shift.graph.add_transition("p", "q", **{ATTR_SYMBOL: EPSILON})
    clone = shift.copy()
    assert next(iter(clone.transitions())).data[ATTR_SYMBOL] is EPSILON
