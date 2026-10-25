"""Tests for the holonomy decomposition, cascades, and Krohn-Rhodes complexity."""

import math

import pytest
from hypothesis import given
from hypothesis import strategies as st
from sympy.combinatorics.named_groups import AlternatingGroup, DirectProduct, SymmetricGroup

from sofic.algebra import (
    TransformationSemigroup,
    holonomy_decomposition,
    holonomy_information,
    krohn_rhodes_complexity,
    prime_decomposition,
)
from sofic.algebra._groups import composition_factor_orders
from sofic.automata.wheeler import WheelerError
from sofic.examples.epsilon_machines import (
    butterfly_process,
    even_process,
    golden_mean,
    noisy_random_phase_slip,
)
from sofic.shifts import factor_krohn_rhodes_complexity, wheeler_cover
from sofic.shifts.sofic import SoficShift

EXAMPLE_7_1 = {"s1": (1, 0, 2), "s2": (0, 1, 1)}
EXAMPLE_7_2 = {
    "s1": (0, 1, 2, 0, 0, 0),
    "s2": (3, 3, 3, 4, 3, 5),
    "s3": (3, 3, 3, 4, 5, 3),
    "s4": (3, 3, 3, 3, 4, 4),
    "s5": (3, 3, 3, 0, 1, 2),
    "s6": (1, 2, 0, 3, 3, 3),
}


def full_transformation_monoid(n, **kwargs):
    rest = tuple(range(2, n))
    return TransformationSemigroup({"t": (1, 0, *rest), "c": (*range(1, n), 0), "e": (0, 0, *rest)}, **kwargs)


def shift(edges):
    s = SoficShift(symbol_alphabet=frozenset(symbol for _, _, symbol in edges))
    for source, target, symbol in edges:
        s.add_transition(source, target, symbol)
    return s


EVEN_SHIFT = [("A", "A", "0"), ("A", "B", "1"), ("B", "A", "1")]
GOLDEN_MEAN_SHIFT = [("A", "A", "0"), ("A", "B", "1"), ("B", "A", "0")]


@st.composite
def semigroups(draw, max_degree=5, max_generators=3):
    degree = draw(st.integers(1, max_degree))
    count = draw(st.integers(1, max_generators))
    gens = {
        f"g{i}": tuple(draw(st.lists(st.integers(0, degree - 1), min_size=degree, max_size=degree)))
        for i in range(count)
    }
    return TransformationSemigroup(gens)


# -- holonomy decomposition ------------------------------------------------------


def test_egri_nagy_nehaniv_example_7_1():
    hd = holonomy_decomposition(TransformationSemigroup(EXAMPLE_7_1))
    assert len(hd.image_set) == 5
    assert [level.order for level in hd.levels] == [1, 2]


def test_egri_nagy_nehaniv_example_7_2():
    hd = holonomy_decomposition(TransformationSemigroup(EXAMPLE_7_2))
    assert len(hd.image_set) == 19
    assert hd.n_levels == 5
    groups = [sorted(int(g.order()) for g in level.groups) for level in hd.levels]
    assert groups == [[1], [3], [2, 6], [2, 3], [1]]


@pytest.mark.parametrize("n", [2, 3, 4, 5])
def test_full_transformation_monoid_has_symmetric_holonomy(n):
    hd = holonomy_decomposition(full_transformation_monoid(n))
    assert [level.order for level in hd.levels] == [math.factorial(k) for k in range(n, 1, -1)]


def test_holonomy_needs_only_generators():
    hd = holonomy_decomposition(full_transformation_monoid(5, max_size=10))
    assert hd.n_levels == 4
    assert hd.semigroup._elements is None


@given(semigroups())
def test_encode_and_decode_are_inverse(sg):
    hd = holonomy_decomposition(sg)
    for chain in hd.chains():
        assert hd.decode(hd.encode(chain)) == chain
    for state in sg.states:
        assert hd.state(hd.coordinates(state)) == state


@given(semigroups())
def test_heights_respect_strict_subduction(sg):
    hd = holonomy_decomposition(sg)
    for parent, tiles in hd.tiles.items():
        assert all(hd.height[tile] < hd.height[parent] for tile in tiles)
        assert frozenset().union(*tiles) == parent


def test_invalid_coordinates_are_rejected():
    hd = holonomy_decomposition(TransformationSemigroup(EXAMPLE_7_1))
    with pytest.raises(ValueError):
        hd.decode((None, None))
    with pytest.raises(ValueError):
        hd.decode((0,))


# -- cascades -------------------------------------------------------------------------


@given(semigroups())
def test_the_holonomy_cascade_emulates_the_semigroup(sg):
    hd = holonomy_decomposition(sg)
    cascade = hd.to_cascade()
    assert cascade.emulates(sg)
    for coords in cascade.states:
        for label in sg.generator_labels:
            assert cascade.act(coords, label) == hd.act(coords, label)


@given(semigroups(), st.lists(st.integers(0, 2), max_size=6))
def test_cascade_runs_track_the_original_machine(sg, picks):
    labels = sg.generator_labels
    word = [labels[i % len(labels)] for i in picks]
    cascade = holonomy_decomposition(sg).to_cascade()
    g = sg.evaluate(word)
    for x, state in enumerate(sg.states):
        assert cascade.decoding[cascade.run(word, state)] == sg.states[g(x)]


@given(semigroups(), st.integers(0, 6))
def test_truncation_commutes_with_every_input(sg, k):
    cascade = holonomy_decomposition(sg).to_cascade()
    k = min(k, cascade.n_levels)
    coarse = cascade.truncate(k)
    for coords, state in cascade.decoding.items():
        superstate = coarse.decoding[coords[:k]]
        assert isinstance(superstate, frozenset) and state in superstate
        for label in sg.generator_labels:
            assert cascade.act(coords, label)[:k] == coarse.act(coords[:k], label)


def test_effective_theory_superstates_overlap():
    hd = holonomy_decomposition(TransformationSemigroup(EXAMPLE_7_2))
    top = hd.effective_theory(1)
    assert sorted(map(sorted, top.superstates.values())) == [[0, 1, 2, 3], [3, 4, 5]]
    assert len(list(top.automaton.states())) == 2
    assert hd.effective_theory(0).superstates == {(): frozenset(range(6))}
    assert len(hd.effective_theory(hd.n_levels).superstates) == len(hd.to_cascade().states)


def test_effective_theory_of_a_process_reports_lumpability():
    machine = noisy_random_phase_slip()
    top = holonomy_decomposition(machine).effective_theory(1, machine=machine)
    assert sum(top.distribution.values()) == pytest.approx(1.0)
    assert top.lumpable is False
    finest = holonomy_decomposition(machine)
    assert finest.effective_theory(finest.n_levels, machine=machine).lumpable is True


def test_cascade_as_dfa():
    sg = TransformationSemigroup(EXAMPLE_7_2)
    cascade = holonomy_decomposition(sg).to_cascade()
    dfa = cascade.as_dfa(initial=0)
    assert len(list(dfa.states())) == len(cascade.states)
    assert dfa.is_deterministic()


def test_wrong_semigroup_is_not_emulated():
    cascade = holonomy_decomposition(TransformationSemigroup(EXAMPLE_7_1)).to_cascade()
    assert not cascade.emulates(TransformationSemigroup({"s1": (0, 1, 2), "s2": (0, 1, 1)}))


# -- complexity -------------------------------------------------------------------------


@pytest.mark.parametrize("n", [2, 3, 4, 5])
def test_full_transformation_monoid_complexity(n):
    assert krohn_rhodes_complexity(full_transformation_monoid(n)) == (n - 1, n - 1, n - 1)


@pytest.mark.parametrize("n", [2, 3, 5, 6, 7])
def test_cyclic_groups_have_complexity_one(n):
    sg = TransformationSemigroup({"a": (*range(1, n), 0)})
    assert krohn_rhodes_complexity(sg).exact == 1
    expected = [p for p in range(2, n + 1) if n % p == 0 and all(p % q for q in range(2, p))]
    assert set(prime_decomposition(sg).simple_group_orders()) == set(expected)


def test_flip_flop_is_aperiodic():
    flip_flop = TransformationSemigroup({"r0": (0, 0), "r1": (1, 1)})
    assert krohn_rhodes_complexity(flip_flop).exact == 0
    decomposition = prime_decomposition(flip_flop)
    assert decomposition.groups == () and decomposition.needs_flip_flop


def test_lower_bound_degrades_gracefully_when_enumeration_is_capped():
    assert krohn_rhodes_complexity(full_transformation_monoid(5, max_size=10)).exact == 4
    rank_three = TransformationSemigroup(
        {"t": (1, 0, 2, 3, 4), "c": (1, 2, 3, 4, 0), "e": (0, 0, 0, 3, 4)}, max_size=10
    )
    bounds = krohn_rhodes_complexity(rank_three)
    assert bounds.lower == 1 and bounds.upper > 1 and bounds.exact is None


@given(semigroups())
def test_complexity_bounds_are_consistent(sg):
    bounds = krohn_rhodes_complexity(sg)
    assert 0 <= bounds.lower <= bounds.upper
    assert (bounds.upper == 0) == sg.is_aperiodic()
    assert bounds.exact == (bounds.lower if bounds.lower == bounds.upper else None)


def test_a5_is_a_nonabelian_simple_prime():
    sg = TransformationSemigroup({"a": (1, 2, 3, 4, 0), "b": (1, 2, 0, 3, 4)})
    (factor,) = prime_decomposition(sg).groups
    assert (factor.kind, factor.order) == ("nonabelian simple", 60)
    assert not prime_decomposition(sg).needs_flip_flop


@pytest.mark.parametrize(
    ("group", "orders"),
    [
        (AlternatingGroup(5), [60]),
        (SymmetricGroup(5), [2, 60]),
        (SymmetricGroup(6), [2, 360]),
        (SymmetricGroup(4), [2, 2, 2, 3]),
        (DirectProduct(AlternatingGroup(5), AlternatingGroup(5)), [60, 60]),
        (DirectProduct(AlternatingGroup(5), SymmetricGroup(3)), [2, 3, 60]),
    ],
)
def test_composition_factor_orders(group, orders):
    assert composition_factor_orders(group) == orders


# -- processes and shifts ----------------------------------------------------------------


def test_even_and_golden_mean_processes():
    assert even_process().krohn_rhodes_complexity().exact == 1
    assert golden_mean().krohn_rhodes_complexity().exact == 0


def test_shift_complexity_ignores_the_presentation():
    redundant = shift([*EVEN_SHIFT, ("A", "C", "1"), ("C", "A", "1")])
    assert not redundant.is_unifilar()
    assert factor_krohn_rhodes_complexity(shift(EVEN_SHIFT)).exact == 1
    assert redundant.krohn_rhodes_complexity().exact == 1
    assert shift(GOLDEN_MEAN_SHIFT).krohn_rhodes_complexity().exact == 0


def test_non_right_resolving_shifts_use_the_fischer_cover():
    redundant = shift([*EVEN_SHIFT, ("A", "C", "1"), ("C", "A", "1")])
    assert len(redundant.transition_semigroup()) == len(shift(EVEN_SHIFT).transition_semigroup())


def test_wheeler_failure_names_the_obstruction():
    with pytest.raises(WheelerError, match="not star-free"):
        wheeler_cover(shift(EVEN_SHIFT))


@pytest.mark.parametrize("factory", [even_process, golden_mean, noisy_random_phase_slip, butterfly_process])
def test_holonomy_information_splits_statistical_complexity(factory):
    machine = factory()
    info = holonomy_information(machine)
    assert info.total == pytest.approx(machine.statistical_complexity())
    assert sum(info.level_entropies) == pytest.approx(info.total)
    assert all(h >= -1e-12 for h in info.level_entropies)
