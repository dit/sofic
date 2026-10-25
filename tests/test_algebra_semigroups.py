"""Tests for transformation semigroups, Green's relations, and syntactic monoids."""

import pytest
from hypothesis import given
from hypothesis import strategies as st

from sofic.algebra import (
    SINK,
    Transformation,
    TransformationSemigroup,
    is_star_free,
    syntactic_monoid,
    transformation_semigroup,
)
from sofic.automata.dfa import DFA
from sofic.automata.nfa import NFA
from sofic.examples.epsilon_machines import even_process, golden_mean
from sofic.exceptions import SemigroupTooLargeError


def full_transformation_monoid(n):
    """T_n from a transposition, an n-cycle, and an elementary collapse."""
    rest = tuple(range(2, n))
    return TransformationSemigroup(
        {"t": (1, 0, *rest), "c": (*range(1, n), 0), "e": (0, 0, *rest)},
    )


@st.composite
def semigroups(draw, max_degree=5, max_generators=3):
    degree = draw(st.integers(1, max_degree))
    count = draw(st.integers(1, max_generators))
    gens = {
        f"g{i}": tuple(draw(st.lists(st.integers(0, degree - 1), min_size=degree, max_size=degree)))
        for i in range(count)
    }
    return TransformationSemigroup(gens, monoid=draw(st.booleans()))


# -- transformations ----------------------------------------------------------


def test_composition_acts_on_the_right():
    s = Transformation((1, 2, 0))
    t = Transformation((0, 0, 2))
    assert (s * t).images == tuple(t(s(x)) for x in range(3))


def test_transformation_basics():
    t = Transformation((2, 2, 0, 1))
    assert t.rank() == 3
    assert t.image() == {0, 1, 2}
    assert t.kernel() == (frozenset({0, 1}), frozenset({2}), frozenset({3}))
    assert not t.is_permutation()
    assert Transformation.identity(3).is_idempotent()
    assert Transformation.constant(3, 1).is_constant()
    assert t.power(0) == Transformation.identity(4)
    assert t.restrict([0, 2]) == (1, 0)


@given(st.lists(st.integers(0, 5), min_size=6, max_size=6))
def test_index_period_is_the_first_repeat(images):
    s = Transformation(tuple(images))
    m, r = s.index_period()
    assert s ** (m + r) == s**m
    assert all(s ** (k + r) != s**k for k in range(1, m))


# -- semigroups -----------------------------------------------------------------


@pytest.mark.parametrize("n", [2, 3, 4, 5])
def test_full_transformation_monoid_has_n_to_the_n_elements(n):
    assert len(full_transformation_monoid(n)) == n**n


def test_monoid_flag_adjoins_the_identity():
    # Example 7.2 of Egri-Nagy & Nehaniv: |S| = 138 without the identity.
    gens = {
        "s1": (0, 1, 2, 0, 0, 0),
        "s2": (3, 3, 3, 4, 3, 5),
        "s3": (3, 3, 3, 4, 5, 3),
        "s4": (3, 3, 3, 3, 4, 4),
        "s5": (3, 3, 3, 0, 1, 2),
        "s6": (1, 2, 0, 3, 3, 3),
    }
    assert len(TransformationSemigroup(gens, monoid=False)) == 138
    assert len(TransformationSemigroup(gens)) == 139


def test_max_size_guard_raises_and_resets():
    t5 = full_transformation_monoid(5)
    t5._max_size = 100
    with pytest.raises(SemigroupTooLargeError):
        len(t5)
    t5._max_size = 10_000
    assert len(t5) == 3125


def test_bad_generators_are_rejected():
    with pytest.raises(ValueError):
        TransformationSemigroup({"a": (0, 1), "b": (0, 1, 2)})
    with pytest.raises(ValueError):
        TransformationSemigroup({"a": (0, 3)})
    with pytest.raises(ValueError):
        TransformationSemigroup({}, monoid=False)


def test_from_maps_adds_a_sink_only_when_needed():
    total = TransformationSemigroup.from_maps({"a": {0: 1, 1: 0}})
    assert SINK not in total.states
    partial = TransformationSemigroup.from_maps({"a": {0: 1}})
    assert partial.states == (0, 1, SINK)
    assert partial.generators["a"].images == (1, 2, 2)
    with pytest.raises(ValueError):
        TransformationSemigroup.from_maps({"a": {0: 1}}, complete="error")


@given(semigroups())
def test_words_evaluate_to_their_elements(sg):
    for i, element in enumerate(sg):
        assert sg.evaluate(sg.word(i)) == element
        assert sg.index(element) == i


@given(semigroups())
def test_cayley_tables_are_products(sg):
    right, left = sg.right_cayley(), sg.left_cayley()
    gens = [sg.generators[label] for label in sg.generator_labels]
    for i, element in enumerate(sg):
        for k, g in enumerate(gens):
            assert sg[right[i][k]] == element * g
            assert sg[left[i][k]] == g * element


def test_cayley_graph_edges_are_labeled():
    graph = TransformationSemigroup({"a": (1, 0)}).cayley_graph()
    assert {d["symbol"] for _, _, d in graph.edges(data=True)} == {"a"}


# -- Green's relations ------------------------------------------------------------


def test_green_structure_of_t3():
    pytest.importorskip("sympy")
    green = full_transformation_monoid(3).green()
    assert sorted(d.size for d in green.d_classes) == [3, 6, 18]
    assert sorted(int(g.order()) for _, g in green.maximal_subgroups()) == [1, 2, 6]
    assert not green.is_aperiodic()
    assert "D0" in green.egg_box()


@given(semigroups())
def test_green_classes_partition_and_form_egg_boxes(sg):
    green = sg.green()
    n = len(sg)
    for classes in (green.r_classes, green.l_classes, green.h_classes):
        assert sorted(i for c in classes for i in c) == list(range(n))
    assert sorted(i for d in green.d_classes for i in d.elements()) == list(range(n))
    for d in green.d_classes:
        sizes = {len(cell) for row in d.cells for cell in row}
        assert len(sizes) == 1 and 0 not in sizes


@given(semigroups())
def test_aperiodicity_agrees_with_trivial_h_classes(sg):
    assert sg.is_aperiodic() == sg.green().is_aperiodic()


@given(semigroups())
def test_maximal_subgroup_orders_match_their_h_classes(sg):
    pytest.importorskip("sympy")
    green = sg.green()
    for e, group in green.maximal_subgroups():
        assert int(group.order()) == len(green.h_classes[green.h_of[e]])


# -- syntactic monoids --------------------------------------------------------------


def test_syntactic_monoid_known_values():
    assert len(syntactic_monoid("(aa)*", alphabet="a")) == 2
    assert len(syntactic_monoid("(a|b)*", alphabet="ab")) == 1
    assert not syntactic_monoid("(aaa)*", alphabet="a").is_aperiodic()


def test_syntactic_monoid_ignores_the_presentation():
    nfa = NFA.from_regex("(ab)*", alphabet="ab")
    dfa = nfa.determinize()
    assert len(syntactic_monoid(nfa)) == len(syntactic_monoid(dfa)) == len(syntactic_monoid("(ab)*", alphabet="ab"))


def test_unused_symbols_act_as_zero():
    assert len(syntactic_monoid("(aa)*", alphabet="a")) == 2
    assert len(syntactic_monoid("(aa)*", alphabet="ab")) == 3


@st.composite
def star_free_regexes(draw, depth=2):
    """Expressions from words and ``A* w A*`` closed under union and concatenation."""
    word = draw(st.text(alphabet="ab", min_size=1, max_size=3))
    atom = draw(st.sampled_from([word, f"(a|b)*{word}(a|b)*"]))
    if depth == 0 or draw(st.booleans()):
        return atom
    other = draw(star_free_regexes(depth=depth - 1))
    return draw(st.sampled_from([f"({atom})({other})", f"({atom})|({other})"]))


@given(star_free_regexes())
def test_star_free_constructions_are_star_free(regex):
    assert is_star_free(regex, alphabet="ab")


@pytest.mark.parametrize("period", [2, 3, 4])
def test_counting_languages_are_not_star_free(period):
    assert not is_star_free("(" + "(a|b)" * period + ")*", alphabet="ab")


# -- adapters -----------------------------------------------------------------------


def test_epsilon_machines_convert_through_their_support():
    sg = even_process().transition_semigroup()
    assert sg.states == ("A", "B", SINK)
    assert len(sg) == 7
    assert not sg.is_aperiodic()
    assert golden_mean().transition_semigroup().is_aperiodic()


def test_dfa_transition_semigroup():
    dfa = DFA(input_alphabet=frozenset("ab"), initial_states=frozenset({0}), accepting_states=frozenset({0}))
    dfa.graph.add_state(0)
    dfa.graph.add_state(1)
    dfa.add_transition(0, 1, "a")
    dfa.add_transition(1, 0, "a")
    dfa.add_transition(0, 0, "b")
    dfa.add_transition(1, 1, "b")
    sg = dfa.transition_semigroup()
    assert len(sg) == 2
    assert transformation_semigroup(sg) is sg


def test_nondeterministic_machines_are_rejected():
    nfa = NFA(input_alphabet=frozenset("a"), initial_states=frozenset({0}), accepting_states=frozenset({0}))
    nfa.add_transition(0, 0, "a")
    nfa.add_transition(0, 1, "a")
    with pytest.raises(ValueError, match="not unifilar"):
        transformation_semigroup(nfa)
