"""Reproductions of DeDeo, "Effective Theories for Circuits and Automata" (Chaos 21, 037106, 2011)."""

import itertools

import numpy as np
import pytest

from sofic.algebra import (
    TransformationSemigroup,
    classify_elements,
    coupled_random_maps,
    element_walk,
    holonomy_decomposition,
    independent_random_maps,
    is_full_transformation_monoid,
    krohn_rhodes_complexity,
    prime_decomposition,
    walk_information,
)


def register(n, rule):
    """Left-shift register on ``n`` bits: state ``(t_{i-n}, ..., t_{i-1})`` appends ``rule(state)``."""
    states = list(itertools.product((0, 1), repeat=n))
    index = {s: i for i, s in enumerate(states)}
    return tuple(index[s[1:] + (rule(s),)] for s in states)


def truth_table_rule(table):
    return lambda s: table[int("".join(map(str, s)), 2)]


def noisy_register(n, p_one):
    """Stochastic matrix of an ``n``-bit register appending 1 with probability ``p_one[state]``."""
    states = list(itertools.product((0, 1), repeat=n))
    index = {s: i for i, s in enumerate(states)}
    matrix = np.zeros((len(states), len(states)))
    for s in states:
        matrix[index[s], index[s[1:] + (1,)]] += p_one[index[s]]
        matrix[index[s], index[s[1:] + (0,)]] += 1 - p_one[index[s]]
    return matrix


#: Fig. 4a, transcribed from the arXiv TeX source (the 101 -> 010 edge, drawn
#: unlabeled, is that state's only edge and so carries both letters).
FIG_4A = [
    ("000", "001", "ab"),
    ("001", "010", "b"),
    ("100", "001", "b"),
    ("100", "000", "a"),
    ("010", "101", "ab"),
    ("101", "010", "ab"),
    ("001", "011", "a"),
    ("110", "100", "a"),
    ("011", "110", "b"),
    ("110", "101", "b"),
    ("011", "111", "a"),
    ("111", "110", "ab"),
]
FIG_4A_NAMES = {"000": "A", "001": "B", "010": "C", "100": "D", "011": "E", "101": "F", "110": "G", "111": "H"}


def left_shift_powerset(n, **kwargs):
    tables = itertools.product((0, 1), repeat=2**n)
    return TransformationSemigroup({t: register(n, truth_table_rule(t)) for t in tables}, **kwargs)


# -- deterministic results (Secs. III-VI, appendix) ----------------------------------


def test_two_bit_xor_is_a_mod_three_counter():
    xor = TransformationSemigroup({"a": register(2, lambda s: s[0] ^ s[1])})
    assert len(xor) == 3
    assert [level.order for level in holonomy_decomposition(xor).levels] == [3]
    assert krohn_rhodes_complexity(xor).exact == 1


def test_left_shift_powerset_is_the_full_transformation_monoid():
    lsp2 = left_shift_powerset(2)
    assert len(lsp2) == 4**4
    assert is_full_transformation_monoid(lsp2)


def test_three_bit_left_shift_powerset_complexity_without_enumeration():
    lsp3 = left_shift_powerset(3, max_size=1000)
    assert is_full_transformation_monoid(lsp3)
    assert krohn_rhodes_complexity(lsp3).exact == 7
    assert lsp3._elements is None


def test_table_one_three_bit_rows():
    pytest.importorskip("sympy")
    one_letter = TransformationSemigroup({"a": register(3, lambda s: s[0] ^ s[1])})
    assert prime_decomposition(one_letter).simple_group_orders() == (7,)
    two_letters = TransformationSemigroup(
        {"a": register(3, lambda s: s[0] ^ s[1]), "b": register(3, lambda s: s[0] ^ s[2])}
    )
    (factor,) = prime_decomposition(two_letters).groups
    assert (factor.kind, factor.order) == ("nonabelian simple", 168)


def test_figure_four_effective_theory():
    maps = {"a": {}, "b": {}}
    for source, target, letters in FIG_4A:
        for letter in letters:
            maps[letter][FIG_4A_NAMES[source]] = FIG_4A_NAMES[target]
    sg = TransformationSemigroup.from_maps(maps, states="ABCDEFGH")
    hd = holonomy_decomposition(sg)
    assert [level.order for level in hd.levels] == [3, 2, 2]
    top = hd.effective_theory(1)
    names = {prefix: "".join(sorted(states)) for prefix, states in top.superstates.items()}
    # Fig. 4b: superstates AH, BG, DE, each also containing C and F.
    assert sorted(names.values()) == ["ACFH", "BCFG", "CDEF"]
    by_name = {name: prefix for prefix, name in names.items()}

    def step(name, letter):
        (target,) = top.automaton.delta(by_name[name], letter)
        return names[target]

    assert [step("ACFH", "a"), step("BCFG", "a"), step("CDEF", "a")] == ["BCFG", "CDEF", "ACFH"]
    assert {step(name, "b") for name in names.values()} == {"BCFG"}


# -- probabilistic automata (Secs. VII-IX) ---------------------------------------------

XOR_MAP = (0, 3, 1, 2)


def noisy_xor(eps):
    return noisy_register(2, [eps, 1 - eps, 1 - eps, eps])


def test_figure_five_decomposition():
    rep = independent_random_maps(noisy_register(2, [0.5, 1, 1, 0.75]))
    assert sorted(rep.maps("t").values()) == pytest.approx([1 / 8, 1 / 8, 3 / 8, 3 / 8])
    levels = holonomy_decomposition(rep.semigroup()).levels
    assert len(levels) == 4
    assert {level.order for level in levels} == {2, 3}


def _z3(sg):
    xor = next(g for g in sg.generators.values() if g.images == XOR_MAP)
    return {sg.index(xor**k) for k in range(3)}


def test_figure_six_independent_noise_decays_to_resets():
    rep = independent_random_maps(noisy_xor(0.01))
    assert len(rep.maps("t")) == 16
    sg = rep.semigroup()
    assert is_full_transformation_monoid(sg)
    classes = classify_elements(sg)
    z3 = _z3(sg)
    other = sorted(classes["units"] - z3)
    walk = element_walk(rep, 1000)
    on_z3 = walk[:, sorted(z3)].sum(axis=1)
    on_resets = walk[:, sorted(classes["resets"])].sum(axis=1)
    on_other = walk[:, other].sum(axis=1)
    assert on_z3[0] == 1.0 and on_z3[-1] < 1e-6
    assert on_resets[-1] > 0.99
    # The brief window in which other permutations outweigh the resets.
    assert (on_other[1:10] > on_resets[1:10]).all()
    assert on_other[-1] < 1e-6


def test_figure_eight_correlated_noise_stays_reversible():
    rep = coupled_random_maps(noisy_xor(0.01))
    weights = {m.images: w for m, w in rep.maps("t").items()}
    assert weights == pytest.approx({XOR_MAP: 0.99, (1, 2, 0, 3): 0.01})
    sg = rep.semigroup()
    assert sg.is_group() and len(sg) == 12
    walk = element_walk(rep, 5000)
    # Both maps are 3-cycles in one coset of the Klein group, so the walk has
    # period three: uniform on a coset at each step, uniform on A_4 on average.
    assert np.count_nonzero(walk[-1] > 1e-9) == 4
    assert walk[-3:].mean(axis=0) == pytest.approx(np.full(12, 1 / 12), abs=1e-3)
    assert walk[-1, sorted(_z3(sg))].sum() == pytest.approx(0.25, abs=1e-3)


def test_irreversibility_versus_dissipation():
    independent = walk_information(independent_random_maps(noisy_xor(0.01)), 1000)
    correlated = walk_information(coupled_random_maps(noisy_xor(0.01)), 5000)
    assert independent.irreversibility[-1] == pytest.approx(2.0, abs=0.01)
    assert np.allclose(correlated.irreversibility, 0.0)
    assert correlated.dissipation[-1] == pytest.approx(2.0, abs=0.01)


def test_full_transformation_recognition_needs_a_rank_deficient_generator():
    assert not is_full_transformation_monoid(TransformationSemigroup({"t": (1, 0, 2), "c": (1, 2, 0)}))
    assert not is_full_transformation_monoid(TransformationSemigroup({"c": (1, 2, 0), "e": (0, 0, 2)}))
    assert is_full_transformation_monoid(TransformationSemigroup({"t": (1, 0, 2), "c": (1, 2, 0), "e": (0, 0, 2)}))
