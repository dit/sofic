"""Distromaton, xor automata, universal automaton, and minimal structured DFAs (Myers et al. 2014; Zetzsche et al. 2021)."""

from itertools import product

import pytest
from hypothesis import given, settings

from sofic.automata.algorithms import equivalent
from sofic.automata.canonical import Atomaton, CanonicalRFSA
from sofic.automata.canonical.closures import (
    Distromaton,
    distromaton_from_language,
    is_intersection_closed,
    minimal_structured_dfa,
    minimal_xor_automaton,
    universal_automaton_from_language,
    xor_caba_automaton,
)
from sofic.automata.canonical.residual import ResidualTable
from sofic.automata.dfa import DFA
from sofic.automata.nfa import NFA
from sofic.automata.xor import XorAutomaton
from sofic.exceptions import SoficValidationError
from sofic.graph import ATTR_SYMBOL, EPSILON
from sofic.testing.strategies import dfas


def _size(aut) -> int:
    return len(list(aut.states()))


def _nfa(alphabet: str, edges, initial, accepting) -> NFA:
    nfa = NFA(
        input_alphabet=frozenset(alphabet), initial_states=frozenset(initial), accepting_states=frozenset(accepting)
    )
    for source, target, _ in edges:
        nfa.graph.add_state(source)
        nfa.graph.add_state(target)
    for state in (*initial, *accepting):
        nfa.graph.add_state(state)
    for source, target, symbol in edges:
        nfa.add_transition(source, target, symbol)
    return nfa


def _ends_in_a() -> NFA:
    """(a + b)* a, the running example of Zetzsche et al. (2021)."""
    return _nfa("ab", [(0, 0, "a"), (0, 0, "b"), (0, 1, "a")], [0], [1])


def _b_then_n_symbols(n: int) -> NFA:
    """(a + b)* b (a + b)^n, Example 1.1 of Myers et al. (2014)."""
    edges = [(0, 0, "a"), (0, 0, "b"), (0, 1, "b")]
    edges += [(i, i + 1, s) for i in range(1, n + 1) for s in "ab"]
    return _nfa("ab", edges, [0], [n + 1])


def _words(alphabet, max_length: int):
    for length in range(max_length + 1):
        yield from product(sorted(alphabet), repeat=length)


def _gf2_rank(rows: list[int]) -> int:
    pivots: dict[int, int] = {}
    for row in rows:
        while row and (lead := row.bit_length() - 1) in pivots:
            row ^= pivots[lead]
        if row:
            pivots[row.bit_length() - 1] = row
    return len(pivots)


def _hankel_gf2_rank(dfa: DFA) -> int:
    """GF(2) rank of [uv in L] over all u, v shorter than the minimal DFA.

    A non-zero GF(2) combination of residuals lies in a derivative-closed space of dimension d <= n, so it contains
    a word shorter than n; rows indexed by words shorter than n reach every residual.
    """
    n = _size(dfa.minimize())
    columns = list(_words(dfa.input_alphabet, n - 1))
    rows = [sum(1 << j for j, v in enumerate(columns) if dfa.recognizes(u + v)) for u in columns]
    return _gf2_rank(rows)


def _atom_count(dfa: DFA) -> int:
    """States of the minimal complete DFA of the reverse language: the atoms, including the negative atom."""
    return len(ResidualTable.from_automaton(dfa.reverse())._bfs_order())


def _myers_distromaton_counts(language) -> tuple[int, int, int, int]:
    """The distromaton read off the minimal DFA of the reverse, ordered by inclusion (Myers et al. 2014, Sect. 3.3).

    States are those of the DFA; z -a-> z' iff delta(z', a) includes z; initial = accepting states of the DFA;
    accepting = states included in the start state.
    """
    table = ResidualTable.from_automaton(language.reverse())
    states = table._bfs_order()
    edges = sum(table.includes(z, table.delta[(w, a)]) for z in states for w in states for a in table.alphabet)
    accepting = sum(table.includes(z, table.start) for z in states)
    return len(states), edges, len(table.accepting), accepting


# --- worked examples ---------------------------------------------------------------------------------------------


def test_running_example_of_zetzsche_et_al():
    language = _ends_in_a()
    distromaton = Distromaton.from_language(language)
    assert (_size(distromaton), len(distromaton.initial_states), len(distromaton.accepting_states)) == (
        3,
        1,
        2,
    )  # Fig. 8b
    assert _size(minimal_xor_automaton(language)) == 2  # Fig. 7b
    sizes = {v: _size(minimal_structured_dfa(language, v)) for v in ("semilattice", "lattice", "boolean", "gf2")}
    assert sizes == {"semilattice": 3, "lattice": 4, "boolean": 8, "gf2": 4}  # Figs. 5a, 8a, 3, 7a


def test_xor_caba_automaton_of_the_running_example_has_one_state_per_atom():
    """Three atoms, so three states. Fig. 10 of Zetzsche et al. (2021) uses {4, 6, 7, 8}, which is a generator but
    not a basis: in the notation of their Fig. 3, 4 xor 6 xor 8 is the empty language."""
    language = _ends_in_a()
    for basis in ("residuals", "atoms"):
        xor = xor_caba_automaton(language, basis=basis)
        assert _size(xor) == 3
        assert equivalent(xor.determinize(), language)


@pytest.mark.parametrize("n", [1, 2, 3])
def test_example_1_1_of_myers_et_al(n):
    language = _b_then_n_symbols(n)
    assert _size(language.minimize()) == 2 ** (n + 1)
    assert _size(CanonicalRFSA.from_language(language)) == n + 2
    assert _size(minimal_xor_automaton(language)) == n + 2
    # the distromaton is the jiromaton plus a final sink state Sigma* (Example 3.22)
    distromaton = distromaton_from_language(language)
    assert _size(distromaton) == n + 3
    assert is_intersection_closed(language)  # Example 4.3(1)
    for aut in (distromaton, universal_automaton_from_language(language)):
        assert equivalent(aut, language)


def test_intersection_closed_examples():
    aa = _nfa("a", [(0, 1, "a"), (1, 2, "a")], [0], [2])
    assert is_intersection_closed(aa)  # a singleton, Example 4.3(2) of Myers et al. (2014)
    # {ab, ac, bc, bd}: the residuals {b, c} and {c, d} meet in {c}, which is not a union of residuals
    edges = [(0, 1, "a"), (0, 2, "b"), (1, 3, "b"), (1, 3, "c"), (2, 3, "c"), (2, 3, "d")]
    assert not is_intersection_closed(_nfa("abcd", edges, [0], [3]))


def test_xor_automaton_counts_paths_mod_two():
    xor = XorAutomaton(
        input_alphabet=frozenset("ab"), initial_states=frozenset({0, 1}), accepting_states=frozenset({2})
    )
    for state in range(3):
        xor.graph.add_state(state)
    xor.add_transition(0, 2, "a")
    xor.add_transition(1, 2, "a")
    xor.add_transition(0, 2, "b")
    xor.validate()
    assert not xor.recognizes(("a",))  # two accepting paths
    assert xor.recognizes(("b",))  # one
    assert equivalent(xor.reverse().determinize(), xor.determinize().reverse())


def test_xor_automaton_rejects_epsilon_moves():
    xor = XorAutomaton(input_alphabet=frozenset("a"), initial_states=frozenset({0}))
    xor.graph.add_state(0)
    xor.graph.add_transition(0, 0, **{ATTR_SYMBOL: EPSILON})
    with pytest.raises(SoficValidationError, match="no ε-transitions"):
        xor.validate()


def test_xor_automaton_input_and_round_trip():
    xor = XorAutomaton.from_language(_ends_in_a())
    assert _size(minimal_xor_automaton(xor)) == _size(xor)
    from sofic.serialization import model_from_dict, model_to_dict

    clone = model_from_dict(model_to_dict(xor))
    assert type(clone) is XorAutomaton
    assert equivalent(clone.determinize(), xor.determinize())


# --- properties --------------------------------------------------------------------------------------------------


@settings(max_examples=60, deadline=None, derandomize=True)
@given(dfa=dfas(max_states=4))
def test_canonical_automata_accept_the_language(dfa):
    for aut in (distromaton_from_language(dfa), universal_automaton_from_language(dfa)):
        assert equivalent(aut, dfa)
    for xor in (minimal_xor_automaton(dfa), xor_caba_automaton(dfa), xor_caba_automaton(dfa, basis="atoms")):
        xor.validate()
        assert equivalent(xor.determinize(), dfa)
    small = _atom_count(dfa) <= 6
    for variety in ("semilattice", "lattice", "boolean", "gf2") if small else ("semilattice", "lattice", "gf2"):
        structured = minimal_structured_dfa(dfa, variety)
        assert equivalent(structured, dfa)
        assert _size(structured.minimize()) == _size(dfa.minimize())


@settings(max_examples=60, deadline=None, derandomize=True)
@given(dfa=dfas(max_states=4))
def test_distromaton_matches_the_reverse_dfa_construction(dfa):
    """Two constructions of Myers et al. (2014): join-irreducible meets, and the ordered reverse minimal DFA."""
    distromaton = distromaton_from_language(dfa)
    counts = (
        _size(distromaton),
        len(list(distromaton.transitions())),
        len(distromaton.initial_states),
        len(distromaton.accepting_states),
    )
    assert counts == _myers_distromaton_counts(dfa)


@settings(max_examples=60, deadline=None, derandomize=True)
@given(dfa=dfas(max_states=4))
def test_sizes_of_the_canonical_automata(dfa):
    atoms = _atom_count(dfa)
    distromaton = distromaton_from_language(dfa)
    rfsa = CanonicalRFSA.from_language(dfa)
    assert _size(distromaton) == atoms  # Corollary 3.21
    assert _size(Atomaton.from_language(dfa)) in (atoms, atoms - 1)  # the átomaton drops the negative atom
    assert _size(minimal_xor_automaton(dfa)) == _hankel_gf2_rank(dfa) <= _size(dfa.minimize())
    caba = xor_caba_automaton(dfa)
    assert _size(caba) == atoms
    reached, frontier = set(caba.initial_states), set(caba.initial_states)
    while frontier:
        frontier = {t.target for s in frontier for t in caba.graph.out_transitions(s)} - reached
        reached |= frontier
    assert len(reached) == _size(minimal_xor_automaton(dfa))
    assert _size(distromaton) <= _size(universal_automaton_from_language(dfa))
    if is_intersection_closed(dfa):
        assert _size(distromaton) <= _size(rfsa) + 1  # Corollary 4.6
