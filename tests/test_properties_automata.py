"""Property-based and metamorphic tests for finite-automaton constructions.

Every language is compared against :mod:`tests.oracles` on all words up to a
small length; the automata are tiny enough that this is exhaustive in practice.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Hashable
from typing import Any

from hypothesis import given
from hypothesis import strategies as st

from sofic.automata import (
    DFA,
    NFA,
    CanonicalRFSA,
    automaton_to_regex,
    complete,
    determinize,
    equivalent,
    minimize,
)
from sofic.automata.languages import atoms, left_quotient, right_quotient
from sofic.graph import ATTR_SYMBOL, EPSILON
from sofic.testing import dfas, nfas
from tests import oracles

AB = ("0", "1")
SIGMA = frozenset(AB)
N = 5

word = st.lists(st.sampled_from(AB), max_size=3).map(tuple)


def lang(aut: Any, n: int = N, alphabet: tuple[Any, ...] = AB) -> frozenset[tuple[Any, ...]]:
    return frozenset(w for w in oracles.words(alphabet, n) if aut.recognizes(w))


def truth(aut: Any, n: int = N, alphabet: tuple[Any, ...] = AB) -> frozenset[tuple[Any, ...]]:
    return oracles.language(lambda w: oracles.nfa_accepts(aut, w), alphabet, n)


def relabel(
    aut: Any,
    state_map: Callable[[Hashable], Hashable],
    symbol_map: Callable[[Any], Any],
) -> Any:
    result = type(aut)(
        input_alphabet=frozenset(symbol_map(a) for a in aut.input_alphabet),
        initial_states=frozenset(state_map(s) for s in aut.initial_states),
        accepting_states=frozenset(state_map(s) for s in aut.accepting_states),
    )
    for state in aut.states():
        result.graph.add_state(state_map(state))
    for t in aut.transitions():
        symbol = t.data.get(ATTR_SYMBOL)
        mapped = symbol if symbol is EPSILON else symbol_map(symbol)
        result.graph.add_transition(state_map(t.source), state_map(t.target), **{ATTR_SYMBOL: mapped})
    return result


def epsilon_nfa() -> NFA:
    nfa = NFA(input_alphabet=SIGMA, initial_states=frozenset({"e"}), accepting_states=frozenset({"e"}))
    nfa.graph.add_state("e")
    return nfa


def has_empty_residual(accepts: Callable[[tuple[Any, ...]], bool], k: int) -> bool:
    suffixes = oracles.words(AB, k)
    return any(not any(accepts(u + v) for v in suffixes) for u in oracles.words(AB, k))


def star_member(language: frozenset[tuple[Any, ...]], w: tuple[Any, ...]) -> bool:
    reach = [True] + [False] * len(w)
    for end in range(1, len(w) + 1):
        reach[end] = any(reach[start] and w[start:end] in language for start in range(end))
    return reach[len(w)]


# --------------------------------------------------------------------------- Boolean / regular operations


@given(nfas())
def test_recognizes_matches_oracle(a: NFA) -> None:
    assert lang(a) == truth(a)


@given(nfas(), nfas())
def test_boolean_operations_match_brute_force(a: NFA, b: NFA) -> None:
    la, lb = truth(a), truth(b)
    everything = frozenset(oracles.words(AB, N))
    assert lang(a.union(b)) == la | lb
    assert lang(a.intersection(b)) == la & lb
    assert lang(a.complement(SIGMA)) == everything - la
    assert lang(a.difference(b, SIGMA)) == la - lb


@given(nfas(), nfas())
def test_concat_matches_brute_force(a: NFA, b: NFA) -> None:
    la, lb = truth(a), truth(b)
    expected = frozenset(w for w in oracles.words(AB, N) if any(w[:i] in la and w[i:] in lb for i in range(len(w) + 1)))
    assert lang(a.concat(b)) == expected


@given(nfas())
def test_kleene_star_matches_brute_force(a: NFA) -> None:
    la = truth(a)
    assert lang(a.kleene_star()) == frozenset(w for w in oracles.words(AB, N) if star_member(la, w))


@given(nfas())
def test_reverse_matches_brute_force(a: NFA) -> None:
    assert lang(a.reverse()) == frozenset(w[::-1] for w in truth(a))


@given(nfas(), word)
def test_quotients_match_brute_force(a: NFA, u: tuple[Any, ...]) -> None:
    left, right = left_quotient(u, a), right_quotient(a, u)
    for w in oracles.words(AB, 4):
        expected_left = oracles.nfa_accepts(a, u + w)
        expected_right = oracles.nfa_accepts(a, w + u)
        assert (w in left) == expected_left
        assert (w in right) == expected_right


# --------------------------------------------------------------------------- determinize / minimize / equivalent


@given(nfas())
def test_determinize_preserves_language(a: NFA) -> None:
    dfa = determinize(a)
    assert dfa.is_deterministic()
    assert lang(dfa) == truth(a)


@given(nfas())
def test_minimization_algorithms_agree(a: NFA) -> None:
    results = [minimize(a, algorithm=algorithm) for algorithm in ("hopcroft", "moore", "brzozowski")]
    sizes = {len(list(m.states())) for m in results}
    assert len(sizes) == 1
    expected = truth(a)
    for m in results:
        assert lang(m) == expected
        assert equivalent(m, results[0])


@given(nfas(max_states=2) | dfas())
def test_minimal_size_is_brute_force_nerode_count(a: Any) -> None:
    def accepts(w: tuple[Any, ...]) -> bool:
        return oracles.nfa_accepts(a, w)

    k = 3
    expected = oracles.nerode_class_count(accepts, AB, k) - has_empty_residual(accepts, k)
    assert len(list(minimize(a).states())) == expected


@given(nfas())
def test_minimize_is_idempotent(a: NFA) -> None:
    once = minimize(a)
    twice = minimize(once)
    assert len(list(twice.states())) == len(list(once.states()))
    assert equivalent(once, twice)


@given(nfas(max_states=2), nfas(max_states=2))
def test_equivalent_matches_brute_force(a: NFA, b: NFA) -> None:
    # Complete DFAs here have <= 4 states, so a shortest distinguishing word has length <= 6.
    assert equivalent(a, b) == (truth(a, 6) == truth(b, 6))


@given(nfas())
def test_equivalent_to_own_constructions(a: NFA) -> None:
    assert equivalent(a, determinize(a))
    assert equivalent(a, minimize(a, algorithm="brzozowski"))
    assert equivalent(a, a.union(a))


# --------------------------------------------------------------------------- regex, RFSA, atoms


@given(nfas())
def test_regex_round_trip_via_python_re(a: NFA) -> None:
    pattern = re.compile(automaton_to_regex(a))
    for w in oracles.words(AB, N):
        assert (pattern.fullmatch("".join(w)) is not None) == oracles.nfa_accepts(a, w)


@given(nfas())
def test_canonical_rfsa_is_no_larger_than_minimal_dfa(a: NFA) -> None:
    rfsa = CanonicalRFSA.from_language(a)
    assert len(list(rfsa.states())) <= len(list(minimize(a).states()))
    assert lang(rfsa) == truth(a)


@given(nfas())
def test_atoms_partition_sigma_star(a: NFA) -> None:
    pieces = list(atoms(a))
    for w in oracles.words(AB, N):
        assert sum(w in atom for atom in pieces) == 1


@given(nfas())
def test_words_in_one_atom_share_quotient_signature(a: NFA) -> None:
    prefixes = oracles.words(AB, 2)
    for atom in atoms(a):
        members = [w for w in oracles.words(AB, 4) if w in atom]
        signatures = {tuple(oracles.nfa_accepts(a, p + w) for p in prefixes) for w in members}
        assert len(signatures) <= 1


# --------------------------------------------------------------------------- metamorphic


@given(nfas())
def test_relabeling_states_and_symbols_maps_language(a: NFA) -> None:
    to_letter = {"0": "x", "1": "y"}
    renamed = relabel(a, lambda s: ("q", s), to_letter.__getitem__)
    expected = frozenset(tuple(to_letter[c] for c in w) for w in truth(a))
    assert lang(renamed, alphabet=("x", "y")) == expected
    assert len(list(minimize(renamed).states())) == len(list(minimize(a).states()))


@given(nfas(), st.lists(st.tuples(st.integers(0, 2), st.sampled_from(AB)), max_size=4), st.booleans())
def test_adding_unreachable_states_preserves_language(a: NFA, edges: list[tuple[int, str]], accepting: bool) -> None:
    padded = relabel(a, lambda s: s, lambda c: c)
    states = list(a.states())
    padded.graph.add_state("junk")
    if accepting:
        padded.accepting_states = padded.accepting_states | {"junk"}
    for index, symbol in edges:
        padded.graph.add_transition("junk", states[index % len(states)], **{ATTR_SYMBOL: symbol})
    assert lang(padded) == truth(a)
    assert len(list(minimize(padded).states())) == len(list(minimize(a).states()))


@given(nfas())
def test_double_reversal_minimizes_to_same_dfa(a: NFA) -> None:
    left = minimize(determinize(a.reverse().reverse()))
    right = minimize(a)
    assert len(list(left.states())) == len(list(right.states()))
    assert equivalent(left, right)


@given(nfas())
def test_concat_with_epsilon_is_identity(a: NFA) -> None:
    expected = truth(a)
    assert lang(a.concat(epsilon_nfa())) == expected
    assert lang(epsilon_nfa().concat(a)) == expected


@given(nfas() | dfas())
def test_double_complement_is_identity(a: NFA | DFA) -> None:
    assert lang(a.complement(SIGMA).complement(SIGMA)) == truth(a)


def test_double_complement_regression_accepting_trap() -> None:
    # The first complement's trap state is accepting; re-completing must not route missing edges into it.
    a = NFA(input_alphabet=SIGMA, initial_states=frozenset({1}), accepting_states=frozenset({0}))
    a.graph.add_state(0)
    a.graph.add_state(1)
    a.add_transition(0, 0, "0")
    a.add_transition(0, 0, "1")
    a.add_transition(1, 0, "0")
    once = a.complement(SIGMA)
    assert lang(once) == frozenset(oracles.words(AB, N)) - truth(a)
    assert lang(once.complement(SIGMA)) == truth(a)
    assert lang(complete(once, SIGMA)) == lang(once)
