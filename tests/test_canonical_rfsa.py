"""Exact canonical RFSA, maximized prime átomaton, residuals, and atoms."""

import pytest
from hypothesis import given, settings

from sofic.automata.algorithms import equivalent
from sofic.automata.canonical.atomaton import Atomaton, MaximizedPrimeAtomaton, is_atomic
from sofic.automata.canonical.rfsa import CanonicalRFSA, ResidualFiniteStateAutomaton
from sofic.automata.dfa import DFA
from sofic.automata.languages.atoms import atoms, prime_atoms
from sofic.automata.languages.base import AutomatonLanguage
from sofic.automata.languages.residuals import prime_residuals
from sofic.automata.nfa import NFA
from sofic.exceptions import SoficValidationError
from sofic.testing.strategies import dfas


def _a_then_n_symbols(n: int) -> NFA:
    """NFA for Sigma* a Sigma^n: minimal DFA 2^(n+1) states, canonical RFSA n+2 (Denis et al. 2002)."""
    nfa = NFA(input_alphabet=frozenset("ab"), initial_states=frozenset({0}), accepting_states=frozenset({n + 1}))
    for state in range(n + 2):
        nfa.graph.add_state(state)
    nfa.add_transition(0, 0, "a")
    nfa.add_transition(0, 0, "b")
    nfa.add_transition(0, 1, "a")
    for state in range(1, n + 1):
        nfa.add_transition(state, state + 1, "a")
        nfa.add_transition(state, state + 1, "b")
    return nfa


@pytest.mark.parametrize("n", [1, 2, 3])
def test_canonical_rfsa_is_exponentially_smaller_than_minimal_dfa(n):
    nfa = _a_then_n_symbols(n)
    assert len(list(nfa.minimize().states())) == 2 ** (n + 1)
    rfsa = CanonicalRFSA.from_language(nfa)
    assert len(list(rfsa.states())) == n + 2
    assert equivalent(rfsa, nfa)
    rfsa.validate()


def test_prime_residual_count_matches_canonical_rfsa():
    assert len(prime_residuals(AutomatonLanguage(_a_then_n_symbols(1)))) == 3


def test_canonical_rfsa_is_invariant_under_state_renaming():
    dfa = _a_then_n_symbols(2).minimize()
    renamed = DFA(
        input_alphabet=dfa.input_alphabet,
        initial_states=frozenset(("r", s) for s in dfa.initial_states),
        accepting_states=frozenset(("r", s) for s in dfa.accepting_states),
    )
    for state in dfa.states():
        renamed.graph.add_state(("r", state))
    for t in dfa.transitions():
        renamed.add_transition(("r", t.source), ("r", t.target), t.data["symbol"])
    original = CanonicalRFSA.from_language(dfa)
    other = CanonicalRFSA.from_language(renamed)
    assert len(list(original.states())) == len(list(other.states()))
    assert len(list(original.transitions())) == len(list(other.transitions()))


def test_residual_validate_rejects_non_residual_state():
    # State "x" accepts {b}, which is not a residual of L = {ab}.
    nfa = ResidualFiniteStateAutomaton(
        input_alphabet=frozenset("ab"), initial_states=frozenset({"s"}), accepting_states=frozenset({"f"})
    )
    for state in ("s", "m", "f", "x"):
        nfa.graph.add_state(state)
    nfa.add_transition("s", "m", "a")
    nfa.add_transition("m", "f", "b")
    nfa.add_transition("x", "f", "b")
    nfa.add_transition("x", "f", "a")
    with pytest.raises(SoficValidationError, match="not a residual"):
        nfa.validate()


@settings(max_examples=60, deadline=None)
@given(dfas(min_states=1, max_states=4))
def test_canonical_rfsa_and_prime_atomaton_recognize_the_language(dfa):
    rfsa = CanonicalRFSA.from_language(dfa)
    assert equivalent(rfsa, dfa)
    rfsa.validate()
    assert len(list(rfsa.states())) <= len(list(dfa.minimize().states()))

    mpa = MaximizedPrimeAtomaton.from_language(dfa)
    assert equivalent(mpa, dfa)
    mpa.validate()


def test_maximized_prime_atomaton_need_not_be_atomic():
    """Its right languages lie between an atom and a maximized atom (Tamm 2015), not on atoms."""
    from sofic.automata.enumeration.icdfa import icdfa_string_to_dfa

    dfa = icdfa_string_to_dfa((0, 1, 0, 2, 0, 1), ("0", "1"), n=3, k=2, final_states=frozenset({0, 1}))
    mpa = MaximizedPrimeAtomaton.from_language(dfa)
    assert equivalent(mpa, dfa)
    assert not is_atomic(mpa)
    assert is_atomic(Atomaton.from_language(dfa))


@settings(max_examples=40, deadline=None)
@given(dfas(min_states=1, max_states=4))
def test_dual_round_trip(dfa):
    rfsa = CanonicalRFSA.from_language(dfa)
    mpa_of_reverse = rfsa.dual()
    assert isinstance(mpa_of_reverse, MaximizedPrimeAtomaton)
    assert equivalent(mpa_of_reverse, dfa.reverse())
    back = mpa_of_reverse.dual()
    assert isinstance(back, CanonicalRFSA)
    assert equivalent(back, dfa)
    expected = MaximizedPrimeAtomaton.from_language(dfa.reverse())
    assert len(list(mpa_of_reverse.states())) == len(list(expected.states()))


@settings(max_examples=40, deadline=None)
@given(dfas(min_states=1, max_states=4))
def test_atoms_label_atomaton_states_and_prime_atoms_label_prime_atomaton(dfa):
    if not dfa.minimize().accepting_states:
        return
    language = AutomatonLanguage(dfa)
    # The átomaton omits the negative atom: it is non-empty exactly when the
    # trimmed minimal DFA of the reverse is incomplete.
    reversed_min = dfa.reverse().determinize().minimize()
    negative_nonempty = any(
        not reversed_min.delta(state, symbol) for state in reversed_min.states() for symbol in dfa.input_alphabet
    )
    assert len(atoms(language)) == len(list(Atomaton.from_language(dfa).states())) + negative_nonempty
    assert len(prime_atoms(language)) == len(list(MaximizedPrimeAtomaton.from_language(dfa).states()))
    for atom in atoms(language):
        assert atom.automaton.minimize().accepting_states
