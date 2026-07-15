"""Canonical automaton extraction from languages and observation tables."""

from __future__ import annotations

from typing import Any

from sofic.automata.atomaton import Atomaton, MaximizedPrimeAtomaton
from sofic.automata.dfa import DFA
from sofic.automata.languages.automaton_ops import (
    minimal_dfa_from_language,
)
from sofic.automata.languages.base import AutomatonLanguage, RegularLanguage, as_language
from sofic.automata.nfa import NFA
from sofic.automata.observation import ObservationTable
from sofic.automata.rfsa import CanonicalRFSA


def _language_automaton(language: RegularLanguage | NFA | DFA) -> NFA | DFA:
    if isinstance(language, (NFA, DFA)):
        return language
    lang = as_language(language)  # type: ignore[arg-type]
    if isinstance(lang, AutomatonLanguage):
        return lang.automaton
    raise TypeError(f"cannot extract automaton from {type(language)!r}")


def canonical_rfsa_from_language(language: RegularLanguage | NFA | DFA) -> CanonicalRFSA:
    """Build canonical RFSA from prime residuals of the language.

    Phase 2 placeholder: currently reuses the minimal DFA structure directly.
    """
    aut = _language_automaton(language)
    dfa = minimal_dfa_from_language(aut)
    return CanonicalRFSA(
        input_alphabet=dfa.input_alphabet,
        initial_states=dfa.initial_states,
        accepting_states=dfa.accepting_states,
        graph=dfa.graph.copy(),
    )


def atomaton_from_language(language: RegularLanguage | NFA | DFA) -> Atomaton:
    """Build átomaton via double-reversal pipeline."""
    aut = _language_automaton(language)
    dfa = minimal_dfa_from_language(aut)
    rev = dfa.reverse().determinize().minimize()
    atom = rev.reverse().determinize()
    return Atomaton(
        input_alphabet=atom.input_alphabet,
        initial_states=atom.initial_states,
        accepting_states=atom.accepting_states,
        graph=atom.graph.copy(),
    )


def maximized_prime_atomaton_from_language(language: RegularLanguage | NFA | DFA) -> MaximizedPrimeAtomaton:
    """Build maximized prime átomaton from a language.

    Phase 2 placeholder: currently reuses the minimal DFA structure directly.
    """
    aut = _language_automaton(language)
    lang = AutomatonLanguage(minimal_dfa_from_language(aut))
    return MaximizedPrimeAtomaton(
        input_alphabet=lang.automaton.input_alphabet,
        initial_states=lang.automaton.initial_states,
        accepting_states=lang.automaton.accepting_states,
        graph=lang.automaton.graph.copy(),
    )


def observation_to_minimal_dfa(table: ObservationTable) -> DFA:
    """Angluin-style DFA extraction from a closed, consistent table."""
    access = sorted(table.access_words, key=lambda w: (len(w), w))
    experiments = sorted(table.experiments, key=lambda w: (len(w), w))

    def signature(word: tuple[Any, ...]) -> tuple[bool, ...]:
        return tuple(table.membership.get(word + exp, False) for exp in experiments)

    classes: dict[tuple[bool, ...], tuple[Any, ...]] = {}
    for word in access:
        sig = signature(word)
        classes.setdefault(sig, word)

    state_for_word = {word: classes[signature(word)] for word in access}
    alphabet = _infer_alphabet(table)
    dfa = DFA(
        input_alphabet=alphabet,
        initial_states=frozenset({()}),
        accepting_states=frozenset(),
    )
    for word in access:
        dfa.graph.add_state(state_for_word[word])
    for word in access:
        for symbol in alphabet:
            target_word = word + (symbol,)
            if target_word in state_for_word:
                dfa.add_transition(state_for_word[word], state_for_word[target_word], symbol)
    accepting = {state_for_word[word] for word in access if table.membership.get(word, False)}
    dfa.accepting_states = frozenset(accepting)
    return dfa


def observation_to_canonical_rfsa(table: ObservationTable) -> CanonicalRFSA:
    dfa = observation_to_minimal_dfa(table)
    return canonical_rfsa_from_language(dfa)


def observation_to_atomaton(table: ObservationTable) -> Atomaton:
    dfa = observation_to_minimal_dfa(table)
    return atomaton_from_language(dfa)


def observation_to_maximized_prime_atomaton(table: ObservationTable) -> MaximizedPrimeAtomaton:
    dfa = observation_to_minimal_dfa(table)
    return maximized_prime_atomaton_from_language(dfa)


def _infer_alphabet(table: ObservationTable) -> frozenset[Any]:
    symbols: set[Any] = set()
    for word in table.access_words | table.experiments:
        symbols.update(word)
    return frozenset(symbols)
