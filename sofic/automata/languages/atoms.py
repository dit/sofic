"""Atoms and prime atoms of regular languages."""

from __future__ import annotations

from sofic.automata.languages._quotient_utils import _alphabet_of, _languages_equal, _residual_from_state
from sofic.automata.languages.automaton_ops import minimal_dfa_from_language
from sofic.automata.languages.base import AutomatonLanguage, RegularLanguage, as_language
from sofic.automata.languages.quotients import left_quotients


def atoms(language: RegularLanguage) -> frozenset[RegularLanguage]:
    lang = as_language(language)  # type: ignore[arg-type]
    result: set[RegularLanguage] = set(left_quotients(language))
    if isinstance(lang, AutomatonLanguage) and not result:
        dfa = minimal_dfa_from_language(lang.automaton)
        for state in dfa.states():
            result.add(_residual_from_state(AutomatonLanguage(dfa), state))
    return frozenset(result)


def prime_atoms(language: RegularLanguage) -> frozenset[RegularLanguage]:
    all_atoms = atoms(language)
    return frozenset(atom for atom in all_atoms if is_prime_atom(atom, all_atoms))


def is_prime_atom(atom: RegularLanguage, all_atoms: frozenset[RegularLanguage]) -> bool:
    others = [candidate for candidate in all_atoms if candidate is not atom]
    if not others:
        return True
    alphabet = _alphabet_of(atom)
    return all(not _languages_equal(atom, other, alphabet) for other in others)
