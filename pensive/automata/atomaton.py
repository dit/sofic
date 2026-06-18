"""Atomic and átomaton NFA presentations."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pensive.automata.dfa import DFA
from pensive.automata.languages.base import RegularLanguage
from pensive.automata.nfa import NFA

if TYPE_CHECKING:
    from pensive.automata.observation import ObservationTable
    from pensive.automata.rfsa import CanonicalRFSA


class AtomicAutomaton(NFA):
    """NFA whose states accept unions of atoms."""

    def validate(self) -> None:
        super().validate()
        # Phase 2: verify right languages are unions of atoms


class Atomaton(AtomicAutomaton):
    """Canonical átomaton whose states are all atoms of L."""

    @classmethod
    def from_language(cls, language: RegularLanguage | NFA, **kwargs: Any) -> Atomaton:
        from pensive.automata.canonical_extraction import atomaton_from_language

        return atomaton_from_language(language)

    def to_minimal_dfa_via_double_reversal(self) -> DFA:
        from pensive.automata.algorithms import minimize

        return minimize(self, algorithm="brzozowski")


class MaximizedPrimeAtomaton(AtomicAutomaton):
    """Maximized prime átomaton — dual of the canonical RFSA."""

    @classmethod
    def from_language(cls, language: RegularLanguage | NFA, **kwargs: Any) -> MaximizedPrimeAtomaton:
        from pensive.automata.canonical_extraction import maximized_prime_atomaton_from_language

        return maximized_prime_atomaton_from_language(language)

    @classmethod
    def from_observation_table(cls, table: ObservationTable, **kwargs: Any) -> MaximizedPrimeAtomaton:
        from pensive.automata.canonical_extraction import observation_to_maximized_prime_atomaton

        return observation_to_maximized_prime_atomaton(table)

    @classmethod
    def from_canonical_rfsa(cls, rfsa: CanonicalRFSA, **kwargs: Any) -> MaximizedPrimeAtomaton:
        from pensive.automata.canonical_dual import dual_atomaton_from_rfsa

        return dual_atomaton_from_rfsa(rfsa)
