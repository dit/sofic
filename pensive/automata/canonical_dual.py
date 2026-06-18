"""Dual constructions between canonical RFSA and maximized prime átomaton."""

from __future__ import annotations

from pensive.automata.atomaton import MaximizedPrimeAtomaton
from pensive.automata.rfsa import CanonicalRFSA


def dual_atomaton_from_rfsa(rfsa: CanonicalRFSA) -> MaximizedPrimeAtomaton:
    from pensive.automata.canonical_extraction import maximized_prime_atomaton_from_language

    return maximized_prime_atomaton_from_language(rfsa)


def dual_rfsa_from_atomaton(atomaton: MaximizedPrimeAtomaton) -> CanonicalRFSA:
    from pensive.automata.canonical_extraction import canonical_rfsa_from_language

    return canonical_rfsa_from_language(atomaton)
