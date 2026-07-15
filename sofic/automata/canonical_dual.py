"""Dual constructions between canonical RFSA and maximized prime átomaton."""

from __future__ import annotations

from sofic.automata.atomaton import MaximizedPrimeAtomaton
from sofic.automata.rfsa import CanonicalRFSA


def dual_atomaton_from_rfsa(rfsa: CanonicalRFSA) -> MaximizedPrimeAtomaton:
    from sofic.automata.canonical_extraction import maximized_prime_atomaton_from_language

    return maximized_prime_atomaton_from_language(rfsa)


def dual_rfsa_from_atomaton(atomaton: MaximizedPrimeAtomaton) -> CanonicalRFSA:
    from sofic.automata.canonical_extraction import canonical_rfsa_from_language

    return canonical_rfsa_from_language(atomaton)
