"""Duality between canonical RFSAs and maximized prime átomata.

The maximized prime átomaton of ``L`` is the reverse of the canonical RFSA of
the reversed language :cite:`MaarandTamm2022`. Reversing either object therefore gives
the other one for the reversed language, and reversing twice is the identity.
"""

from __future__ import annotations

from sofic.automata.atomaton import MaximizedPrimeAtomaton
from sofic.automata.rfsa import CanonicalRFSA


def dual_atomaton_from_rfsa(rfsa: CanonicalRFSA) -> MaximizedPrimeAtomaton:
    """Reverse the canonical RFSA of ``L`` into the maximized prime átomaton of ``L^R``."""
    from sofic.automata.canonical_extraction import _reverse_into

    return _reverse_into(MaximizedPrimeAtomaton, rfsa)


def dual_rfsa_from_atomaton(atomaton: MaximizedPrimeAtomaton) -> CanonicalRFSA:
    """Reverse the maximized prime átomaton of ``L`` into the canonical RFSA of ``L^R``."""
    from sofic.automata.canonical_extraction import _reverse_into

    return _reverse_into(CanonicalRFSA, atomaton)
