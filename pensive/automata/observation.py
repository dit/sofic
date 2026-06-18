"""Angluin-style observation tables for canonical extraction."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pensive.automata.atomaton import Atomaton, MaximizedPrimeAtomaton
    from pensive.automata.dfa import DFA
    from pensive.automata.rfsa import CanonicalRFSA


@dataclass
class ObservationTable:
    """Membership table with prefix-closed access words and suffix experiments."""

    access_words: frozenset[tuple[Any, ...]] = field(default_factory=lambda: frozenset({()}))
    experiments: frozenset[tuple[Any, ...]] = field(default_factory=lambda: frozenset({()}))
    membership: dict[tuple[Any, ...], bool] = field(default_factory=dict)

    def to_minimal_dfa(self) -> DFA:
        from pensive.automata.canonical_extraction import observation_to_minimal_dfa

        return observation_to_minimal_dfa(self)

    def to_canonical_rfsa(self) -> CanonicalRFSA:
        from pensive.automata.canonical_extraction import observation_to_canonical_rfsa

        return observation_to_canonical_rfsa(self)

    def to_atomaton(self) -> Atomaton:
        from pensive.automata.canonical_extraction import observation_to_atomaton

        return observation_to_atomaton(self)

    def to_maximized_prime_atomaton(self) -> MaximizedPrimeAtomaton:
        from pensive.automata.canonical_extraction import observation_to_maximized_prime_atomaton

        return observation_to_maximized_prime_atomaton(self)
