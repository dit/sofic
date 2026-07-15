"""Residual and canonical RFSA automata."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from sofic.automata.languages.base import RegularLanguage
from sofic.automata.nfa import NFA

if TYPE_CHECKING:
    from sofic.automata.observation import ObservationTable


class ResidualFiniteStateAutomaton(NFA):
    """NFA whose states accept residual languages of the recognized language."""

    def validate(self) -> None:
        super().validate()
        # Phase 2: verify each state's right language is in Res(L(R))


class CanonicalRFSA(ResidualFiniteStateAutomaton):
    """Canonical residual finite-state automaton R(L)."""

    @classmethod
    def from_language(cls, language: RegularLanguage | NFA, **kwargs: Any) -> CanonicalRFSA:
        from sofic.automata.canonical_extraction import canonical_rfsa_from_language

        return canonical_rfsa_from_language(language)

    @classmethod
    def from_observation_table(cls, table: ObservationTable, **kwargs: Any) -> CanonicalRFSA:
        from sofic.automata.canonical_extraction import observation_to_canonical_rfsa

        return observation_to_canonical_rfsa(table)
