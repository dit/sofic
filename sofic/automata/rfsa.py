"""Residual and canonical RFSA automata."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from sofic.automata.languages.base import RegularLanguage
from sofic.automata.nfa import NFA

if TYPE_CHECKING:
    from sofic.automata.atomaton import MaximizedPrimeAtomaton
    from sofic.automata.observation import ObservationTable


class ResidualFiniteStateAutomaton(NFA):
    """NFA whose states accept residual languages of the recognized language :cite:`Denis2002`."""

    def validate(self) -> None:
        super().validate()
        from sofic.automata.algorithms import equivalent
        from sofic.automata.canonical_extraction import ResidualTable

        if not self.initial_states:
            return
        table = ResidualTable.from_automaton(self)
        residuals = [table.residual_automaton(q) for q in table.states]
        for state in self.states():
            right = NFA(
                input_alphabet=self.input_alphabet,
                initial_states=frozenset({state}),
                accepting_states=self.accepting_states,
                graph=self.graph,
            )
            if not any(equivalent(right, residual, frozenset(table.alphabet)) for residual in residuals):
                self._require(False, f"right language of state {state!r} is not a residual of the language")


class CanonicalRFSA(ResidualFiniteStateAutomaton):
    """Canonical residual finite-state automaton R(L) :cite:`Denis2002`."""

    @classmethod
    def from_language(cls, language: RegularLanguage | NFA, **kwargs: Any) -> CanonicalRFSA:
        from sofic.automata.canonical_extraction import canonical_rfsa_from_language

        return canonical_rfsa_from_language(language)

    @classmethod
    def from_observation_table(cls, table: ObservationTable, **kwargs: Any) -> CanonicalRFSA:
        from sofic.automata.canonical_extraction import observation_to_canonical_rfsa

        return observation_to_canonical_rfsa(table)

    def dual(self) -> MaximizedPrimeAtomaton:
        """Return the reverse automaton: the maximized prime átomaton of the reversed language."""
        from sofic.automata.canonical_dual import dual_atomaton_from_rfsa

        return dual_atomaton_from_rfsa(self)
