"""Atomic and átomaton NFA presentations."""

from __future__ import annotations

from collections.abc import Hashable
from typing import TYPE_CHECKING, Any, cast

from sofic.automata.dfa import DFA
from sofic.automata.languages.base import RegularLanguage
from sofic.automata.nfa import NFA
from sofic.exceptions import SoficValidationError

if TYPE_CHECKING:
    from sofic.automata.observation import ObservationTable
    from sofic.automata.rfsa import CanonicalRFSA


def atomic_states(nfa: NFA, *, alphabet: frozenset[Any] | None = None) -> frozenset[Hashable]:
    r"""Return the states of ``nfa`` whose right language is a union of atoms.

    An *atom* of a regular language is a non-empty intersection of complemented
    or uncomplemented left quotients; the atoms partition :math:`\Sigma^*` and
    every quotient is a union of them.  A state is *atomic* when its right
    language is such a union.

    Implements Theorem 4 of :cite:`BrzozowskiTamm2014`: a state :math:`q` is
    atomic if and only if :math:`\{s \in N^{RD} : q \in s\}` is a union of
    Nerode classes of :math:`N^{RD}`, the subset construction applied to the
    reversal of ``nfa``.
    """
    from sofic.automata.algorithms import _effective_alphabet, nerode_partition

    symbols = alphabet if alphabet is not None else _effective_alphabet(nfa)
    reverse_subsets = nfa.reverse().determinize(alphabet=symbols)
    blocks = nerode_partition(reverse_subsets, alphabet=symbols)
    # Subset-construction states are frozensets of the original NFA's states.
    subsets = cast("list[frozenset[Hashable]]", list(reverse_subsets.states()))

    atomic = set()
    for state in nfa.states():
        containing = {subset for subset in subsets if state in subset}
        if all(block <= containing or block.isdisjoint(containing) for block in blocks):
            atomic.add(state)
    return frozenset(atomic)


def is_atomic(nfa: NFA, *, alphabet: frozenset[Any] | None = None) -> bool:
    """Return whether every state of ``nfa`` has a right language of atoms.

    Equivalently -- Corollary 2 of :cite:`BrzozowskiTamm2014` -- whether
    :math:`N^{RD}` is minimal, which is exactly the condition under which the
    subset construction applied to :math:`N^R` yields a minimal DFA.
    """
    return atomic_states(nfa, alphabet=alphabet) == frozenset(nfa.states())


class AtomicAutomaton(NFA):
    """NFA whose states accept unions of atoms."""

    def validate(self) -> None:
        super().validate()
        non_atomic = frozenset(self.states()) - atomic_states(self)
        if non_atomic:
            listed = ", ".join(sorted(map(repr, non_atomic)))
            raise SoficValidationError(f"right language is not a union of atoms for state(s) {listed}")


class Atomaton(AtomicAutomaton):
    """Canonical átomaton whose states are all atoms of L."""

    @classmethod
    def from_language(cls, language: RegularLanguage | NFA, **kwargs: Any) -> Atomaton:
        from sofic.automata.canonical_extraction import atomaton_from_language

        return atomaton_from_language(language)

    def to_minimal_dfa_via_double_reversal(self) -> DFA:
        from sofic.automata.algorithms import minimize

        return minimize(self, algorithm="brzozowski")


class MaximizedPrimeAtomaton(AtomicAutomaton):
    """Maximized prime átomaton — dual of the canonical RFSA."""

    @classmethod
    def from_language(cls, language: RegularLanguage | NFA, **kwargs: Any) -> MaximizedPrimeAtomaton:
        from sofic.automata.canonical_extraction import maximized_prime_atomaton_from_language

        return maximized_prime_atomaton_from_language(language)

    @classmethod
    def from_observation_table(cls, table: ObservationTable, **kwargs: Any) -> MaximizedPrimeAtomaton:
        from sofic.automata.canonical_extraction import observation_to_maximized_prime_atomaton

        return observation_to_maximized_prime_atomaton(table)

    @classmethod
    def from_canonical_rfsa(cls, rfsa: CanonicalRFSA, **kwargs: Any) -> MaximizedPrimeAtomaton:
        from sofic.automata.canonical_dual import dual_atomaton_from_rfsa

        return dual_atomaton_from_rfsa(rfsa)
