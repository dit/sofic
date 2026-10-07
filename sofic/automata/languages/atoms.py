r"""Atoms and prime atoms of regular languages :cite:`BrzozowskiTamm2014`.

An *atom* of ``L`` is a non-empty intersection of complemented or
uncomplemented left quotients of ``L``; the atoms partition :math:`\Sigma^*`
(:cite:`BrzozowskiTamm2014`, Section 4.2). They correspond one-to-one with the
reachable states ``q`` of the minimal *complete* DFA of the reversed language
:math:`L^R`: the atom :math:`A_q` is the reverse of the set of words leading to
``q``. The *negative* atom, in which every quotient is complemented, belongs to
the empty quotient of :math:`L^R` when that is reachable. The átomaton is built
from the trimmed minimal DFA of :math:`L^R`, so its states are the atoms other
than the negative atom (whose state has an empty right language). An atom is *prime*
when its matching quotient of :math:`L^R` is a prime residual; prime atoms label
the states of the maximized prime átomaton.
"""

from __future__ import annotations

from sofic.automata.languages.base import AutomatonLanguage, RegularLanguage, as_language


def _reversed_table(language: RegularLanguage):
    from sofic.automata.canonical.residual import ResidualTable

    lang = as_language(language)  # type: ignore[arg-type]
    if not isinstance(lang, AutomatonLanguage):
        raise TypeError("atoms are defined for automaton-backed regular languages")
    return ResidualTable.from_automaton(lang.automaton.reverse())


def _atom(table, state) -> AutomatonLanguage:
    from sofic.automata.languages.automaton_ops import minimal_dfa_from_language

    return AutomatonLanguage(minimal_dfa_from_language(table.left_language_automaton(state).reverse()))


def atoms(language: RegularLanguage) -> frozenset[RegularLanguage]:
    """Return the atoms of ``language``, including the negative atom when it is non-empty."""
    table = _reversed_table(language)
    return frozenset(_atom(table, q) for q in table._bfs_order())


def prime_atoms(language: RegularLanguage) -> frozenset[RegularLanguage]:
    """Return the prime atoms of ``language``."""
    table = _reversed_table(language)
    return frozenset(_atom(table, q) for q in table.prime_states())
