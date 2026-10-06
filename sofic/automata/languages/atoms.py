r"""Atoms and prime atoms of regular languages :cite:`BrzozowskiTamm2014`.

An *atom* of ``L`` is a non-empty intersection of complemented or
uncomplemented left quotients of ``L``. The atoms correspond one-to-one with
the states ``q`` of the minimal DFA of the reversed language :math:`L^R`: the
atom :math:`A_q` is the reverse of the set of words leading to ``q``, and these
are exactly the right languages of the átomaton's states. An atom is *prime*
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
    """Return the atoms of ``language``."""
    table = _reversed_table(language)
    return frozenset(_atom(table, q) for q in table.states if not table.is_empty(q))


def prime_atoms(language: RegularLanguage) -> frozenset[RegularLanguage]:
    """Return the prime atoms of ``language``."""
    table = _reversed_table(language)
    return frozenset(_atom(table, q) for q in table.prime_states())
