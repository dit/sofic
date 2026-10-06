"""Prime and composed residual languages :cite:`Denis2002`.

A residual (left quotient) of ``L`` is *composed* when it is the union of the
residuals strictly contained in it, and *prime* otherwise (the empty residual
is the empty union, so it is composed). For automaton-backed languages these
questions are decided exactly on the minimal DFA; for an
:class:`~sofic.automata.languages.base.ExplicitLanguage` (a finite labeled
sample) they are answered on the sample only.
"""

from __future__ import annotations

from sofic.automata.languages._quotient_utils import _is_union_of_others
from sofic.automata.languages.base import AutomatonLanguage, ExplicitLanguage, RegularLanguage, as_language
from sofic.automata.languages.quotients import left_quotients


def prime_residuals(language: RegularLanguage) -> frozenset[RegularLanguage]:
    """Return the prime residuals of ``language``."""
    lang = as_language(language)  # type: ignore[arg-type]
    if isinstance(lang, AutomatonLanguage):
        from sofic.automata.canonical_extraction import ResidualTable

        table = ResidualTable.from_automaton(lang.automaton)
        return frozenset(AutomatonLanguage(table.residual_automaton(q)) for q in table.prime_states())
    all_residuals = left_quotients(lang)
    return frozenset(r for r in all_residuals if not is_composed_residual(r, all_residuals))


def is_composed_residual(residual: RegularLanguage, all_residuals: frozenset[RegularLanguage]) -> bool:
    """Return whether ``residual`` is the union of the members of ``all_residuals`` strictly inside it."""
    if isinstance(residual, ExplicitLanguage):
        others = [r for r in all_residuals if r is not residual and isinstance(r, ExplicitLanguage)]
        return _is_union_of_others(residual, others)

    from sofic.automata.languages.automaton_ops import union_nfa

    target = as_language(residual).automaton  # type: ignore[union-attr]
    inside = []
    for other in all_residuals:
        if other is residual:
            continue
        candidate = as_language(other).automaton  # type: ignore[union-attr]
        if _is_subset(candidate, target) and not _is_subset(target, candidate):
            inside.append(candidate)
    if not inside:
        return _is_empty(target)
    union = inside[0]
    for candidate in inside[1:]:
        union = union_nfa(union, candidate)
    return _is_subset(target, union)


def _is_subset(smaller, larger) -> bool:
    from sofic.automata.algorithms import _transition_alphabet
    from sofic.automata.languages.automaton_ops import difference_dfa

    alphabet = _transition_alphabet(smaller) | _transition_alphabet(larger)
    return _is_empty(difference_dfa(smaller, larger, alphabet=alphabet))


def _is_empty(aut) -> bool:
    from sofic.automata.algorithms import trim

    return not trim(aut).accepting_states
