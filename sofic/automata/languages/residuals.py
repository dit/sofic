"""Prime and composed residual languages."""

from __future__ import annotations

from sofic.automata.languages._quotient_utils import (
    _alphabet_of,
    _is_union_of_others,
    _languages_equal,
)
from sofic.automata.languages.base import ExplicitLanguage, RegularLanguage
from sofic.automata.languages.quotients import left_quotients


def prime_residuals(language: RegularLanguage) -> frozenset[RegularLanguage]:
    all_residuals = left_quotients(language)
    return frozenset(r for r in all_residuals if not is_composed_residual(r, all_residuals))


def is_composed_residual(residual: RegularLanguage, all_residuals: frozenset[RegularLanguage]) -> bool:
    if isinstance(residual, ExplicitLanguage):
        others = [r for r in all_residuals if r is not residual and isinstance(r, ExplicitLanguage)]
        return _is_union_of_others(residual, others)
    others = [r for r in all_residuals if r is not residual]
    alphabet = _alphabet_of(residual)
    return any(_languages_equal(residual, other, alphabet) for other in others)
