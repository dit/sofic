"""Left and right quotients of regular languages."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from pensive.automata.languages._quotient_utils import (
    _prefixes_if_suffix,
    _residual_from_state,
    _suffixes_if_prefix,
    _words_up_to,
)
from pensive.automata.languages.automaton_ops import (
    left_quotient_automaton,
    minimal_dfa_from_language,
    right_quotient_automaton,
)
from pensive.automata.languages.base import AutomatonLanguage, ExplicitLanguage, RegularLanguage, as_language


def left_quotient(u: Sequence[Any], language: RegularLanguage) -> RegularLanguage:
    """Return u^{-1} L = {w | uw in L}."""
    lang = as_language(language)  # type: ignore[arg-type]
    if isinstance(lang, ExplicitLanguage):
        positive = {w for uw in lang._positive for w in _suffixes_if_prefix(uw, u)}
        negative = {w for uw in lang._negative for w in _suffixes_if_prefix(uw, u)}
        return ExplicitLanguage(positive=positive, negative=negative, alphabet=lang.alphabet)
    if isinstance(lang, AutomatonLanguage):
        return AutomatonLanguage(left_quotient_automaton(u, lang.automaton))
    raise TypeError(f"unsupported language type {type(lang)!r}")


def right_quotient(language: RegularLanguage, u: Sequence[Any]) -> RegularLanguage:
    """Return L u^{-1} = {w | wu in L}."""
    lang = as_language(language)  # type: ignore[arg-type]
    if isinstance(lang, ExplicitLanguage):
        positive = {w for wu in lang._positive for w in _prefixes_if_suffix(wu, u)}
        negative = {w for wu in lang._negative for w in _prefixes_if_suffix(wu, u)}
        return ExplicitLanguage(positive=positive, negative=negative, alphabet=lang.alphabet)
    if isinstance(lang, AutomatonLanguage):
        return AutomatonLanguage(right_quotient_automaton(lang.automaton, u))
    raise TypeError(f"unsupported language type {type(lang)!r}")


def left_quotients(language: RegularLanguage) -> frozenset[RegularLanguage]:
    lang = as_language(language)  # type: ignore[arg-type]
    if isinstance(lang, AutomatonLanguage):
        dfa = minimal_dfa_from_language(lang.automaton)
        quotients: set[RegularLanguage] = set()
        for state in dfa.states():
            quotients.add(_residual_from_state(AutomatonLanguage(dfa), state))
        return frozenset(quotients)
    if isinstance(lang, ExplicitLanguage):
        alphabet = lang.alphabet
        quotients: set[RegularLanguage] = set()
        for length in range(4):
            for word in _words_up_to(length, alphabet):
                quotients.add(left_quotient(word, lang))
        return frozenset(quotients)
    raise TypeError(f"unsupported language type {type(lang)!r}")


def residuals(language: RegularLanguage) -> frozenset[RegularLanguage]:
    """Residual languages Res(L) = {u^{-1}L | u in Sigma*}."""
    return left_quotients(language)
