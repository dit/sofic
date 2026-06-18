"""Boolean and structural regular-language operations."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from pensive.automata.languages.automaton_ops import (
    complement_dfa,
    concat_nfa,
    intersection_dfa,
    kleene_star_nfa,
    union_nfa,
)
from pensive.automata.languages.base import AutomatonLanguage, ExplicitLanguage, RegularLanguage, as_language


def union(left: RegularLanguage, right: RegularLanguage) -> RegularLanguage:
    left_lang = as_language(left)  # type: ignore[arg-type]
    right_lang = as_language(right)  # type: ignore[arg-type]
    if isinstance(left_lang, AutomatonLanguage) and isinstance(right_lang, AutomatonLanguage):
        return AutomatonLanguage(union_nfa(left_lang.automaton, right_lang.automaton))
    raise TypeError("union requires automaton-backed languages")


def intersection(left: RegularLanguage, right: RegularLanguage) -> RegularLanguage:
    left_lang = as_language(left)  # type: ignore[arg-type]
    right_lang = as_language(right)  # type: ignore[arg-type]
    if isinstance(left_lang, AutomatonLanguage) and isinstance(right_lang, AutomatonLanguage):
        return AutomatonLanguage(intersection_dfa(left_lang.automaton, right_lang.automaton))
    raise TypeError("intersection requires automaton-backed languages")


def complement(language: RegularLanguage, alphabet: frozenset[Any]) -> RegularLanguage:
    lang = as_language(language)  # type: ignore[arg-type]
    if isinstance(lang, AutomatonLanguage):
        return AutomatonLanguage(complement_dfa(lang.automaton, alphabet))
    raise TypeError("complement requires automaton-backed languages")


def reverse(language: RegularLanguage) -> RegularLanguage:
    lang = as_language(language)  # type: ignore[arg-type]
    if isinstance(lang, ExplicitLanguage):
        positive = {tuple(reversed(word)) for word in lang._positive}
        negative = {tuple(reversed(word)) for word in lang._negative}
        return ExplicitLanguage(positive=positive, negative=negative, alphabet=lang.alphabet)
    if isinstance(lang, AutomatonLanguage):
        return AutomatonLanguage.from_automaton(lang.automaton.reverse())
    raise TypeError(f"unsupported language type {type(lang)!r}")


def concat(left: RegularLanguage, right: RegularLanguage) -> RegularLanguage:
    left_lang = as_language(left)  # type: ignore[arg-type]
    right_lang = as_language(right)  # type: ignore[arg-type]
    if isinstance(left_lang, AutomatonLanguage) and isinstance(right_lang, AutomatonLanguage):
        return AutomatonLanguage(concat_nfa(left_lang.automaton, right_lang.automaton))
    raise TypeError("concat requires automaton-backed languages")


def kleene_star(language: RegularLanguage) -> RegularLanguage:
    lang = as_language(language)  # type: ignore[arg-type]
    if isinstance(lang, AutomatonLanguage):
        return AutomatonLanguage(kleene_star_nfa(lang.automaton))
    raise TypeError("kleene_star requires automaton-backed languages")


def product(left: RegularLanguage, right: RegularLanguage) -> RegularLanguage:
    return intersection(left, right)
