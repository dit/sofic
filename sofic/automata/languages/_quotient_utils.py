"""Shared helpers for quotient and atom computations."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from sofic.automata.languages.automaton_ops import minimal_dfa_from_language
from sofic.automata.languages.base import AutomatonLanguage, ExplicitLanguage, RegularLanguage, as_language


def _words_up_to(length: int, alphabet: frozenset[Any]) -> list[tuple[Any, ...]]:
    if length < 0:
        return []
    if length == 0:
        return [()]
    shorter = _words_up_to(length - 1, alphabet)
    return [word + (symbol,) for word in shorter for symbol in alphabet]


def _languages_equal(left: RegularLanguage, right: RegularLanguage, alphabet: frozenset[Any], max_len: int = 6) -> bool:
    for length in range(max_len + 1):
        for word in _words_up_to(length, alphabet):
            if (word in left) != (word in right):
                return False
    return True


def _residual_from_state(aut: AutomatonLanguage, state) -> AutomatonLanguage:
    sub = aut.automaton.copy()
    sub.initial_states = frozenset({state})
    return AutomatonLanguage(minimal_dfa_from_language(sub))


def _alphabet_of(language: RegularLanguage) -> frozenset[Any]:
    lang = as_language(language)  # type: ignore[arg-type]
    if isinstance(lang, ExplicitLanguage):
        return lang.alphabet
    if isinstance(lang, AutomatonLanguage):
        return lang.automaton.input_alphabet
    return frozenset()


def _suffixes_if_prefix(word: tuple[Any, ...], prefix: Sequence[Any]) -> set[tuple[Any, ...]]:
    p = tuple(prefix)
    if word[: len(p)] == p:
        return {word[len(p) :]}
    return set()


def _prefixes_if_suffix(word: tuple[Any, ...], suffix: Sequence[Any]) -> set[tuple[Any, ...]]:
    s = tuple(suffix)
    if len(word) >= len(s) and word[-len(s) :] == s:
        return {word[: len(word) - len(s)]}
    return set()


def _is_union_of_others(target: ExplicitLanguage, others: list[ExplicitLanguage]) -> bool:
    union_pos: set[tuple] = set()
    union_neg: set[tuple] = set()
    for lang in others:
        union_pos |= lang._positive
        union_neg |= lang._negative
    return target._positive == union_pos and target._negative == union_neg
