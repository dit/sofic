"""Shared helpers for quotient and atom computations."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from sofic.automata.languages.automaton_ops import minimal_dfa_from_language
from sofic.automata.languages.base import AutomatonLanguage, ExplicitLanguage


def _residual_from_state(aut: AutomatonLanguage, state) -> AutomatonLanguage:
    sub = aut.automaton.copy()
    sub.initial_states = frozenset({state})
    return AutomatonLanguage(minimal_dfa_from_language(sub))


def _suffixes_if_prefix(word: tuple[Any, ...], prefix: Sequence[Any]) -> set[tuple[Any, ...]]:
    p = tuple(prefix)
    if word[: len(p)] == p:
        return {word[len(p) :]}
    return set()


def _prefixes_if_suffix(word: tuple[Any, ...], suffix: Sequence[Any]) -> set[tuple[Any, ...]]:
    s = tuple(suffix)
    if len(word) >= len(s) and word[len(word) - len(s) :] == s:
        return {word[: len(word) - len(s)]}
    return set()


def _is_union_of_others(target: ExplicitLanguage, others: list[ExplicitLanguage]) -> bool:
    """Whether ``target``'s positive sample is the union of those strictly inside it."""
    union_pos: set[tuple] = set()
    for lang in others:
        if lang._positive < target._positive:
            union_pos |= lang._positive
    return target._positive == union_pos
