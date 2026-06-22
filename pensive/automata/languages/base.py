"""Regular language protocol and concrete wrappers."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol, runtime_checkable

from pensive.automata.base import LabeledAutomaton


@runtime_checkable
class RegularLanguage(Protocol):
    """Membership oracle for a regular (or finitely specified) language."""

    def __contains__(self, word: Sequence[Any]) -> bool: ...


class ExplicitLanguage:
    """Language given by explicit positive/negative word sets."""

    __slots__ = ("_positive", "_negative", "_alphabet")

    def __init__(
        self,
        positive: set[tuple[Any, ...]] | None = None,
        negative: set[tuple[Any, ...]] | None = None,
        alphabet: frozenset[Any] | None = None,
    ) -> None:
        self._positive = positive if positive is not None else set()
        self._negative = negative if negative is not None else set()
        self._alphabet = alphabet if alphabet is not None else frozenset()

    def __contains__(self, word: Sequence[Any]) -> bool:
        key = tuple(word)
        if key in self._negative:
            return False
        return key in self._positive

    @property
    def alphabet(self) -> frozenset[Any]:
        return self._alphabet


class AutomatonLanguage:
    """Language recognized by a :class:`LabeledAutomaton`."""

    __slots__ = ("_automaton",)

    def __init__(self, automaton: LabeledAutomaton) -> None:
        self._automaton = automaton

    def __contains__(self, word: Sequence[Any]) -> bool:
        return self._automaton.recognizes(word)

    @property
    def automaton(self) -> LabeledAutomaton:
        return self._automaton

    @classmethod
    def from_automaton(cls, aut: LabeledAutomaton) -> AutomatonLanguage:
        return cls(aut)


def as_language(language: RegularLanguage | LabeledAutomaton) -> RegularLanguage:
    if isinstance(language, LabeledAutomaton):
        return AutomatonLanguage.from_automaton(language)
    return language
