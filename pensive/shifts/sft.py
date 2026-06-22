"""Shifts of finite type."""

from __future__ import annotations

from itertools import product
from typing import Any

from pensive.graph import TransitionGraph
from pensive.shifts.base import SymbolicModel


class ShiftOfFiniteType(SymbolicModel):
    """SFT specified by forbidden words or an explicit presentation."""

    _forbidden_words: frozenset[tuple[Any, ...]]
    _has_forbidden_word_spec: bool

    def __init__(
        self,
        forbidden_words: set[tuple[Any, ...]] | frozenset[tuple[Any, ...]] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._has_forbidden_word_spec = forbidden_words is not None
        self._forbidden_words = frozenset(tuple(word) for word in (forbidden_words or set()))

    @classmethod
    def from_forbidden_words(
        cls,
        forbidden: set[tuple[Any, ...]],
        symbol_alphabet: frozenset[Any],
        **kwargs: Any,
    ) -> ShiftOfFiniteType:
        from pensive.shifts.sft_construction import from_forbidden_words

        return from_forbidden_words(forbidden, symbol_alphabet, **kwargs)

    @classmethod
    def from_presentation(
        cls,
        graph: TransitionGraph,
        symbol_alphabet: frozenset[Any],
        **kwargs: Any,
    ) -> ShiftOfFiniteType:
        return cls(graph=graph, symbol_alphabet=symbol_alphabet, **kwargs)

    def forbidden_words(
        self,
        *,
        length: int | None = None,
        max_length: int | None = None,
        minimal: bool = False,
    ) -> frozenset[tuple[Any, ...]]:
        """Return forbidden words stored on or inferred from this presentation.

        With no bounds, this returns the finite defining set supplied to
        :meth:`from_forbidden_words`. For presentation-only SFTs, pass
        ``length`` or ``max_length`` to enumerate forbidden blocks up to that
        bound.
        """
        if length is None and max_length is None:
            if minimal and self._has_forbidden_word_spec:
                return _minimal_forbidden(self._forbidden_words)
            if self._has_forbidden_word_spec:
                return self._forbidden_words
            raise ValueError("pass length or max_length to infer forbidden words from a presentation")
        if length is not None and max_length is not None:
            raise ValueError("pass either length or max_length, not both")
        if length is not None and length < 0:
            raise ValueError("length must be nonnegative")
        if max_length is not None and max_length < 0:
            raise ValueError("max_length must be nonnegative")
        if length is not None:
            lengths = range(length, length + 1)
        else:
            assert max_length is not None
            lengths = range(1, max_length + 1)
        forbidden = set().union(*(self._forbidden_words_by_length(n) for n in lengths))
        return _minimal_forbidden(frozenset(forbidden)) if minimal else frozenset(forbidden)

    def _forbidden_words_by_length(self, length: int) -> frozenset[tuple[Any, ...]]:
        if length < 0:
            raise ValueError("length must be nonnegative")
        alphabet = sorted(self.symbol_alphabet, key=repr)
        allowed = set(self.factor_language(length))
        candidates = set(product(alphabet, repeat=length))
        return frozenset(candidates - allowed)


def _minimal_forbidden(words: frozenset[tuple[Any, ...]]) -> frozenset[tuple[Any, ...]]:
    minimal: set[tuple[Any, ...]] = set()
    for word in words:
        has_forbidden_subword = False
        for start in range(len(word)):
            for stop in range(start + 1, len(word) + 1):
                subword = word[start:stop]
                if subword != word and subword in words:
                    has_forbidden_subword = True
                    break
            if has_forbidden_subword:
                break
        if not has_forbidden_subword:
            minimal.add(word)
    return frozenset(minimal)
