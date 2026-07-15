"""Finite-state process equivalence for HMM presentations.

The algorithm follows the sufficient history/future word-list construction used
by CMPy, based on Dan Upper's finite-dimensional process-equivalence test.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Hashable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from sofic.generators.base import HiddenMarkovModel
from sofic.generators.hmm_inference import _emission_transition_tensors_from_mealy
from sofic.generators.words import _start_vector

_DEFAULT_RTOL = 1e-9
_DEFAULT_ATOL = 1e-12


def is_equal_process(
    g1: HiddenMarkovModel,
    g2: HiddenMarkovModel,
    *,
    start1: Hashable | Mapping[Hashable, float] | Sequence[float] | np.ndarray | None = None,
    start2: Hashable | Mapping[Hashable, float] | Sequence[float] | np.ndarray | None = None,
    rtol: float | None = None,
    atol: float | None = None,
) -> bool:
    """Return whether two finite HMMs generate the same word process.

    The test compares finite bases for the history and future spaces rather
    than brute-force enumerating all words up to a fixed cutoff.
    """
    if set(g1.observation_alphabet) != set(g2.observation_alphabet):
        return False

    rtol = _DEFAULT_RTOL if rtol is None else float(rtol)
    atol = _DEFAULT_ATOL if atol is None else float(atol)

    hf1 = _HistoryFutureWordList.from_hmm(g1, start=start1)
    hf2 = _HistoryFutureWordList.from_hmm(g2, start=start2)

    history_words = _sorted_words(set(hf1.history_word_list()) | set(hf2.history_word_list()))
    future_words = _sorted_words(set(hf1.future_word_list()) | set(hf2.future_word_list()))

    if not _compare_future_probabilities(hf1, hf2, future_words, rtol=rtol, atol=atol):
        return False
    if not _compare_conditional_tables(hf1, hf2, history_words, future_words, rtol=rtol, atol=atol):
        return False

    one_step_future_words = set(future_words)
    for word in future_words:
        for symbol in hf1.alphabet:
            one_step_future_words.add((symbol,) + word)
    extended_future_words = _sorted_words(one_step_future_words)
    return _compare_conditional_tables(hf1, hf2, history_words, extended_future_words, rtol=rtol, atol=atol)


@dataclass
class _HistoryFutureWordList:
    alphabet: tuple[Any, ...]
    matrices: dict[Any, np.ndarray]
    start: np.ndarray
    _word_matrices: dict[tuple[Any, ...], np.ndarray] = field(default_factory=dict)
    _future_words: list[tuple[Any, ...]] | None = None
    _history_words: list[tuple[Any, ...]] | None = None

    @classmethod
    def from_hmm(
        cls,
        hmm: HiddenMarkovModel,
        *,
        start: Hashable | Mapping[Hashable, float] | Sequence[float] | np.ndarray | None = None,
    ) -> _HistoryFutureWordList:
        mealy = hmm.to_mealy()
        pi, matrices = _emission_transition_tensors_from_mealy(mealy)
        return cls(
            alphabet=tuple(sorted(mealy.observation_alphabet, key=repr)),
            matrices=matrices,
            start=_start_vector(mealy, pi, start),
        )

    @property
    def dimension(self) -> int:
        return len(self.start)

    def word_matrix(self, word: tuple[Any, ...]) -> np.ndarray:
        if not self._word_matrices:
            self._word_matrices[()] = np.eye(self.dimension, dtype=float)
        cached = self._word_matrices.get(word)
        if cached is not None:
            return cached
        if not word:
            return self._word_matrices[()]

        prefix = word[:-1]
        prefix_matrix = self.word_matrix(prefix)
        symbol_matrix = self.matrices.get(word[-1])
        if symbol_matrix is None:
            matrix = np.zeros((self.dimension, self.dimension), dtype=float)
        else:
            matrix = prefix_matrix @ symbol_matrix
        self._word_matrices[word] = matrix
        return matrix

    def future_vector(self, word: tuple[Any, ...]) -> np.ndarray:
        return self.word_matrix(word) @ np.ones(self.dimension, dtype=float)

    def history_vector(self, word: tuple[Any, ...]) -> np.ndarray:
        vector = self.start @ self.word_matrix(word)
        total = float(vector.sum())
        if total != 0.0:
            vector = vector / total
        return vector

    def future_word_list(self) -> list[tuple[Any, ...]]:
        if self._future_words is not None:
            return list(self._future_words)

        queue: deque[tuple[Any, ...]] = deque([()])
        words: list[tuple[Any, ...]] = []
        basis: list[np.ndarray] = []
        rank = 0
        while queue:
            word = queue.popleft()
            candidate = self.future_vector(word)
            matrix = np.vstack([*basis, candidate]) if basis else np.asarray([candidate])
            new_rank = np.linalg.matrix_rank(matrix)
            if new_rank <= rank:
                continue
            rank = int(new_rank)
            basis.append(candidate)
            words.append(word)
            for symbol in self.alphabet:
                queue.append((symbol,) + word)

        self._future_words = _sorted_words(words)
        return list(self._future_words)

    def history_word_list(self) -> list[tuple[Any, ...]]:
        if self._history_words is not None:
            return list(self._history_words)

        queue: deque[tuple[Any, ...]] = deque([()])
        words: list[tuple[Any, ...]] = []
        basis: list[np.ndarray] = []
        rank = 0
        while queue:
            word = queue.popleft()
            candidate = self.history_vector(word)
            matrix = np.vstack([*basis, candidate]) if basis else np.asarray([candidate])
            new_rank = np.linalg.matrix_rank(matrix)
            if new_rank <= rank:
                continue
            rank = int(new_rank)
            basis.append(candidate)
            words.append(word)
            for symbol in self.alphabet:
                queue.append(word + (symbol,))

        self._history_words = _sorted_words(words)
        return list(self._history_words)

    def future_matrix(self, words: Sequence[tuple[Any, ...]]) -> np.ndarray:
        if not words:
            return np.zeros((self.dimension, 0), dtype=float)
        return np.column_stack([self.future_vector(word) for word in words])

    def history_matrix(self, words: Sequence[tuple[Any, ...]]) -> np.ndarray:
        if not words:
            return np.zeros((0, self.dimension), dtype=float)
        return np.vstack([self.history_vector(word) for word in words])


def _compare_future_probabilities(
    hf1: _HistoryFutureWordList,
    hf2: _HistoryFutureWordList,
    future_words: Sequence[tuple[Any, ...]],
    *,
    rtol: float,
    atol: float,
) -> bool:
    probs1 = hf1.start @ hf1.future_matrix(future_words)
    probs2 = hf2.start @ hf2.future_matrix(future_words)
    return bool(np.allclose(probs1, probs2, rtol=rtol, atol=atol))


def _compare_conditional_tables(
    hf1: _HistoryFutureWordList,
    hf2: _HistoryFutureWordList,
    history_words: Sequence[tuple[Any, ...]],
    future_words: Sequence[tuple[Any, ...]],
    *,
    rtol: float,
    atol: float,
) -> bool:
    table1 = hf1.history_matrix(history_words) @ hf1.future_matrix(future_words)
    table2 = hf2.history_matrix(history_words) @ hf2.future_matrix(future_words)
    return bool(np.allclose(table1, table2, rtol=rtol, atol=atol))


def _sorted_words(words: Sequence[tuple[Any, ...]] | set[tuple[Any, ...]]) -> list[tuple[Any, ...]]:
    return sorted(words, key=lambda word: (len(word), tuple(repr(symbol) for symbol in word)))
