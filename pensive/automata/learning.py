"""NL* active learning for maximized prime átomata."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from pensive.automata.atomaton import MaximizedPrimeAtomaton
from pensive.automata.languages.base import RegularLanguage
from pensive.automata.observation import ObservationTable


def learn_maximized_prime_atomaton(
    teacher: RegularLanguage,
    alphabet: frozenset[Any],
    max_rounds: int = 32,
) -> MaximizedPrimeAtomaton:
    """Simulated NL* teacher loop using membership queries."""
    access_words: set[tuple[Any, ...]] = {()}
    experiments: set[tuple[Any, ...]] = {()}

    def membership(word: Sequence[Any]) -> bool:
        return tuple(word) in teacher

    def close_table(max_depth: int = 6) -> None:
        changed = True
        while changed:
            changed = False
            for word in list(access_words):
                if len(word) >= max_depth:
                    continue
                for symbol in alphabet:
                    successor = word + (symbol,)
                    if successor not in access_words:
                        access_words.add(successor)
                        changed = True

    for _ in range(max_rounds):
        close_table()
        table = _build_table(access_words, experiments, membership)
        hypothesis = table.to_maximized_prime_atomaton()
        counterexample = _find_counterexample(teacher, hypothesis, alphabet)
        if counterexample is None:
            return hypothesis
        for length in range(len(counterexample) + 1):
            experiments.add(counterexample[length:])

    return _build_table(access_words, experiments, membership).to_maximized_prime_atomaton()


def _build_table(
    access_words: set[tuple[Any, ...]],
    experiments: set[tuple[Any, ...]],
    membership,
) -> ObservationTable:
    membership_map: dict[tuple[Any, ...], bool] = {}
    for prefix in access_words:
        for suffix in experiments:
            membership_map[prefix + suffix] = membership(prefix + suffix)
    return ObservationTable(
        access_words=frozenset(access_words),
        experiments=frozenset(experiments),
        membership=membership_map,
    )


def _find_counterexample(
    teacher: RegularLanguage,
    hypothesis: MaximizedPrimeAtomaton,
    alphabet: frozenset[Any],
    max_len: int = 8,
) -> tuple[Any, ...] | None:
    from pensive.automata.languages._quotient_utils import _words_up_to

    for length in range(max_len + 1):
        for word in _words_up_to(length, alphabet):
            if (word in teacher) != hypothesis.recognizes(word):
                return word
    return None
