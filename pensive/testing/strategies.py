"""Hypothesis strategies for pensive models.

These helpers live outside the main package imports so Hypothesis remains a
test-only dependency. Install ``pensive[test]`` to use them.
"""

from __future__ import annotations

from collections.abc import Sequence
from functools import cache
from typing import Any

from pensive.automata.dfa import DFA
from pensive.automata.icdfa import icdfa_string_to_dfa, iter_icdfa_empty_strings
from pensive.generators.epsilon_machine import EpsilonMachine
from pensive.generators.topological_epsilon_enumeration import (
    idfa_string_to_epsilon_machine,
    iter_topological_epsilon_strings,
)

DEFAULT_ALPHABET = ("0", "1")


def dfas(
    *,
    alphabet: Sequence[Any] = DEFAULT_ALPHABET,
    min_states: int = 1,
    max_states: int = 3,
) -> Any:
    """Return a strategy for complete initially-connected DFAs.

    The DFA transition structures are drawn from the canonical ICDFA
    enumeration, and accepting states are drawn independently.
    """
    st = _hypothesis_strategies()
    symbols = _validate_alphabet(alphabet)
    _validate_state_bounds(min_states=min_states, max_states=max_states)
    k = len(symbols)

    @st.composite
    def strategy(draw: Any) -> DFA:
        n = draw(st.integers(min_value=min_states, max_value=max_states))
        transitions = draw(st.sampled_from(_icdfa_transition_strings(k, n)))
        final_states = draw(st.frozensets(st.integers(min_value=0, max_value=n - 1)))
        return icdfa_string_to_dfa(
            transitions,
            symbols,
            n=n,
            k=k,
            final_states=final_states,
            symbol_order=symbols,
        )

    return strategy()


def epsilon_machines(
    *,
    alphabet: Sequence[Any] = DEFAULT_ALPHABET,
    min_states: int = 1,
    max_states: int = 3,
) -> Any:
    """Return a strategy for topological epsilon-machines.

    Machines are drawn from the canonical topological epsilon-machine
    enumeration and use the enumeration module's uniform row probabilities.
    """
    st = _hypothesis_strategies()
    symbols = _validate_alphabet(alphabet)
    _validate_state_bounds(min_states=min_states, max_states=max_states)
    k = len(symbols)
    state_counts = tuple(
        n for n in range(min_states, max_states + 1) if _topological_epsilon_transition_strings(k, n)
    )
    if not state_counts:
        raise ValueError("no topological epsilon-machine strings exist for the requested bounds")

    @st.composite
    def strategy(draw: Any) -> EpsilonMachine:
        n = draw(st.sampled_from(state_counts))
        transitions = draw(st.sampled_from(_topological_epsilon_transition_strings(k, n)))
        return idfa_string_to_epsilon_machine(transitions, n=n, k=k, alphabet=symbols)

    return strategy()


def _hypothesis_strategies() -> Any:
    try:
        from hypothesis import strategies as st
    except ImportError as exc:  # pragma: no cover - exercised only without test extra
        raise ImportError("pensive.testing.strategies requires Hypothesis; install pensive[test].") from exc
    return st


def _validate_alphabet(alphabet: Sequence[Any]) -> tuple[Any, ...]:
    symbols = tuple(alphabet)
    if not symbols:
        raise ValueError("alphabet must be non-empty")
    try:
        unique = frozenset(symbols)
    except TypeError as exc:
        raise ValueError("alphabet symbols must be hashable") from exc
    if len(unique) != len(symbols):
        raise ValueError("alphabet symbols must be unique")
    return symbols


def _validate_state_bounds(*, min_states: int, max_states: int) -> None:
    if min_states < 1:
        raise ValueError("min_states must be positive")
    if max_states < 1:
        raise ValueError("max_states must be positive")
    if min_states > max_states:
        raise ValueError("min_states must be less than or equal to max_states")


@cache
def _icdfa_transition_strings(k: int, n: int) -> tuple[tuple[int, ...], ...]:
    return tuple(iter_icdfa_empty_strings(k, n))


@cache
def _topological_epsilon_transition_strings(k: int, n: int) -> tuple[tuple[int, ...], ...]:
    return tuple(iter_topological_epsilon_strings(k, n))
