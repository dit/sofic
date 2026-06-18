"""Cross-cutting operations on pensive state-machine models."""

from __future__ import annotations

from pensive.base import StateMachine


def reverse(model: StateMachine) -> StateMachine:
    """Reverse ``model`` using its concrete :meth:`~StateMachine.reverse` implementation.

    This is the canonical top-level ``reverse`` for all :class:`~pensive.base.StateMachine`
    subtypes (HMMs, ε-machines, automata, shifts). For finite automata, prefer
    :meth:`~pensive.automata.nfa.NFA.reverse` or import
    :func:`~pensive.automata.algorithms.reverse` directly.
    """
    return model.reverse()
