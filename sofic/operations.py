"""Cross-cutting operations on sofic state-machine models."""

from __future__ import annotations

from sofic.base import StateMachine


def reverse(model: StateMachine) -> StateMachine:
    """Reverse ``model`` using its concrete :meth:`~StateMachine.reverse` implementation.

    This is the canonical top-level ``reverse`` for all :class:`~sofic.base.StateMachine`
    subtypes (HMMs, ε-machines, automata, shifts). For finite automata, prefer
    :meth:`~sofic.automata.nfa.NFA.reverse` or import
    :func:`~sofic.automata.algorithms.reverse` directly.
    """
    return model.reverse()
