"""Structural predicates for pensive state-machine models."""

from __future__ import annotations

from collections.abc import Hashable, Iterable
from typing import Any

import numpy as np

from pensive.base import StateMachine
from pensive.graph import ATTR_EMISSION, ATTR_PROB, ATTR_SYMBOL, EPSILON, TransitionGraph


def is_unifilar_labeled(
    model: StateMachine,
    *,
    label_attr: str,
    exclude_labels: Iterable[Any] = (),
) -> bool:
    """Return whether each state has at most one outgoing edge per label value."""
    excluded = set(exclude_labels)
    seen: set[tuple[Hashable, Any]] = set()
    for transition in model.transitions():
        label = transition.data.get(label_attr)
        if label is None or label in excluded:
            continue
        key = (transition.source, label)
        if key in seen:
            return False
        seen.add(key)
    return True


def is_unifilar_symbols(model: StateMachine) -> bool:
    """Right-resolving on input symbols (ε excluded)."""
    return is_unifilar_labeled(model, label_attr=ATTR_SYMBOL, exclude_labels=(EPSILON,))


def is_unifilar_emissions(model: StateMachine) -> bool:
    """Row-unifilar on edge emissions (CM generator sense)."""
    return is_unifilar_labeled(model, label_attr=ATTR_EMISSION)


def is_deterministic_automaton(aut: Any) -> bool:
    """DFA-style determinism: one initial, no ε, unifilar on symbols."""
    if len(aut.initial_states) != 1:
        return False
    for transition in aut.transitions():
        if transition.data.get(ATTR_SYMBOL) is EPSILON:
            return False
    return is_unifilar_symbols(aut)


def is_deterministic_markov(chain: StateMachine) -> bool:
    """Each state has exactly one successor with probability 1."""
    for state in chain.states():
        outgoing = list(chain.graph.out_transitions(state))
        if len(outgoing) != 1:
            return False
        prob = float(outgoing[0].data.get(ATTR_PROB, 0.0))
        if not np.isclose(prob, 1.0):
            return False
    return True


def is_deterministic_transducer(tr: StateMachine) -> bool:
    """At most one transition per (state, input symbol)."""
    return is_unifilar_symbols(tr)
