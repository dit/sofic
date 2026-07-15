"""Time-reversal helpers for stochastic generators."""

from __future__ import annotations

from typing import Any, TypeVar

from pensive.base import StateMachine
from pensive.generators.prob import (
    as_prob,
    has_symbolic,
    is_positive_mass,
    is_zero,
    simplify_prob,
)
from pensive.graph import ATTR_EMISSION, ATTR_EMISSION_DIST, ATTR_PROB, TransitionGraph

S = TypeVar("S", bound=StateMachine)


def is_markov_like(model: StateMachine) -> bool:
    """Return whether ``model`` has only Markov transition probabilities on edges."""
    for state in model.states():
        if model.graph.state_attrs(state).get(ATTR_EMISSION_DIST) is not None:
            return False
    for transition in model.transitions():
        data = transition.data
        if ATTR_EMISSION in data or ATTR_EMISSION_DIST in data:
            return False
        if ATTR_PROB not in data:
            return False
    return True


def time_reverse_stochastic(model: S) -> S:  # noqa: UP047 - keep Python 3.11 compatibility.
    """Build the time-reversed chain using the forward stationary distribution."""
    pi = model.stationary_distribution()
    idx = model.reindex()
    rev = model.copy()
    rev.graph = TransitionGraph()
    for state in idx.states:
        rev.graph.add_state(state)

    symbolic = pi.dtype == object or has_symbolic(pi.ravel())
    for source in idx.states:
        i = idx.index(source)
        for transition in model.graph.out_transitions(source):
            target = transition.target
            j = idx.index(target)
            prob = as_prob(transition.data.get(ATTR_PROB, 0.0))
            if not is_positive_mass(prob) or is_zero(pi[j]):
                continue
            if symbolic or has_symbolic([prob]):
                rev_prob = simplify_prob(as_prob(pi[i]) * as_prob(prob) / as_prob(pi[j]))
            else:
                rev_prob = float(pi[i] * float(prob) / float(pi[j]))
            attrs: dict[str, Any] = {ATTR_PROB: as_prob(rev_prob)}
            if ATTR_EMISSION in transition.data:
                attrs[ATTR_EMISSION] = transition.data[ATTR_EMISSION]
            rev.graph.add_transition(target, source, **attrs)

    if hasattr(rev, "initial_distribution"):
        if symbolic:
            rev.initial_distribution = {idx.state(i): as_prob(pi[i]) for i in range(len(idx))}
        else:
            rev.initial_distribution = {idx.state(i): float(pi[i]) for i in range(len(idx))}
    return rev
