"""Core graph-backed model primitives."""

from sofic.base import StateMachine
from sofic.graph import (
    ATTR_EMISSION,
    ATTR_EMISSION_DIST,
    ATTR_EMISSION_EDGE,
    ATTR_FUTURE_SYMBOL,
    ATTR_HIER_STATE,
    ATTR_KIND,
    ATTR_MULTIPLICITY,
    ATTR_OUTPUT,
    ATTR_PROB,
    ATTR_QUASIPROB,
    ATTR_STACK_SYMBOL,
    ATTR_SYMBOL,
    EPSILON,
    KIND_CALL,
    KIND_INTERNAL,
    KIND_RETURN,
    Transition,
    TransitionGraph,
)
from sofic.indexing import StateIndex

__all__ = [
    "ATTR_EMISSION",
    "ATTR_EMISSION_DIST",
    "ATTR_EMISSION_EDGE",
    "ATTR_FUTURE_SYMBOL",
    "ATTR_HIER_STATE",
    "ATTR_KIND",
    "ATTR_MULTIPLICITY",
    "ATTR_OUTPUT",
    "ATTR_PROB",
    "ATTR_QUASIPROB",
    "ATTR_STACK_SYMBOL",
    "ATTR_SYMBOL",
    "EPSILON",
    "KIND_CALL",
    "KIND_INTERNAL",
    "KIND_RETURN",
    "StateIndex",
    "StateMachine",
    "Transition",
    "TransitionGraph",
]
