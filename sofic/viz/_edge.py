"""Backend-agnostic edge-label decomposition for viz.

A single dispatch (:func:`edge_spec`) maps each model type to a structured
:class:`EdgeSpec`: an ordered list of semantic :class:`EdgePart` pieces plus a
``style`` tag selecting the TikZ rendering family. The Graphviz backend renders
parts uniformly (join with ``" | "``); the TikZ backend switches on ``style``.

Adding a new model type means adding one branch here, not editing two parallel
per-backend ladders.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from sofic.base import StateMachine
from sofic.graph import (
    ATTR_EMISSION,
    ATTR_KIND,
    ATTR_MULTIPLICITY,
    ATTR_OUTPUT,
    ATTR_PROB,
    ATTR_QUASIPROB,
    ATTR_STACK_SYMBOL,
    ATTR_SYMBOL,
    EPSILON,
    Transition,
)

# Semantic part kinds.
PART_SYMBOL = "symbol"
PART_EMISSION = "emission"
PART_PROB = "prob"
PART_QUASIPROB = "quasiprob"
PART_OUTPUT = "output"
PART_KIND = "kind"
PART_STACK = "stack"
PART_MULTIPLICITY = "multiplicity"
PART_MATCH_TAG = "match_tag"

# TikZ rendering families.
STYLE_SYMBOL_ONLY = "symbol_only"
STYLE_TRANSDUCER = "transducer"
STYLE_EDGE = "edge"
STYLE_PROB_ONLY = "prob_only"
STYLE_VPA = "vpa"
STYLE_DYCK = "dyck"
STYLE_TMC = "tmc"
STYLE_FALLBACK = "fallback"


@dataclass(frozen=True, slots=True)
class EdgePart:
    """One semantic piece of an edge label."""

    kind: str
    value: Any


@dataclass(frozen=True, slots=True)
class EdgeSpec:
    """Structured edge label: ordered parts (Graphviz) + TikZ ``style``."""

    parts: tuple[EdgePart, ...]
    style: str


def part_value(spec: EdgeSpec, *kinds: str) -> Any:
    """Return the value of the first part whose kind is in ``kinds`` (or None)."""
    for part in spec.parts:
        if part.kind in kinds:
            return part.value
    return None


def _primary_symbol_part(emission: Any, symbol: Any, *, symbol_fallback: bool) -> EdgePart | None:
    """Emission-or-symbol primary label part (emission preferred)."""
    if emission is not None:
        return EdgePart(PART_EMISSION, emission)
    if symbol_fallback and symbol is not None:
        return EdgePart(PART_SYMBOL, symbol)
    return None


def _prob_part(prob: Any, quasiprob: Any) -> EdgePart | None:
    if prob is not None:
        return EdgePart(PART_PROB, prob)
    if quasiprob is not None:
        return EdgePart(PART_QUASIPROB, quasiprob)
    return None


def edge_spec(model: StateMachine, transition: Transition) -> EdgeSpec:
    """Decompose ``transition`` for ``model`` into a backend-agnostic spec."""
    from sofic.automata.base import LabeledAutomaton
    from sofic.automata.transducers import MooreMachine, Transducer
    from sofic.automata.vpa import VisiblyPushdownAutomaton
    from sofic.generators.base import QuasiStochasticModel, StochasticModel
    from sofic.generators.bidirectional_epsilon_machine import BidirectionalEpsilonMachine
    from sofic.generators.mealy import MealyHMM
    from sofic.generators.mixed_state import MixedStatePresentation
    from sofic.generators.moore import MooreHMM
    from sofic.generators.nmachine import NMachine
    from sofic.shifts.base import SymbolicModel
    from sofic.shifts.sofic_dyck import SoficDyckShift, transition_ref
    from sofic.shifts.tmc import TopologicalMarkovChain

    data = transition.data
    symbol = data.get(ATTR_SYMBOL)
    emission = data.get(ATTR_EMISSION)
    prob = data.get(ATTR_PROB)
    quasiprob = data.get(ATTR_QUASIPROB)

    if isinstance(model, LabeledAutomaton):
        sym = symbol if symbol is not None else EPSILON
        return EdgeSpec((EdgePart(PART_SYMBOL, sym),), STYLE_SYMBOL_ONLY)

    if isinstance(model, Transducer):
        if isinstance(model, MooreMachine):
            parts = (EdgePart(PART_SYMBOL, symbol),) if symbol is not None else ()
            return EdgeSpec(parts, STYLE_SYMBOL_ONLY)
        parts_list: list[EdgePart] = []
        if symbol is not None:
            parts_list.append(EdgePart(PART_SYMBOL, symbol))
        output = data.get(ATTR_OUTPUT)
        if output is not None:
            parts_list.append(EdgePart(PART_OUTPUT, output))
        return EdgeSpec(tuple(parts_list), STYLE_TRANSDUCER)

    if isinstance(model, VisiblyPushdownAutomaton):
        parts_list = []
        if symbol is not None:
            parts_list.append(EdgePart(PART_SYMBOL, symbol))
        kind = data.get(ATTR_KIND)
        if kind is not None:
            parts_list.append(EdgePart(PART_KIND, kind))
        stack = data.get(ATTR_STACK_SYMBOL)
        if stack is not None:
            parts_list.append(EdgePart(PART_STACK, stack))
        return EdgeSpec(tuple(parts_list), STYLE_VPA)

    if isinstance(model, NMachine):
        parts_list = []
        if emission is not None:
            parts_list.append(EdgePart(PART_EMISSION, emission))
        if quasiprob is not None:
            parts_list.append(EdgePart(PART_QUASIPROB, quasiprob))
        return EdgeSpec(tuple(parts_list), STYLE_EDGE)

    if isinstance(model, MooreHMM):
        parts = (EdgePart(PART_PROB, prob),) if prob is not None else ()
        return EdgeSpec(parts, STYLE_PROB_ONLY)

    if isinstance(model, (BidirectionalEpsilonMachine, MixedStatePresentation, MealyHMM, StochasticModel)):
        parts_list = []
        primary = _primary_symbol_part(emission, symbol, symbol_fallback=True)
        if primary is not None:
            parts_list.append(primary)
        prob_part = _prob_part(prob, None)
        if prob_part is not None:
            parts_list.append(prob_part)
        return EdgeSpec(tuple(parts_list), STYLE_EDGE)

    if isinstance(model, QuasiStochasticModel):
        parts_list = []
        if emission is not None:
            parts_list.append(EdgePart(PART_EMISSION, emission))
        if quasiprob is not None:
            parts_list.append(EdgePart(PART_QUASIPROB, quasiprob))
        return EdgeSpec(tuple(parts_list), STYLE_EDGE)

    if isinstance(model, SoficDyckShift):
        parts_list = []
        if symbol is not None:
            parts_list.append(EdgePart(PART_SYMBOL, symbol))
        kind = data.get(ATTR_KIND)
        if kind is not None:
            parts_list.append(EdgePart(PART_KIND, kind))
        match_tags = _dyck_match_tags(model.matched_edges)
        for tag in match_tags.get(transition_ref(transition), ()):
            parts_list.append(EdgePart(PART_MATCH_TAG, tag))
        return EdgeSpec(tuple(parts_list), STYLE_DYCK)

    if isinstance(model, TopologicalMarkovChain):
        parts_list = []
        if symbol is not None:
            parts_list.append(EdgePart(PART_SYMBOL, symbol))
        mult = data.get(ATTR_MULTIPLICITY)
        if mult is not None and mult != 1:
            parts_list.append(EdgePart(PART_MULTIPLICITY, mult))
        return EdgeSpec(tuple(parts_list), STYLE_TMC)

    if isinstance(model, SymbolicModel):
        parts = (EdgePart(PART_SYMBOL, symbol),) if symbol is not None else ()
        return EdgeSpec(parts, STYLE_SYMBOL_ONLY)

    parts_list = []
    if symbol is not None:
        parts_list.append(EdgePart(PART_SYMBOL, symbol))
    if emission is not None:
        parts_list.append(EdgePart(PART_EMISSION, emission))
    if prob is not None:
        parts_list.append(EdgePart(PART_PROB, prob))
    if quasiprob is not None:
        parts_list.append(EdgePart(PART_QUASIPROB, quasiprob))
    return EdgeSpec(tuple(parts_list), STYLE_FALLBACK)


def _dyck_match_tags(matched_edges: Any) -> dict[Any, tuple[str, ...]]:
    tags: dict[Any, list[str]] = {}
    for index, (call_ref, return_ref) in enumerate(sorted(matched_edges, key=repr), start=1):
        tag = f"m{index}"
        tags.setdefault(call_ref, []).append(tag)
        tags.setdefault(return_ref, []).append(tag)
    return {ref: tuple(ref_tags) for ref, ref_tags in tags.items()}
