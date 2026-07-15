"""Per-model styling metadata for Graphviz diagrams."""

from __future__ import annotations

from collections.abc import Callable, Hashable, Mapping
from dataclasses import dataclass, field
from typing import Any

from pensive.base import StateMachine
from pensive.graph import (
    ATTR_EMISSION,
    ATTR_EMISSION_DIST,
    ATTR_FUTURE_SYMBOL,
    ATTR_KIND,
    ATTR_OUTPUT,
    KIND_CALL,
    KIND_INTERNAL,
    KIND_RETURN,
    Transition,
)
from pensive.viz._edge import (
    PART_KIND,
    PART_MATCH_TAG,
    PART_MULTIPLICITY,
    PART_PROB,
    PART_QUASIPROB,
    PART_STACK,
    EdgePart,
    EdgeSpec,
    edge_spec,
)
from pensive.viz._format import (
    format_belief,
    format_distribution,
    format_prob_label,
    format_state,
    format_symbol,
)


@dataclass(frozen=True, slots=True)
class VizContext:
    """Rendering policy for one :class:`~pensive.base.StateMachine`."""

    title: str
    initial_states: frozenset[Hashable]
    accepting_states: frozenset[Hashable]
    state_labels: Mapping[Hashable, str]
    edge_label: Callable[[Transition], str]
    edge_color: Callable[[Transition], str | None] = field(default=lambda _t: None)
    edge_style: Callable[[Transition], str | None] = field(default=lambda _t: None)
    state_tooltip: Callable[[Hashable, Mapping[str, Any]], str | None] = field(default=lambda _s, _a: None)
    node_fillcolor: Callable[[Hashable], str | None] = field(default=lambda _s: None)
    show_start_node: bool = True
    highlight_initial_states: bool = True
    graph_engine: str | None = None
    rankdir: str | None = None


def _stochastic_initials(model: StateMachine) -> frozenset[Hashable]:
    dist = getattr(model, "initial_distribution", None)
    if isinstance(dist, Mapping) and dist:
        return frozenset(dist)
    quasidist = getattr(model, "initial_quasidistribution", None)
    if isinstance(quasidist, Mapping) and quasidist:
        return frozenset(quasidist)
    return frozenset()


def _state_label_with_attrs(state: Hashable, attrs: Mapping[str, Any], extras: list[str]) -> str:
    base = format_state(state)
    lines = [base, *extras]
    future = attrs.get(ATTR_FUTURE_SYMBOL)
    if future is not None:
        lines.append(f"γ={format_symbol(future)}")
    output = attrs.get(ATTR_OUTPUT)
    if output is not None:
        lines.append(f"out={format_symbol(output)}")
    emission_dist = attrs.get(ATTR_EMISSION_DIST)
    if emission_dist:
        lines.append(format_distribution(emission_dist))
    return "\\n".join(lines)


def _edge_state_label_with_attrs(attrs: Mapping[str, Any]) -> str | None:
    from pensive.generators.edge_machine import ATTR_EDGE_SOURCE, ATTR_EDGE_TARGET

    if ATTR_EDGE_SOURCE not in attrs or ATTR_EMISSION not in attrs or ATTR_EDGE_TARGET not in attrs:
        return None
    source = format_state(attrs[ATTR_EDGE_SOURCE])
    emission = format_symbol(attrs[ATTR_EMISSION])
    target = format_state(attrs[ATTR_EDGE_TARGET])
    return f"({source}, {emission}, {target})"


def _render_dot_part(part: EdgePart) -> str:
    if part.kind == PART_KIND:
        return str(part.value)
    if part.kind == PART_STACK:
        return f"↑{format_symbol(part.value)}"
    if part.kind == PART_MULTIPLICITY:
        return f"×{part.value}"
    if part.kind == PART_MATCH_TAG:
        return str(part.value)
    if part.kind in (PART_PROB, PART_QUASIPROB):
        return format_prob_label(part.value)
    # symbol / emission / output
    return format_symbol(part.value)


def _dot_edge_label(spec: EdgeSpec) -> str:
    parts = [_render_dot_part(part) for part in spec.parts]
    return " | ".join(parts) if parts else ""


def _dyck_edge_color(kind: Any) -> str | None:
    if kind == KIND_CALL:
        return "seagreen"
    if kind == KIND_RETURN:
        return "firebrick"
    if kind == KIND_INTERNAL:
        return "steelblue"
    return None


_TRANSIENT_FILL = "mistyrose"
_RECURRENT_FILL = "honeydew"
_RECURRENCE_ATOL = 1e-12


def _recurrence_fill_sets(
    model: StateMachine,
    *,
    initial_states: frozenset[Hashable],
) -> tuple[frozenset[Hashable], frozenset[Hashable]]:
    """Return ``(transient, recurrent_highlight)`` state sets for node fill colors."""
    from pensive.generators.bidirectional_epsilon_machine import BidirectionalEpsilonMachine
    from pensive.generators.epsilon_machine import EpsilonMachine
    from pensive.generators.mixed_state import MixedStatePresentation

    if isinstance(model, MixedStatePresentation):
        return model.transient_states, model.pure_states

    if isinstance(model, (EpsilonMachine, BidirectionalEpsilonMachine)):
        recurrent = model.graph.terminal_recurrent_states()
        if isinstance(model, BidirectionalEpsilonMachine):
            from pensive.generators.prob import is_positive_mass

            starts = {
                state
                for state, mass in model.joint_distribution().items()
                if is_positive_mass(mass, atol=_RECURRENCE_ATOL)
            }
            if not starts:
                starts = set(model.states())
        elif initial_states:
            starts = set(initial_states)
        else:
            starts = set(model.states())
        reachable = model.graph.forward_reachable(starts)
        transient = frozenset(reachable - recurrent)
        return transient, frozenset(reachable & recurrent)

    return frozenset(), frozenset()


def viz_context(model: StateMachine, *, style: str = "auto") -> VizContext:
    from pensive.automata.base import LabeledAutomaton
    from pensive.automata.transducers import Transducer
    from pensive.automata.vpa import VisiblyPushdownAutomaton
    from pensive.generators.base import QuasiStochasticModel, StochasticModel
    from pensive.generators.bidirectional_epsilon_machine import BidirectionalEpsilonMachine
    from pensive.generators.epsilon_machine import EpsilonMachine
    from pensive.generators.mealy import MealyHMM
    from pensive.generators.mixed_state import MixedState, MixedStatePresentation, pure_state_index
    from pensive.generators.moore import MooreHMM
    from pensive.generators.nmachine import NMachine
    from pensive.shifts.sofic_dyck import SoficDyckShift

    paper_style = style == "paper" or (style == "auto" and isinstance(model, BidirectionalEpsilonMachine))
    epsilon_paper = style == "paper"
    is_msp = isinstance(model, MixedStatePresentation)
    stationary_hmm = isinstance(model, (EpsilonMachine, MealyHMM, MooreHMM)) and not is_msp
    annotate_stationary_mass = not (paper_style or epsilon_paper or stationary_hmm)
    show_start_node = not (paper_style or stationary_hmm or is_msp)
    highlight_initial_states = (not (paper_style or stationary_hmm)) or is_msp
    graph_engine = "circo" if paper_style else None
    rankdir = None if paper_style else "LR"

    title = model.__class__.__name__
    initial_states: frozenset[Hashable] = frozenset()
    accepting_states: frozenset[Hashable] = frozenset()
    state_labels: dict[Hashable, str] = {}

    def edge_label(transition: Transition) -> str:
        return _dot_edge_label(edge_spec(model, transition))

    def _dyck_color(transition: Transition) -> str | None:
        return _dyck_edge_color(transition.data.get(ATTR_KIND))

    edge_color: Callable[[Transition], str | None] = lambda _t: None
    edge_style: Callable[[Transition], str | None] = lambda _t: None

    if isinstance(model, LabeledAutomaton):
        initial_states = model.initial_states
        accepting_states = model.accepting_states
    elif isinstance(model, Transducer):
        initial_states = model.initial_states
    elif isinstance(model, VisiblyPushdownAutomaton):
        if model.initial_state is not None:
            initial_states = frozenset({model.initial_state})
        accepting_states = model.accepting_states
        edge_color = _dyck_color
    elif isinstance(model, MixedStatePresentation):
        initial_states = frozenset({model.initial_mixed_state})
    elif isinstance(model, BidirectionalEpsilonMachine):
        initial_states = frozenset()
    elif isinstance(model, (NMachine, MooreHMM, MealyHMM, StochasticModel, QuasiStochasticModel)):
        initial_states = _stochastic_initials(model)
    elif isinstance(model, SoficDyckShift):
        edge_color = _dyck_color

    for state in model.states():
        attrs = model.graph.state_attrs(state)
        extras: list[str] = []
        if isinstance(model, MixedStatePresentation) and isinstance(state, MixedState):
            index = pure_state_index(state)
            if index is not None:
                state_labels[state] = format_state(model.basis_states[index])
                continue
            state_labels[state] = format_belief(state.belief)
            continue
        if annotate_stationary_mass and isinstance(model, StochasticModel):
            dist = getattr(model, "initial_distribution", {})
            if state in dist:
                extras.append(f"π={format_prob_label(dist[state])}")
        elif annotate_stationary_mass and isinstance(model, QuasiStochasticModel):
            quasidist = getattr(model, "initial_quasidistribution", {})
            if state in quasidist:
                extras.append(f"π={format_prob_label(quasidist[state])}")
        state_labels[state] = _edge_state_label_with_attrs(attrs) or _state_label_with_attrs(state, attrs, extras)

    transient_fill, recurrent_fill = _recurrence_fill_sets(model, initial_states=initial_states)

    def state_tooltip(state: Hashable, attrs: Mapping[str, Any]) -> str | None:
        if isinstance(model, MixedStatePresentation) and isinstance(state, MixedState):
            return format_belief(state.belief)
        emission_dist = attrs.get(ATTR_EMISSION_DIST)
        if emission_dist:
            return format_distribution(emission_dist)
        return None

    def node_fillcolor(state: Hashable) -> str | None:
        if state in transient_fill:
            return _TRANSIENT_FILL
        if state in recurrent_fill:
            return _RECURRENT_FILL
        return None

    return VizContext(
        title=title,
        initial_states=initial_states,
        accepting_states=accepting_states,
        state_labels=state_labels,
        edge_label=edge_label,
        edge_color=edge_color,
        edge_style=edge_style,
        state_tooltip=state_tooltip,
        node_fillcolor=node_fillcolor,
        show_start_node=show_start_node,
        highlight_initial_states=highlight_initial_states,
        graph_engine=graph_engine,
        rankdir=rankdir,
    )
