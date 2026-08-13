"""Per-model styling metadata for Graphviz diagrams."""

from __future__ import annotations

from collections.abc import Callable, Hashable, Mapping
from dataclasses import dataclass, field
from typing import Any

from sofic.base import StateMachine
from sofic.graph import (
    ATTR_EMISSION,
    ATTR_EMISSION_DIST,
    ATTR_FUTURE_SYMBOL,
    ATTR_KIND,
    ATTR_OUTPUT,
    EPSILON,
    KIND_CALL,
    KIND_INTERNAL,
    KIND_RETURN,
    Transition,
)
from sofic.viz._edge import (
    PART_EMISSION,
    PART_KIND,
    PART_MATCH_TAG,
    PART_MULTIPLICITY,
    PART_PROB,
    PART_QUASIPROB,
    PART_STACK,
    PART_SYMBOL,
    EdgePart,
    EdgeSpec,
    edge_spec,
    part_value,
)
from sofic.viz._format import (
    format_belief,
    format_distribution,
    format_prob_label,
    format_state,
    format_symbol,
)


@dataclass(frozen=True, slots=True)
class VizContext:
    """Rendering policy for one :class:`~sofic.base.StateMachine`."""

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
    from sofic.generators.edge_machine import ATTR_EDGE_SOURCE, ATTR_EDGE_TARGET

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


# Tableau 10 — Graphviz hex, stable assignment by sorted ``repr`` of the symbol.
EMISSION_PALETTE: tuple[str, ...] = (
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
)

_NAMED_RGB: dict[str, tuple[int, int, int]] = {
    "seagreen": (46, 139, 87),
    "firebrick": (178, 34, 34),
    "steelblue": (70, 130, 180),
}


def _rgb_from_graphviz_color(color: str) -> tuple[int, int, int] | None:
    if color.startswith("#") and len(color) == 7:
        return int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16)
    return _NAMED_RGB.get(color)


def tikz_draw_color(color: str) -> str:
    """TikZ ``draw=`` value for a Graphviz color string (hex or named)."""
    rgb = _rgb_from_graphviz_color(color)
    if rgb is None:
        return color
    red, green, blue = rgb
    return f"{{rgb,255:red,{red};green,{green};blue,{blue}}}"


def _emission_color_key(model: StateMachine, transition: Transition) -> Any:
    """Emission (else input/label symbol) used to color ``transition``, or None."""
    key = part_value(edge_spec(model, transition), PART_EMISSION, PART_SYMBOL)
    if key is None or key is EPSILON:
        return None
    return key


def _emission_color_map(model: StateMachine) -> dict[Any, str]:
    keys = sorted(
        {key for transition in model.transitions() if (key := _emission_color_key(model, transition)) is not None},
        key=repr,
    )
    return {key: EMISSION_PALETTE[index % len(EMISSION_PALETTE)] for index, key in enumerate(keys)}


_TRANSIENT_FILL = "mistyrose"
_RECURRENT_FILL = "honeydew"
_RECURRENCE_ATOL = 1e-12


def _recurrence_fill_sets(
    model: StateMachine,
    *,
    initial_states: frozenset[Hashable],
) -> tuple[frozenset[Hashable], frozenset[Hashable]]:
    """Return ``(transient, recurrent_highlight)`` state sets for node fill colors."""
    from sofic.generators.bidirectional_epsilon_machine import BidirectionalEpsilonMachine
    from sofic.generators.epsilon_machine import EpsilonMachine
    from sofic.generators.mixed_state import MixedStatePresentation

    if isinstance(model, MixedStatePresentation):
        return model.transient_states, model.pure_states

    if isinstance(model, (EpsilonMachine, BidirectionalEpsilonMachine)):
        recurrent = model.graph.terminal_recurrent_states()
        if isinstance(model, BidirectionalEpsilonMachine):
            from sofic.generators.prob import is_positive_mass

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


def viz_context(
    model: StateMachine,
    *,
    style: str = "auto",
    color_by_emission: bool = True,
) -> VizContext:
    from sofic.automata.base import LabeledAutomaton
    from sofic.automata.transducers import Transducer
    from sofic.automata.vpa import VisiblyPushdownAutomaton
    from sofic.generators.base import QuasiStochasticModel, StochasticModel
    from sofic.generators.bidirectional_epsilon_machine import BidirectionalEpsilonMachine
    from sofic.generators.epsilon_machine import EpsilonMachine
    from sofic.generators.mealy import MealyHMM
    from sofic.generators.mixed_state import MixedState, MixedStatePresentation, pure_state_index
    from sofic.generators.moore import MooreHMM
    from sofic.generators.nmachine import NMachine
    from sofic.shifts.sofic_dyck import SoficDyckShift

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

    specialized_color: Callable[[Transition], str | None] = lambda _t: None
    edge_style: Callable[[Transition], str | None] = lambda _t: None
    emission_colors = _emission_color_map(model) if color_by_emission else {}

    if isinstance(model, LabeledAutomaton):
        initial_states = model.initial_states
        accepting_states = model.accepting_states
    elif isinstance(model, Transducer):
        initial_states = model.initial_states
    elif isinstance(model, VisiblyPushdownAutomaton):
        if model.initial_state is not None:
            initial_states = frozenset({model.initial_state})
        accepting_states = model.accepting_states
        specialized_color = _dyck_color
    elif isinstance(model, MixedStatePresentation):
        initial_states = frozenset({model.initial_mixed_state})
    elif isinstance(model, BidirectionalEpsilonMachine):
        initial_states = frozenset()
    elif isinstance(model, (NMachine, MooreHMM, MealyHMM, StochasticModel, QuasiStochasticModel)):
        initial_states = _stochastic_initials(model)
    elif isinstance(model, SoficDyckShift):
        specialized_color = _dyck_color

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

    def edge_color(transition: Transition) -> str | None:
        color = specialized_color(transition)
        if color:
            return color
        key = _emission_color_key(model, transition)
        if key is None:
            return None
        return emission_colors.get(key)

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
