"""Graphviz rendering for sofic state-machine models."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from sofic.base import StateMachine
from sofic.viz._context import VizContext, viz_context
from sofic.viz._format import format_state
from sofic.viz._names import node_name as _node_name

if TYPE_CHECKING:
    import graphviz


def _require_graphviz() -> Any:
    try:
        import graphviz
    except ImportError as exc:
        raise ImportError(
            "Graphviz rendering requires the optional sofic[viz] extra "
            "(pip install 'sofic[viz]') and the Graphviz system binaries."
        ) from exc
    return graphviz


def _model_for_viz(model: StateMachine) -> StateMachine:
    from sofic.generators.quasi_realization import QuasiRealization

    if isinstance(model, QuasiRealization) and not any(model.states()):
        return model.to_nmachine()
    return model


def model_to_graphviz(
    model: StateMachine,
    *,
    rankdir: str | None = None,
    style: str = "auto",
    graph_attr: dict[str, str] | None = None,
    node_attr: dict[str, str] | None = None,
    edge_attr: dict[str, str] | None = None,
) -> graphviz.Digraph:
    """Return a :class:`graphviz.Digraph` for ``model``."""
    graphviz = _require_graphviz()
    model = _model_for_viz(model)
    context = viz_context(model, style=style)

    resolved_rankdir = rankdir if rankdir is not None else (context.rankdir or "LR")
    attrs = {
        "rankdir": resolved_rankdir,
        "fontsize": "12",
        "fontname": "Helvetica",
        "splines": "true",
        "overlap": "false",
    }
    if context.graph_engine == "circo":
        attrs["margin"] = "0.08"
    if graph_attr:
        attrs.update(graph_attr)

    dot = graphviz.Digraph(
        name=context.title,
        graph_attr=attrs,
        node_attr={
            "fontname": "Helvetica",
            "fontsize": "11",
            "shape": "circle",
            "style": "filled",
            "fillcolor": "white",
            **(node_attr or {}),
        },
        edge_attr={
            "fontname": "Helvetica",
            "fontsize": "10",
            "arrowsize": "0.8",
            **(edge_attr or {}),
        },
        engine=context.graph_engine,
    )

    _add_states(dot, model, context)
    _add_transitions(dot, model, context)
    return dot


def _add_states(dot: graphviz.Digraph, model: StateMachine, context: VizContext) -> None:
    for state in model.states():
        dot.node(
            _node_name(state),
            label=context.state_labels.get(state, format_state(state)),
            shape="doublecircle" if state in context.accepting_states else "circle",
            peripheries="2" if state in context.accepting_states else "1",
            penwidth="2.5" if context.highlight_initial_states and state in context.initial_states else "1.0",
            fillcolor=context.node_fillcolor(state) or "white",
            tooltip=context.state_tooltip(state, model.graph.state_attrs(state)),
        )

    if context.show_start_node and context.initial_states:
        dot.node("__start__", label="", shape="point", width="0.12", height="0.12")
        for state in sorted(context.initial_states, key=str):
            dot.edge("__start__", _node_name(state))


def _add_transitions(dot: graphviz.Digraph, model: StateMachine, context: VizContext) -> None:
    for transition in model.transitions():
        label = context.edge_label(transition)
        attrs: dict[str, str] = {}
        if label:
            attrs["label"] = label
        color = context.edge_color(transition)
        if color:
            attrs["color"] = color
        style = context.edge_style(transition)
        if style:
            attrs["style"] = style
        dot.edge(_node_name(transition.source), _node_name(transition.target), **attrs)


def model_to_svg(model: StateMachine, **kwargs: Any) -> str:
    """Render ``model`` to an SVG string."""
    dot = model_to_graphviz(model, **kwargs)
    return dot.pipe(format="svg").decode("utf-8")


def model_to_png(model: StateMachine, **kwargs: Any) -> bytes:
    """Render ``model`` to PNG bytes."""
    dot = model_to_graphviz(model, **kwargs)
    return dot.pipe(format="png")


def draw(
    model: StateMachine,
    filename: str | None = None,
    *,
    format: str = "svg",
    view: bool = False,
    **kwargs: Any,
) -> str | None:
    """Render ``model`` to a file or open it in a viewer.

    Returns the output path when ``filename`` is given.
    """
    dot = model_to_graphviz(model, **kwargs)
    if filename is None:
        dot.view(format=format, cleanup=True)
        return None
    path = dot.render(filename=filename, format=format, cleanup=True)
    if view:
        dot.view(filename=path, cleanup=True)
    return path
