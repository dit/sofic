"""TikZ / Vaucanson rendering for pensive state-machine models."""

from __future__ import annotations

import os
import shutil
import tempfile
from collections import defaultdict
from collections.abc import Hashable, Mapping
from pathlib import Path
from typing import Any

from pensive.base import StateMachine
from pensive.graph import (
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
from pensive.viz._context import VizContext, viz_context
from pensive.viz._tikz_format import (
    format_belief_tikz_node,
    format_edge_latex,
    format_prob_latex,
    format_state_latex,
    format_state_tikz_node,
    format_symbol_latex,
    format_symbol_only_latex,
    format_transducer_edge_latex,
    latex_escape,
)
from pensive.viz._tikz_layout import (
    edge_style,
    layout_circle,
    layout_graphviz,
    placement_to_xy,
    plan_loop_styles,
    state_node_name,
)
from pensive.viz.graphviz import _model_for_viz

_VAUCANSON_ASSET = Path(__file__).resolve().parent / "assets" / "vaucanson.tikz"


def _tikz_state_label(context: VizContext, state: Hashable) -> str:
    from pensive.generators.mixed_state import MixedState, pure_state_index

    if isinstance(state, MixedState):
        if pure_state_index(state) is not None:
            label = context.state_labels.get(state, format_state_latex(state))
            return latex_escape(label)
        return format_belief_tikz_node(state.belief)
    return format_state_tikz_node(state)


def _tikz_edge_label(model: StateMachine, transition: Transition) -> str:
    from pensive.automata.base import LabeledAutomaton
    from pensive.automata.transducers import MooreMachine, Transducer
    from pensive.automata.vpa import VisiblyPushdownAutomaton
    from pensive.generators.base import QuasiStochasticModel, StochasticModel
    from pensive.generators.bidirectional_epsilon_machine import BidirectionalEpsilonMachine
    from pensive.generators.mealy import MealyHMM
    from pensive.generators.moore import MooreHMM
    from pensive.generators.nmachine import NMachine
    from pensive.shifts.base import SymbolicModel
    from pensive.shifts.tmc import TopologicalMarkovChain

    data = transition.data
    symbol = data.get(ATTR_SYMBOL)
    emission = data.get(ATTR_EMISSION)
    prob = data.get(ATTR_PROB)
    quasiprob = data.get(ATTR_QUASIPROB)
    output = data.get(ATTR_OUTPUT)

    if isinstance(model, LabeledAutomaton):
        sym = symbol if symbol is not None else EPSILON
        return format_symbol_only_latex(sym)

    if isinstance(model, Transducer):
        if isinstance(model, MooreMachine):
            return format_symbol_only_latex(symbol) if symbol is not None else ""
        return format_transducer_edge_latex(symbol, output)

    if isinstance(model, VisiblyPushdownAutomaton):
        parts: list[str] = []
        if symbol is not None:
            parts.append(format_symbol_latex(symbol))
        kind = data.get(ATTR_KIND)
        if kind is not None:
            parts.append(latex_escape(str(kind)))
        stack = data.get(ATTR_STACK_SYMBOL)
        if stack is not None:
            parts.append(rf"\uparrow{format_symbol_latex(stack)}")
        if not parts:
            return ""
        return "$" + r"\mid".join(parts) + "$"

    if isinstance(model, BidirectionalEpsilonMachine):
        if emission is None or prob is None:
            return ""
        return format_edge_latex(emission, float(prob))

    if isinstance(model, MooreHMM):
        if prob is None:
            return ""
        return rf"${format_prob_latex(float(prob))}$"

    if isinstance(model, NMachine):
        if emission is None or quasiprob is None:
            return ""
        return format_edge_latex(emission, float(quasiprob))

    if isinstance(model, QuasiStochasticModel):
        if emission is None or quasiprob is None:
            return ""
        return format_edge_latex(emission, float(quasiprob))

    if isinstance(model, TopologicalMarkovChain):
        parts = [format_symbol_latex(symbol)] if symbol is not None else []
        mult = data.get(ATTR_MULTIPLICITY)
        if mult is not None and mult != 1:
            parts.append(latex_escape(f"\\times {mult}"))
        if not parts:
            return ""
        return "$" + r"\mid".join(parts) + "$"

    if isinstance(model, SymbolicModel):
        if symbol is None:
            return ""
        return format_symbol_only_latex(symbol)

    if isinstance(model, (MealyHMM, StochasticModel)):
        label_symbol = emission if emission is not None else symbol
        if label_symbol is None or prob is None:
            return ""
        return format_edge_latex(label_symbol, float(prob))

    label_symbol = emission if emission is not None else symbol
    if label_symbol is not None and prob is not None:
        return format_edge_latex(label_symbol, float(prob))
    if label_symbol is not None:
        return format_symbol_only_latex(label_symbol)
    if prob is not None:
        return rf"${format_prob_latex(float(prob))}$"
    return ""


def _tikz_display_kwargs(model: StateMachine) -> dict[str, Any]:
    """Kwargs for notebook / default TikZ rendering."""
    from pensive.generators.bidirectional_epsilon_machine import BidirectionalEpsilonMachine

    if isinstance(model, BidirectionalEpsilonMachine):
        return {"style": "auto", "layout": "graphviz"}
    return {"style": "auto"}


def model_to_tikz(
    model: StateMachine,
    *,
    fragment: bool = True,
    tikzset_filename: str | None = None,
    style: str = "auto",
    layout: str = "circle",
    radius: str = "2cm",
    bend_angle: int | float = 15,
    scale: float = 1,
    positions: Mapping[Hashable, tuple[float, float]] | None = None,
    angles: Mapping[Hashable, float] | None = None,
    rankdir: str | None = None,
    label: str | None = None,
) -> str:
    """Return a Vaucanson-style TikZ picture for ``model``."""
    model = _model_for_viz(model)
    context = viz_context(model, style=style)

    if layout == "circle":
        coords = layout_circle(model, radius=radius, positions=positions, angles=angles)
    elif layout == "graphviz":
        coords = layout_graphviz(model, style=style, rankdir=rankdir)
    else:
        raise ValueError(f"unknown layout {layout!r}; expected 'circle' or 'graphviz'")

    lines: list[str] = []
    if tikzset_filename:
        lines.append(rf"\tikzsetnextfilename{{{latex_escape(tikzset_filename)}}}")

    picture_options = [
        "style=vaucanson",
        f"bend angle={bend_angle}",
        f"scale={scale}",
        "every node/.style={transform shape}",
    ]
    lines.append(r"\begin{tikzpicture}[" + ",\n                    ".join(picture_options) + "]")

    from pensive.generators.bidirectional_epsilon_machine import BidirectionalEpsilonMachine
    from pensive.generators.mixed_state import MixedState, pure_state_index

    for state in sorted(model.states(), key=repr):
        node = state_node_name(state)
        placement = coords[state]
        node_label = _tikz_state_label(context, state)
        state_opts = ["state"]
        if (
            isinstance(model, BidirectionalEpsilonMachine)
            or isinstance(state, MixedState)
            and pure_state_index(state) is None
        ):
            state_opts.append(r"font=\footnotesize")
        if state in context.accepting_states:
            state_opts.append("accepting")
        fillcolor = context.node_fillcolor(state)
        if fillcolor is not None:
            state_opts.append(f"fill={fillcolor}")
        if context.highlight_initial_states and state in context.initial_states:
            if context.show_start_node:
                state_opts.append("initial")
            else:
                # Match Graphviz penwidth=2.5 (vs 1.0) with Vaucanson's line width=2 default.
                state_opts.append("line width=5pt")
        lines.append(f"  \\node [{', '.join(state_opts)}] ({node}) {placement}  {{{node_label}}};")

    transitions = list(model.transitions())
    grouped: dict[tuple[Hashable, Hashable], list[Transition]] = defaultdict(list)
    for transition in transitions:
        key = (transition.source, transition.target)
        grouped[key].append(transition)

    position_xy = {state: placement_to_xy(coords[state]) for state in coords}
    loop_styles = plan_loop_styles(position_xy, grouped)

    edge_lines: list[str] = []
    for key in sorted(grouped, key=lambda item: (repr(item[0]), repr(item[1]))):
        source, target = key
        group = grouped[key]
        has_reverse = (target, source) in grouped
        for index, transition in enumerate(group):
            style_opts = edge_style(
                source,
                target,
                parallel_index=index,
                total_parallel=len(group),
                has_reverse=has_reverse,
                loop_style=loop_styles.get((source, target, index)),
            )
            edge_label = _tikz_edge_label(model, transition)
            source_name = state_node_name(source)
            target_name = state_node_name(target)
            opts = f"[{style_opts}]" if style_opts else ""
            label_part = f" node {{{edge_label}}}" if edge_label else ""
            edge_lines.append(f"({source_name}) edge {opts}{label_part} ({target_name})")

    if edge_lines:
        lines.append("  \\path " + "\n        ".join(edge_lines) + ";")

    lines.append(r"\end{tikzpicture}")
    if label:
        lines.append(rf"\label{{{latex_escape(label)}}}")

    body = "\n".join(lines) + "\n"
    if fragment:
        return body
    return _standalone_document(body)


def _standalone_document(body: str) -> str:
    from pensive.viz._tikz_compile import compilation_document

    return compilation_document(body)


def compile_tikz(
    fragment: str,
    *,
    format: str = "png",
) -> bytes:
    """Compile a TikZ fragment to PDF, PNG, or SVG bytes."""
    from pensive.viz._tikz_compile import compile_tikz_fragment

    return compile_tikz_fragment(fragment, format=format)


def model_to_tikz_image(
    model: StateMachine,
    *,
    format: str = "png",
    **kwargs: Any,
) -> bytes:
    """Render ``model`` to compiled image bytes via ``pdflatex``."""
    display_kwargs = {**_tikz_display_kwargs(model), **kwargs}
    fragment = model_to_tikz(model, fragment=True, **display_kwargs)
    return compile_tikz(fragment, format=format)


def draw_tikz(
    model: StateMachine,
    filename: str | None = None,
    *,
    format: str | None = None,
    view: bool = False,
    **kwargs: Any,
) -> str | None:
    """Write a TikZ figure for ``model``.

    When ``format`` is ``png``, ``svg``, or ``pdf`` (or inferred from ``filename``),
    compile with ``pdflatex`` and write the rendered image. Otherwise write a
    ``.tikz`` LaTeX fragment.
    """
    import subprocess
    import sys

    inferred = None
    if filename is not None:
        inferred = Path(filename).suffix.lstrip(".").lower() or None
    resolved_format = (format or inferred or "tikz").lower()

    fragment = model_to_tikz(model, fragment=True, **_tikz_display_kwargs(model), **kwargs)
    if resolved_format == "tikz":
        if filename is None:
            return None
        path = Path(filename)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(fragment, encoding="utf-8")
        if view:
            subprocess.run(["open", str(path)], check=False)
        return str(path)

    image_bytes = compile_tikz(fragment, format=resolved_format)
    if filename is None:
        suffix = f".{resolved_format}"
        fd, tmp_name = tempfile.mkstemp(suffix=suffix)
        os.close(fd)
        tmp = Path(tmp_name)
        tmp.write_bytes(image_bytes)
        if view:
            if sys.platform == "darwin":
                subprocess.run(["open", str(tmp)], check=False)
            elif shutil.which("xdg-open"):
                subprocess.run(["xdg-open", str(tmp)], check=False)
        return str(tmp)

    path = Path(filename)
    if path.suffix == "":
        path = path.with_suffix(f".{resolved_format}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(image_bytes)
    if view:
        if sys.platform == "darwin":
            subprocess.run(["open", str(path)], check=False)
        elif shutil.which("xdg-open"):
            subprocess.run(["xdg-open", str(path)], check=False)
    return str(path)
