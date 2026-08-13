"""TikZ / Vaucanson rendering for sofic state-machine models."""

from __future__ import annotations

import os
import shutil
import tempfile
from collections import defaultdict
from collections.abc import Hashable, Mapping
from pathlib import Path
from typing import Any

from sofic.base import StateMachine
from sofic.graph import Transition
from sofic.viz._context import VizContext, tikz_draw_color, viz_context
from sofic.viz._edge import (
    PART_EMISSION,
    PART_KIND,
    PART_MATCH_TAG,
    PART_MULTIPLICITY,
    PART_OUTPUT,
    PART_PROB,
    PART_QUASIPROB,
    PART_STACK,
    PART_SYMBOL,
    STYLE_DYCK,
    STYLE_EDGE,
    STYLE_PROB_ONLY,
    STYLE_SYMBOL_ONLY,
    STYLE_TMC,
    STYLE_TRANSDUCER,
    STYLE_VPA,
    edge_spec,
    part_value,
)
from sofic.viz._names import node_name
from sofic.viz._tikz_format import (
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
from sofic.viz._tikz_layout import (
    edge_style,
    layout_circle,
    layout_graphviz,
    placement_to_xy,
    plan_loop_styles,
)
from sofic.viz.graphviz import _model_for_viz


def _tikz_state_label(context: VizContext, state: Hashable) -> str:
    from sofic.generators.mixed_state import MixedState, pure_state_index

    if isinstance(state, MixedState):
        if pure_state_index(state) is not None:
            label = context.state_labels.get(state, format_state_latex(state))
            return latex_escape(label)
        return format_belief_tikz_node(state.belief)
    return format_state_tikz_node(state)


def _format_dyck_match_tag_latex(tag: str) -> str:
    if tag.startswith("m") and tag[1:].isdigit():
        return rf"m_{{{tag[1:]}}}"
    return latex_escape(tag)


def _tikz_edge_label(model: StateMachine, transition: Transition) -> str:
    spec = edge_spec(model, transition)
    style = spec.style

    if style == STYLE_SYMBOL_ONLY:
        symbol = part_value(spec, PART_SYMBOL)
        return format_symbol_only_latex(symbol) if symbol is not None else ""

    if style == STYLE_TRANSDUCER:
        symbol = part_value(spec, PART_SYMBOL)
        output = part_value(spec, PART_OUTPUT)
        if symbol is None and output is None:
            return ""
        return format_transducer_edge_latex(symbol, output)

    if style == STYLE_PROB_ONLY:
        prob = part_value(spec, PART_PROB)
        return rf"${format_prob_latex(prob)}$" if prob is not None else ""

    if style == STYLE_EDGE:
        label_symbol = part_value(spec, PART_EMISSION, PART_SYMBOL)
        value = part_value(spec, PART_PROB, PART_QUASIPROB)
        if label_symbol is not None and value is not None:
            return format_edge_latex(label_symbol, value)
        if label_symbol is not None:
            return format_symbol_only_latex(label_symbol)
        if value is not None:
            return rf"${format_prob_latex(value)}$"
        return ""

    if style == STYLE_VPA:
        parts: list[str] = []
        for part in spec.parts:
            if part.kind == PART_SYMBOL:
                parts.append(format_symbol_latex(part.value))
            elif part.kind == PART_KIND:
                parts.append(latex_escape(str(part.value)))
            elif part.kind == PART_STACK:
                parts.append(rf"\uparrow{format_symbol_latex(part.value)}")
        return "$" + r"\mid".join(parts) + "$" if parts else ""

    if style == STYLE_DYCK:
        parts = []
        for part in spec.parts:
            if part.kind == PART_SYMBOL:
                parts.append(rf"\Symbol{{{format_symbol_latex(part.value)}}}")
            elif part.kind == PART_KIND:
                parts.append(rf"\mathrm{{{latex_escape(str(part.value))}}}")
            elif part.kind == PART_MATCH_TAG:
                parts.append(_format_dyck_match_tag_latex(part.value))
        return "$" + r"\mid ".join(parts) + "$" if parts else ""

    if style == STYLE_TMC:
        parts = []
        for part in spec.parts:
            if part.kind == PART_SYMBOL:
                parts.append(format_symbol_latex(part.value))
            elif part.kind == PART_MULTIPLICITY:
                parts.append(latex_escape(f"\\times {part.value}"))
        return "$" + r"\mid".join(parts) + "$" if parts else ""

    # STYLE_FALLBACK
    label_symbol = part_value(spec, PART_EMISSION, PART_SYMBOL)
    prob = part_value(spec, PART_PROB)
    if label_symbol is not None and prob is not None:
        return format_edge_latex(label_symbol, prob)
    if label_symbol is not None:
        return format_symbol_only_latex(label_symbol)
    if prob is not None:
        return rf"${format_prob_latex(prob)}$"
    return ""


def _tikz_display_kwargs(model: StateMachine) -> dict[str, Any]:
    """Kwargs for notebook / default TikZ rendering."""
    from sofic.generators.bidirectional_epsilon_machine import BidirectionalEpsilonMachine

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
    edge_label_pos: float = 0.5,
    color_by_emission: bool = True,
) -> str:
    """Return a Vaucanson-style TikZ picture for ``model``.

    Args:
        edge_label_pos: Fraction along each edge (0 = source, 1 = target) at
            which to place the edge label.  Use ``1/3`` to keep labels clear of
            mid-edge crossings.
        color_by_emission: Colour edges by emission (or input/label symbol).
            Defaults to ``True``. Visibly pushdown / Dyck kind colours take
            precedence.
    """
    model = _model_for_viz(model)
    context = viz_context(model, style=style, color_by_emission=color_by_emission)

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

    from sofic.generators.bidirectional_epsilon_machine import BidirectionalEpsilonMachine
    from sofic.generators.mixed_state import MixedState, pure_state_index

    for state in sorted(model.states(), key=repr):
        node = node_name(state)
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
            color = context.edge_color(transition)
            if color:
                draw = f"draw={tikz_draw_color(color)}"
                style_opts = f"{style_opts}, {draw}" if style_opts else draw
            edge_label = _tikz_edge_label(model, transition)
            source_name = node_name(source)
            target_name = node_name(target)
            opts = f"[{style_opts}]" if style_opts else ""
            if edge_label:
                label_opts = [
                    f"pos={edge_label_pos:g}",
                    "fill=white",
                    "inner sep=1pt",
                    "font=\\scriptsize",
                ]
                # On reciprocal pairs, park labels on opposite sides of the bend.
                if has_reverse and source != target:
                    label_opts.append("auto")
                    if source_name > target_name:
                        label_opts.append("swap")
                label_part = f" node[{','.join(label_opts)}] {{{edge_label}}}"
            else:
                label_part = ""
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
    from sofic.viz._tikz_compile import compilation_document

    return compilation_document(body)


def compile_tikz(
    fragment: str,
    *,
    format: str = "png",
) -> bytes:
    """Compile a TikZ fragment to PDF, PNG, or SVG bytes."""
    from sofic.viz._tikz_compile import compile_tikz_fragment

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

    display_kwargs = {**_tikz_display_kwargs(model), **kwargs}
    fragment = model_to_tikz(model, fragment=True, **display_kwargs)
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
