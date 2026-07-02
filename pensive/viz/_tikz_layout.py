"""Layout helpers for TikZ machine figures."""

from __future__ import annotations

import math
import re
from collections import defaultdict
from collections.abc import Hashable, Mapping
from typing import Any

from pensive.base import StateMachine
from pensive.viz._names import node_name

_LOOP_STYLES = ("loop above", "loop right", "loop below", "loop left")
_BEND_STYLES = ("bend left", "bend right")
_LOOP_STYLE_ANGLES = {
    "loop above": 90.0,
    "loop right": 0.0,
    "loop below": -90.0,
    "loop left": 180.0,
}
_LOOP_CONFLICT_RADIUS = 55.0


def layout_circle(
    model: StateMachine,
    *,
    radius: str = "2cm",
    positions: Mapping[Hashable, tuple[float, float]] | None = None,
    angles: Mapping[Hashable, float] | None = None,
) -> dict[Hashable, str]:
    """Return TikZ ``at (...)`` clauses for each state on a circle."""
    states = sorted(model.states(), key=repr)
    if not states:
        return {}

    radius_value, radius_unit = _split_dim(radius)
    coords: dict[Hashable, str] = {}

    if positions is not None:
        for state in states:
            if state not in positions:
                raise ValueError(f"missing position override for state {state!r}")
            x, y = positions[state]
            coords[state] = f"at ({x}{radius_unit}, {y}{radius_unit})"
        return coords

    count = len(states)
    for index, state in enumerate(states):
        angle = angles[state] if angles is not None and state in angles else 90.0 - index * (360.0 / count)
        coords[state] = f"at ({angle:.4g}:{radius_value}{radius_unit})"
    return coords


def _split_dim(value: str) -> tuple[str, str]:
    match = re.fullmatch(r"([0-9.]+)([a-zA-Z]+)", value.strip())
    if match is None:
        raise ValueError(f"expected dimension like '2cm', got {value!r}")
    return match.group(1), match.group(2)


def layout_graphviz(
    model: StateMachine,
    *,
    style: str = "auto",
    rankdir: str | None = None,
) -> dict[Hashable, str]:
    """Return TikZ ``at (...)`` clauses using Graphviz node positions."""
    from pensive.viz.graphviz import model_to_graphviz

    dot = model_to_graphviz(model, style=style, rankdir=rankdir)
    plain = dot.pipe(format="plain").decode("utf-8")
    positions = _parse_plain_positions(plain)
    coords: dict[Hashable, str] = {}
    for state in model.states():
        node = node_name(state)
        if node not in positions:
            raise RuntimeError(f"graphviz layout missing position for node {node!r}")
        x_cm, y_cm = positions[node]
        coords[state] = f"at ({x_cm:.4f}cm, {y_cm:.4f}cm)"
    return coords


def _parse_plain_positions(plain: str) -> dict[str, tuple[float, float]]:
    """Parse Graphviz plain format node lines into cm coordinates.

    Graphviz ``plain`` node positions are in **inches** (same unit as the graph
    width/height on the ``graph`` header line), not PostScript points.
    """
    inch_to_cm = 2.54
    positions: dict[str, tuple[float, float]] = {}
    for line in plain.splitlines():
        if not line.startswith("node "):
            continue
        parts = line.split()
        if len(parts) < 4:
            continue
        name = parts[1].strip('"')
        x_cm = float(parts[2]) * inch_to_cm
        y_cm = float(parts[3]) * inch_to_cm
        positions[name] = (x_cm, y_cm)
    return positions


def placement_to_xy(placement: str) -> tuple[float, float]:
    """Convert a TikZ ``at (...)`` clause to ``(x, y)`` in cm (0° = east)."""
    inner = placement.removeprefix("at (").removesuffix(")")
    polar = re.fullmatch(r"([-\d.]+):([-\d.]+)(cm)", inner)
    if polar is not None:
        angle_deg = float(polar.group(1))
        radius = float(polar.group(2))
        rad = math.radians(angle_deg)
        return radius * math.cos(rad), radius * math.sin(rad)
    cart = re.fullmatch(r"([-\d.]+)cm, ([-\d.]+)cm", inner)
    if cart is not None:
        return float(cart.group(1)), float(cart.group(2))
    raise ValueError(f"unsupported placement {placement!r}")


def _angle_diff_deg(a: float, b: float) -> float:
    return abs((a - b + 180.0) % 360.0 - 180.0)


def _outgoing_bearings(
    source: Hashable,
    positions: Mapping[Hashable, tuple[float, float]],
    grouped: Mapping[tuple[Hashable, Hashable], list[Any]],
) -> list[float]:
    """Bearings at ``source`` that outgoing edges already occupy."""
    bearings: list[float] = []
    sx, sy = positions[source]
    for (src, target), _group in grouped.items():
        if src != source or target == source:
            continue
        if target not in positions:
            continue
        tx, ty = positions[target]
        angle = math.degrees(math.atan2(ty - sy, tx - sx))
        bearings.append(angle)
        if (target, source) in grouped:
            bearings.append(angle + 90.0)
    return bearings


def _pick_loop_style(
    bearings: list[float],
    used: list[str],
    *,
    preferred_order: tuple[str, ...] = _LOOP_STYLES,
) -> str:
    def score(style: str) -> float:
        anchor = _LOOP_STYLE_ANGLES[style]
        penalty = 1000.0 if style in used else 0.0
        for bearing in bearings:
            diff = _angle_diff_deg(anchor, bearing)
            if diff < _LOOP_CONFLICT_RADIUS:
                penalty += (_LOOP_CONFLICT_RADIUS - diff) ** 2
        return penalty

    return min(preferred_order, key=score)


def plan_loop_styles(
    positions: Mapping[Hashable, tuple[float, float]],
    grouped: Mapping[tuple[Hashable, Hashable], list[Any]],
) -> dict[tuple[Hashable, Hashable, int], str]:
    """Assign self-loop styles that avoid outgoing-edge corridors."""
    styles: dict[tuple[Hashable, Hashable, int], str] = {}
    loops_by_state: dict[Hashable, list[tuple[Hashable, Hashable, int]]] = defaultdict(list)
    for key, group in grouped.items():
        source, target = key
        if source != target:
            continue
        for index in range(len(group)):
            loops_by_state[source].append((source, target, index))

    for source, loops in loops_by_state.items():
        bearings = _outgoing_bearings(source, positions, grouped)
        used: list[str] = []
        for key in loops:
            style = _pick_loop_style(bearings, used)
            styles[key] = style
            used.append(style)
    return styles


def edge_style(
    source: Hashable,
    target: Hashable,
    *,
    parallel_index: int,
    total_parallel: int,
    has_reverse: bool = False,
    loop_style: str | None = None,
) -> str:
    """Choose bend/loop options for one transition."""
    if source == target:
        return loop_style or _LOOP_STYLES[parallel_index % len(_LOOP_STYLES)]
    if has_reverse:
        return _reciprocal_bend(parallel_index, total_parallel)
    if total_parallel == 1:
        return ""
    if total_parallel == 2:
        return _BEND_STYLES[parallel_index % 2]
    base = 30 + 25 * parallel_index
    return f"bend left, out={base}, in={180 - base}"


def _reciprocal_bend(parallel_index: int, total_parallel: int) -> str:
    """Bend both directions of a reciprocal pair the same way (cpfci convention)."""
    if total_parallel == 1:
        return "bend left"
    if total_parallel == 2 and parallel_index == 0:
        return "bend left, out=45, in=135"
    if total_parallel == 2:
        return "bend left"
    base = 30 + 25 * parallel_index
    return f"bend left, out={base}, in={180 - base}"
