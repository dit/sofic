"""Label formatting helpers for Graphviz output."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from pensive.viz import _labels
from pensive.viz._rational import two_digit_rational


def dot_escape(text: str) -> str:
    """Escape a string for use inside a Graphviz double-quoted label."""
    return (
        text.replace("\\", "\\\\")
        .replace('"', '\\"')
        .replace("\n", "\\n")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace("{", "\\{")
        .replace("}", "\\}")
    )


def format_state(state: Any) -> str:
    return _labels.format_state(state, escape=dot_escape, epsilon="ε")


def format_symbol(symbol: Any) -> str:
    return _labels.format_symbol(symbol, escape=dot_escape, epsilon="ε")


def format_prob(value: float, *, precision: int = 3) -> str:
    return dot_escape(f"{value:.{precision}g}")


def format_prob_rational(value: float, *, precision: int = 3) -> str:
    """Format a probability as p/q when exact with two-digit numerator and denominator."""
    if value <= 0.0:
        return "0"
    if value >= 1.0:
        return "1"
    frac = two_digit_rational(value)
    if frac is not None:
        if frac.numerator == frac.denominator:
            return "1"
        return dot_escape(f"{frac.numerator}/{frac.denominator}")
    return format_prob(value, precision=precision)


def format_distribution(dist: Mapping[Any, float], *, precision: int = 3) -> str:
    parts = [
        f"{format_symbol(symbol)}:{format_prob_rational(prob, precision=precision)}"
        for symbol, prob in sorted(dist.items(), key=str)
    ]
    return ", ".join(parts)


def format_belief(belief: Sequence[float], *, rational: bool = True) -> str:
    """Compact simplex label for a mixed-state belief vector."""
    formatter = format_prob_rational if rational else format_prob
    parts = [formatter(float(value)) for value in belief]
    return f"μ=({', '.join(parts)})"
