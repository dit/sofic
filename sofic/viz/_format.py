"""Label formatting helpers for Graphviz output."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from fractions import Fraction
from typing import Any

from sofic.viz import _labels


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


def _dot_fraction(frac: Fraction) -> str:
    return dot_escape(f"{frac.numerator}/{frac.denominator}")


def _dot_sympy(expr: Any) -> str:
    import sympy as sp

    return dot_escape(sp.sstr(expr))


def format_prob_rational(value: float, *, precision: int = 3) -> str:
    """Format a probability as p/q when exact with two-digit numerator and denominator."""
    return format_prob_label(float(value), precision=precision)


def format_prob_label(value: Any, *, precision: int = 3) -> str:
    """Format a probability for Graphviz edge/π labels (float or sympy Expr)."""
    return _labels.format_prob(
        value, escape=dot_escape, fraction=_dot_fraction, symbolic=_dot_sympy, precision=precision
    )


def format_distribution(dist: Mapping[Any, Any], *, precision: int = 3) -> str:
    parts = [
        f"{format_symbol(symbol)}:{format_prob_label(prob, precision=precision)}"
        for symbol, prob in sorted(dist.items(), key=str)
    ]
    return ", ".join(parts)


def format_belief(belief: Sequence[Any], *, rational: bool = True) -> str:
    """Compact simplex label for a mixed-state belief vector."""
    if rational:
        parts = [format_prob_label(value) for value in belief]
    else:
        parts = [format_prob(float(value)) for value in belief]
    return f"μ=({', '.join(parts)})"
