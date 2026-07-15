"""Label formatting helpers for Graphviz output."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from sofic.viz import _labels
from sofic.viz._rational import two_digit_rational


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


def format_prob_label(value: Any, *, precision: int = 3) -> str:
    """Format a probability for Graphviz edge/π labels (float or sympy Expr)."""
    try:
        from sofic.generators.prob import is_symbolic, simplify_prob
    except ImportError:  # pragma: no cover

        def is_symbolic(_v: Any) -> bool:
            return False

        def simplify_prob(v: Any) -> Any:
            return v

    if is_symbolic(value):
        simplified = simplify_prob(value)
        try:
            import sympy as sp

            text = sp.sstr(simplified)
        except Exception:
            text = str(simplified)
        return dot_escape(text)
    return format_prob_rational(float(value), precision=precision)


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
