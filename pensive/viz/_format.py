"""Label formatting helpers for Graphviz output."""

from __future__ import annotations

from collections.abc import Mapping
from fractions import Fraction
from typing import Any, Sequence

from pensive.graph import EPSILON

_TWO_DIGIT_RATIONAL_ATOL = 1e-9
_MAX_TWO_DIGIT_RATIONAL = 99


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
    if isinstance(state, tuple):
        inner = ", ".join(format_state(part) for part in state)
        return f"({inner})"
    if state is EPSILON:
        return "ε"
    return dot_escape(str(state))


def format_symbol(symbol: Any) -> str:
    if symbol is EPSILON:
        return "ε"
    return dot_escape(str(symbol))


def format_prob(value: float, *, precision: int = 3) -> str:
    return dot_escape(f"{value:.{precision}g}")


def _two_digit_rational(value: float, *, atol: float = _TWO_DIGIT_RATIONAL_ATOL) -> Fraction | None:
    """Return a reduced rational with 1 <= p, q <= 99 when ``value`` matches exactly."""
    if value <= 0.0 or value >= 1.0:
        return None
    frac = Fraction(value).limit_denominator(_MAX_TWO_DIGIT_RATIONAL)
    if abs(float(frac) - value) >= atol:
        return None
    if not (1 <= frac.numerator <= _MAX_TWO_DIGIT_RATIONAL and 1 <= frac.denominator <= _MAX_TWO_DIGIT_RATIONAL):
        return None
    return frac


def format_prob_rational(value: float, *, precision: int = 3) -> str:
    """Format a probability as p/q when exact with two-digit numerator and denominator."""
    if value <= 0.0:
        return "0"
    if value >= 1.0:
        return "1"
    frac = _two_digit_rational(value)
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
