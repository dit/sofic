"""Backend-agnostic skeletons for state/symbol/probability label formatting.

The graphviz and LaTeX backends share the same structural recursion for
formatting states (tuples recurse, frozensets render as sorted ``{...}``,
``EPSILON`` becomes a glyph), symbols, and probabilities (symbolic
expressions, clamped ``0`` / ``1``, exact two-digit fractions, else
``precision`` significant digits); they differ only in the leaf escape
function, the epsilon glyph, and how fractions / sympy expressions are
spelled. These helpers capture that shared shape.
"""

from __future__ import annotations

from collections.abc import Callable
from fractions import Fraction
from typing import Any

from sofic.graph import EPSILON

_TWO_DIGIT_RATIONAL_ATOL = 1e-9
_MAX_TWO_DIGIT_RATIONAL = 99


def two_digit_rational(value: float, *, atol: float = _TWO_DIGIT_RATIONAL_ATOL) -> Fraction | None:
    """Return a reduced rational with 1 <= p, q <= 99 when ``value`` matches exactly."""
    if value <= 0.0 or value >= 1.0:
        return None
    frac = Fraction(value).limit_denominator(_MAX_TWO_DIGIT_RATIONAL)
    if abs(float(frac) - value) >= atol:
        return None
    if not (1 <= frac.numerator <= _MAX_TWO_DIGIT_RATIONAL and 1 <= frac.denominator <= _MAX_TWO_DIGIT_RATIONAL):
        return None
    return frac


def format_state(state: Any, *, escape: Callable[[str], str], epsilon: str) -> str:
    """Format a (possibly nested) state using ``escape`` for leaves."""
    if isinstance(state, tuple):
        inner = ", ".join(format_state(part, escape=escape, epsilon=epsilon) for part in state)
        return f"({inner})"
    if isinstance(state, frozenset):
        inner = ", ".join(sorted(format_state(part, escape=escape, epsilon=epsilon) for part in state))
        return rf"\{{{inner}\}}"
    if state is EPSILON:
        return epsilon
    return escape(str(state))


def format_symbol(symbol: Any, *, escape: Callable[[str], str], epsilon: str) -> str:
    """Format a symbol using ``escape`` for the leaf, ``epsilon`` for EPSILON."""
    if symbol is EPSILON:
        return epsilon
    return escape(str(symbol))


def format_prob(
    value: Any,
    *,
    escape: Callable[[str], str],
    fraction: Callable[[Fraction], str],
    symbolic: Callable[[Any], str],
    precision: int = 3,
) -> str:
    """Format a probability (float or sympy Expr).

    ``symbolic`` spells a simplified sympy expression and ``fraction`` an exact
    two-digit rational, each returning backend-ready text. Decimals and the
    ``str`` fallback (used when ``symbolic`` raises) pass through ``escape``.
    """
    from sofic.generators.prob import is_symbolic, simplify_prob

    if is_symbolic(value):
        simplified = simplify_prob(value)
        try:
            return symbolic(simplified)
        except Exception:
            return escape(str(simplified))

    numeric = float(value)
    if numeric <= 0.0:
        return "0"
    if numeric >= 1.0:
        return "1"
    frac = two_digit_rational(numeric)
    if frac is not None:
        if frac.numerator == frac.denominator:
            return "1"
        return fraction(frac)
    return escape(f"{numeric:.{precision}g}")
