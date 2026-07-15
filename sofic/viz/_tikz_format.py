"""LaTeX / TikZ label formatting for Vaucanson-style machine figures."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from sofic.viz import _labels
from sofic.viz._rational import _TWO_DIGIT_RATIONAL_ATOL, two_digit_rational

_LATEX_SPECIAL = {
    "\\": r"\textbackslash{}",
    "{": r"\{",
    "}": r"\}",
    "#": r"\#",
    "$": r"\$",
    "%": r"\%",
    "&": r"\&",
    "_": r"\_",
    "^": r"\^{}",
    "~": r"\textasciitilde{}",
}


def latex_escape(text: str) -> str:
    """Escape plain text for use outside math mode in TikZ node labels."""
    escaped = []
    for char in text:
        escaped.append(_LATEX_SPECIAL.get(char, char))
    return "".join(escaped)


def _latex_arg(text: str) -> str:
    """Escape content for a LaTeX macro argument (no math mode)."""
    if text == "":
        return "{}"
    if any(char in text for char in "{}\\#%&_^~$"):
        return f"{{{latex_escape(text)}}}"
    return text


def format_state_latex(state: Any) -> str:
    return _labels.format_state(state, escape=latex_escape, epsilon=r"\varepsilon")


def format_state_tikz_node(state: Any) -> str:
    """TikZ-safe state label (commas in joint states must not parse as options)."""
    label = format_state_latex(state)
    return label.replace(",", "{,}")


def format_belief_tikz_node(belief: Sequence[Any]) -> str:
    """LaTeX-safe belief simplex label for TikZ node text (μ implied)."""
    parts = [format_prob_latex(value) for value in belief]
    return rf"$\scriptstyle({', '.join(parts)})$"


def format_symbol_latex(symbol: Any) -> str:
    return _labels.format_symbol(symbol, escape=_latex_arg, epsilon=r"\varepsilon")


def format_prob_latex(value: Any, *, precision: int = 3) -> str:
    """Format a probability for Vaucanson edge labels (float or sympy Expr)."""
    try:
        from sofic.generators.prob import is_symbolic, simplify_prob
    except ImportError:  # pragma: no cover
        is_symbolic = lambda _v: False  # noqa: E731
        simplify_prob = lambda v: v  # noqa: E731

    if is_symbolic(value):
        simplified = simplify_prob(value)
        try:
            import sympy as sp

            return latex_escape(sp.latex(simplified))
        except Exception:
            return latex_escape(str(simplified))

    numeric = float(value)
    if numeric <= 0.0:
        return "0"
    if numeric >= 1.0:
        return "1"
    if abs(numeric - 0.5) < _TWO_DIGIT_RATIONAL_ATOL:
        return r"\half"
    frac = two_digit_rational(numeric)
    if frac is not None:
        if frac.numerator == frac.denominator:
            return "1"
        return rf"\nicefrac{{{frac.numerator}}}{{{frac.denominator}}}"
    return latex_escape(f"{numeric:.{precision}g}")


def format_symbol_macro(symbol: Any) -> str:
    return rf"\Symbol{{{format_symbol_latex(symbol)}}}"


def format_edge_latex(symbol: Any, prob: Any) -> str:
    """Return ``$\\Edge{sym}{prob}$`` math content."""
    return rf"$\Edge{{{format_symbol_latex(symbol)}}}{{{format_prob_latex(prob)}}}$"


def format_tedge_latex(forward: Any, reverse: Any, prob: Any) -> str:
    """Return ``$\\TEdge{f}{r}{prob}$`` math content."""
    return (
        rf"$\TEdge{{{format_symbol_latex(forward)}}}"
        rf"{{{format_symbol_latex(reverse)}}}"
        rf"{{{format_prob_latex(prob)}}}$"
    )


def format_symbol_only_latex(symbol: Any) -> str:
    return rf"$\Symbol{{{format_symbol_latex(symbol)}}}$"


def format_transducer_edge_latex(input_symbol: Any, output_symbol: Any | None = None) -> str:
    if output_symbol is None:
        return format_symbol_only_latex(input_symbol)
    return (
        rf"$\Symbol{{{format_symbol_latex(input_symbol)}}}"
        rf"\mid\Symbol{{{format_symbol_latex(output_symbol)}}}$"
    )
