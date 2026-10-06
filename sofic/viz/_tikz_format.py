"""LaTeX / TikZ label formatting for Vaucanson-style machine figures."""

from __future__ import annotations

from collections.abc import Sequence
from fractions import Fraction
from typing import Any

from sofic.viz import _labels

_HALF = Fraction(1, 2)

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


def _latex_fraction(frac: Fraction) -> str:
    if frac == _HALF:
        return r"\half"
    return rf"\nicefrac{{{frac.numerator}}}{{{frac.denominator}}}"


def _latex_sympy(expr: Any) -> str:
    import sympy as sp

    return sp.latex(expr)


def format_prob_latex(value: Any, *, precision: int = 3) -> str:
    """Format a probability for Vaucanson edge labels (float or sympy Expr)."""
    return _labels.format_prob(
        value, escape=latex_escape, fraction=_latex_fraction, symbolic=_latex_sympy, precision=precision
    )


def format_edge_latex(symbol: Any, prob: Any) -> str:
    """Return ``$\\Edge{sym}{prob}$`` math content."""
    return rf"$\Edge{{{format_symbol_latex(symbol)}}}{{{format_prob_latex(prob)}}}$"
