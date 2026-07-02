"""Backend-agnostic skeletons for state/symbol label formatting.

The graphviz and LaTeX backends share the same structural recursion for
formatting states (tuples recurse, frozensets render as sorted ``{...}``,
``EPSILON`` becomes a glyph) and symbols; they differ only in the leaf escape
function and the epsilon glyph. These helpers capture that shared shape.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from pensive.graph import EPSILON


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
