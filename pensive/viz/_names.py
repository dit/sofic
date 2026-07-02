"""Shared node-identifier helper for viz backends."""

from __future__ import annotations

import re
from collections.abc import Hashable


def node_name(state: Hashable) -> str:
    """Return a Graphviz-compatible node identifier for ``state``."""
    text = repr(state)
    ident = re.sub(r"[^A-Za-z0-9_]+", "_", text).strip("_")
    if not ident:
        ident = "state"
    if ident[0].isdigit():
        ident = f"s_{ident}"
    return ident
