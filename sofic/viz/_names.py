"""Shared node-identifier helper for viz backends."""

from __future__ import annotations

import re
from collections.abc import Hashable, Iterable


def node_name(state: Hashable) -> str:
    """Return a readable Graphviz-compatible identifier for ``state``.

    Not injective (``'q_1'`` and ``'q-1'`` both give ``q_1``); use
    :func:`node_names` to name every state of one model.
    """
    text = repr(state)
    ident = re.sub(r"[^A-Za-z0-9_]+", "_", text).strip("_")
    if not ident:
        ident = "state"
    if ident[0].isdigit():
        ident = f"s_{ident}"
    return ident


def node_names(states: Iterable[Hashable]) -> dict[Hashable, str]:
    """Return distinct identifiers for ``states``.

    Each state gets :func:`node_name` unless an earlier state (in ``repr``
    order) already took it; later ones get the first free ``_2``, ``_3``, ...
    suffix, so the mapping is injective and depends only on the set of states.
    """
    ordered = sorted(set(states), key=repr)
    bases = {state: node_name(state) for state in ordered}
    used = set(bases.values())
    claimed: set[str] = set()
    names: dict[Hashable, str] = {}
    for state in ordered:
        base = bases[state]
        if base not in claimed:
            claimed.add(base)
            names[state] = base
            continue
        index = 2
        while f"{base}_{index}" in used:
            index += 1
        name = f"{base}_{index}"
        used.add(name)
        names[state] = name
    return names
