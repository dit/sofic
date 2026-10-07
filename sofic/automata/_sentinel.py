"""Identity-compared state labels that survive ``copy`` and ``deepcopy``."""

from __future__ import annotations

from typing import Any


class Sentinel:
    """A unique marker that copies as itself.

    Automaton copies deep-copy their initial and accepting state sets, which
    would clone a bare ``object()`` label into a different, unequal state.
    """

    __slots__ = ("_name",)

    def __init__(self, name: str) -> None:
        self._name = name

    def __repr__(self) -> str:
        return f"<{self._name}>"

    def __copy__(self) -> Sentinel:
        return self

    def __deepcopy__(self, memo: dict[int, Any]) -> Sentinel:
        return self
