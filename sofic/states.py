"""Sequential capital-letter state labels: A, ..., Z, then AA, ..., ZZ, then AAA, ..., ZZZ, and so on."""

from __future__ import annotations

from collections.abc import Hashable, Iterable

_LETTERS = 26


def _label(index: int) -> str:
    """Bijective base-26 numeral of ``index + 1`` (spreadsheet column names)."""
    letters = []
    n = index + 1
    while n:
        n, r = divmod(n - 1, _LETTERS)
        letters.append(chr(ord("A") + r))
    return "".join(reversed(letters))


def sequential_labels(count: int, *, start: int = 0) -> tuple[str, ...]:
    """Return ``count`` state labels starting at the ``start``-th label (``A`` is 0, ``AA`` is 26)."""
    if count < 0:
        raise ValueError("count must be non-negative")
    if start < 0:
        raise ValueError("start must be non-negative")
    return tuple(_label(start + index) for index in range(count))


def letter_label_index(label: Hashable) -> int | None:
    """Index of a capital-letter label (``A`` is 0, ``Z`` 25, ``AA`` 26), else ``None``."""
    if not (isinstance(label, str) and label and all("A" <= c <= "Z" for c in label)):
        return None
    n = 0
    for c in label:
        n = n * _LETTERS + ord(c) - ord("A") + 1
    return n - 1


def next_sequential_label_index(existing: Iterable[Hashable]) -> int:
    """Index of the next unused capital-letter label after ``existing`` labels."""
    indices = [index for item in existing if (index := letter_label_index(item)) is not None]
    return max(indices, default=-1) + 1
