"""Sequential capital-letter state labels (A, B, C, ...)."""

from __future__ import annotations

from collections.abc import Hashable, Iterable


def sequential_labels(count: int, *, start: int = 0) -> tuple[str, ...]:
    """Return ``count`` state labels starting at ``chr(ord('A') + start)``."""
    if count < 0:
        raise ValueError("count must be non-negative")
    if start < 0 or start + count > 26:
        raise ValueError("sequential_labels supports at most 26 states A–Z")
    return tuple(chr(ord("A") + start + index) for index in range(count))


def letter_label_index(label: Hashable) -> int | None:
    """Map a single capital letter ``A``–``Z`` to 0–25, else ``None``."""
    if isinstance(label, str) and len(label) == 1 and "A" <= label <= "Z":
        return ord(label) - ord("A")
    return None


def next_sequential_label_index(existing: Iterable[Hashable]) -> int:
    """Index of the next unused capital letter after ``existing`` labels."""
    indices = [index for item in existing if (index := letter_label_index(item)) is not None]
    return max(indices, default=-1) + 1
