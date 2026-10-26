"""A zero-error feedback code for the trapdoor channel.

Permuter, Cuff, Van Roy and Weissman :cite:`Permuter2008` give a simple scheme
that communicates through the trapdoor channel with feedback at rates
approaching its feedback capacity :math:`\\log_2 \\varphi`, with no errors at
all. This module implements it as a :class:`dit.coding.FeedbackCode`, so it
needs a dit with feedback codes and is not imported by :mod:`sofic.generators`.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from dit.coding.feedback import FeedbackCode  # ty: ignore[unresolved-import]
from dit.exceptions import ditException

__all__ = ["TrapdoorCode", "trapdoor_code"]


def _counts(length: int) -> list[int]:
    """``counts[r]`` is the number of binary words of length ``r`` with no ``11``."""
    counts = [1, 2]
    while len(counts) <= length:
        counts.append(counts[-1] + counts[-2])
    return counts


def rll_encode(index: int, length: int) -> tuple[int, ...]:
    """The `index`-th (lexicographic) binary word of `length` with no two adjacent ``1`` s."""
    counts = _counts(length)
    if not 0 <= index < counts[length]:
        raise ValueError(f"index must lie in [0, {counts[length]})")
    word, previous = [], 0
    for i in range(length):
        remaining = length - i - 1
        if previous == 0 and index >= counts[remaining]:
            index -= counts[remaining]
            previous = 1
        else:
            previous = 0
        word.append(previous)
    return tuple(word)


def rll_decode(word: Sequence[int]) -> int:
    """The lexicographic index of a binary word with no two adjacent ``1`` s; inverse of :func:`rll_encode`."""
    counts = _counts(len(word))
    index = 0
    for i, bit in enumerate(word):
        if bit:
            index += counts[len(word) - i - 1]
    return index


class TrapdoorCode(FeedbackCode):
    """The capacity-achieving zero-error feedback code for the trapdoor channel :cite:`Permuter2008`.

    The transmitter works in *actions*: ``0`` sends a ball of the same type as
    the one in the channel, ``1`` the opposite type. It knows the channel's
    state from the feedback. A message is mapped by enumerative coding to an
    action word of length ``N`` that ends in ``0`` and has no two ``1`` s in a
    row; there are about :math:`\\varphi^N` such words, so the rate approaches
    :math:`\\log_2 \\varphi`. The receiver recovers the actions backward from
    the last one, using only whether consecutive outputs differ.

    The initial state is unknown, so each transmission first *flushes* the
    channel with the inputs ``0, 1, 0, 1, ...`` until an output differs from
    its input; from then on both ends know the state. Flushing takes 3.5 uses
    on average from a uniform state, and is counted in the rate.

    Parameters
    ----------
    k : int
        The number of message bits.
    """

    def __init__(self, k: int) -> None:
        super().__init__(k)
        length = 0
        while _counts(length)[length] < 2**k:
            length += 1
        self.length = length + 1

    def __repr__(self) -> str:
        return f"TrapdoorCode(k={self.k}, N={self.length})"

    def start(self, channel: Any) -> tuple[int, ...]:
        """The outputs received so far: none."""
        if tuple(channel.inputs) != (0, 1) or tuple(channel.outputs) != (0, 1):
            raise ditException("The trapdoor code needs a channel with inputs and outputs {0, 1}.")
        return ()

    @staticmethod
    def _flushed(received: tuple[int, ...]) -> int | None:
        """The number of flushing uses, or ``None`` if the channel is still being flushed."""
        for t, y in enumerate(received):
            if y != t % 2:
                return t + 1
        return None

    def _actions(self, message: Sequence[int]) -> tuple[int, ...]:
        index = int("".join(map(str, message)), 2)
        return (*rll_encode(index, self.length - 1), 0)

    def next_input(self, state: tuple[int, ...], message: Sequence[int]) -> int | None:
        """The next flushing input, or the next action applied to the current state."""
        flushed = self._flushed(state)
        if flushed is None:
            return len(state) % 2
        data = state[flushed:]
        if len(data) == self.length:
            return None
        actions = self._actions(message)
        ball = (flushed - 1) % 2
        for action, y in zip(actions, data, strict=False):
            x = action ^ ball
            ball ^= x ^ y
        return actions[len(data)] ^ ball

    def update(self, state: tuple[int, ...], y: int) -> tuple[int, ...]:
        """Append the output `y`."""
        return (*state, y)

    def decode(self, state: tuple[int, ...]) -> tuple[int, ...]:
        """Recover the actions backward from the last, then invert the enumerative code."""
        flushed = self._flushed(state)
        data = state[flushed:] if flushed is not None else ()
        if len(data) < self.length:
            return (0,) * self.k
        actions = [0] * self.length
        for t in range(self.length - 2, -1, -1):
            differs = data[t + 1] ^ data[t]
            actions[t] = int(differs and not actions[t + 1])
        index = rll_decode(actions[:-1])
        if index >= 2**self.k:
            return (0,) * self.k
        return tuple(int(b) for b in format(index, f"0{self.k}b"))


def trapdoor_code(k: int) -> TrapdoorCode:
    """A :class:`TrapdoorCode` for `k` message bits."""
    return TrapdoorCode(k)
