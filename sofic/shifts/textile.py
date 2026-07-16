"""Textile systems (Nasu 1995): the machine form of a sliding block code.

A textile system encodes a factor map between subshifts as a pair of labelings
over a shared edge graph -- one reading the domain (input) symbols, one the range
(output) symbols. This is Nasu's formalization (*Textile Systems for Endomorphisms
and Automorphisms of the Shift*, Memoirs AMS 546, 1995); operationally it is a
topological transducer whose input labeling, when right-resolving, induces a
:class:`~sofic.shifts.sliding_block_code.SlidingBlockCode` from the input shift to
the output shift.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Hashable
from typing import Any

from sofic.graph import ATTR_OUTPUT, ATTR_SYMBOL, EPSILON
from sofic.shifts.sliding_block_code import SlidingBlockCode
from sofic.shifts.sofic import SoficShift


class TextileSystem:
    """A shared edge graph with paired input/output labels (a topological transducer)."""

    def __init__(self, transducer: Any) -> None:
        self._transducer = transducer

    @classmethod
    def from_transducer(cls, transducer: Any) -> TextileSystem:
        """Wrap a (topological) Mealy machine as a textile system."""
        return cls(transducer.copy())

    def to_transducer(self) -> Any:
        """Return the underlying Mealy machine."""
        return self._transducer.copy()

    def to_sofic_relation(self) -> Any:
        """Return the product-alphabet subshift of paired labels."""
        from sofic.shifts.sofic_relation import SoficRelation

        return SoficRelation.from_transducer(self._transducer)

    def input_shift(self) -> SoficShift:
        """Return the input subshift (``p`` labeling)."""
        return self.to_sofic_relation().input_shift()

    def output_shift(self) -> SoficShift:
        """Return the output subshift (``q`` labeling)."""
        return self.to_sofic_relation().output_shift()

    def induced_code(self, *, max_window: int = 4) -> SlidingBlockCode:
        """Return the induced sliding block code (memory only), if it has finite window.

        Searches windows of length ``1 .. max_window``; for each, the code is
        well-defined when every readable input window determines a unique output
        on its final position. Raises :class:`ValueError` if the input labeling is
        not right-resolving with a finite window up to ``max_window``.
        """
        transducer = self._transducer
        input_alphabet = frozenset(transducer.input_alphabet) or {
            transition.data.get(ATTR_SYMBOL) for transition in transducer.transitions()
        }
        for window in range(1, max_window + 1):
            block_map = self._window_block_map(transducer, window)
            if block_map is not None:
                return SlidingBlockCode(
                    block_map,
                    memory=window - 1,
                    anticipation=0,
                    input_alphabet=input_alphabet,
                    output_alphabet={value for value in block_map.values()},
                )
        raise ValueError(f"no finite-memory induced code up to window {max_window}")

    def _window_block_map(self, transducer: Any, window: int) -> dict[tuple[Any, ...], Any] | None:
        # For each readable input word of `window` symbols, collect the set of
        # possible outputs on the final edge across all paths reading that word.
        outputs: dict[tuple[Any, ...], set[Any]] = {}
        start_states = list(transducer.states())
        for start in start_states:
            queue: deque[tuple[Hashable, tuple[Any, ...], Any]] = deque([(start, (), None)])
            while queue:
                state, word, last_output = queue.popleft()
                if len(word) == window:
                    outputs.setdefault(word, set()).add(last_output)
                    continue
                for transition in transducer.graph.out_transitions(state):
                    symbol = transition.data.get(ATTR_SYMBOL)
                    if symbol is None or symbol is EPSILON:
                        continue
                    emitted = transition.data.get(ATTR_OUTPUT)
                    queue.append((transition.target, (*word, symbol), emitted))
        if not outputs:
            return None
        block_map: dict[tuple[Any, ...], Any] = {}
        for word, possible in outputs.items():
            if len(possible) != 1:
                return None
            value = next(iter(possible))
            if value is None or value is EPSILON:
                return None
            block_map[word] = value
        return block_map
