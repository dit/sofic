"""Sliding block codes (factor maps) between subshifts.

A sliding block code is the symbolic-dynamics form of a transducer: a map
``Phi`` between shift spaces induced by a local rule on a window of ``memory + 1
+ anticipation`` consecutive symbols (the Curtis-Hedlund-Lyndon theorem
characterizes exactly the shift-commuting continuous maps this way). See
Lind & Marcus, *An Introduction to Symbolic Dynamics and Coding* (1995), ch. 1
and 6.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from itertools import product
from typing import Any

from sofic.graph import ATTR_OUTPUT, ATTR_SYMBOL, TransitionGraph
from sofic.shifts.sofic import SoficShift


class SlidingBlockCode:
    """A block map ``Phi`` with given memory and anticipation.

    ``block_map`` sends each length-``(memory + 1 + anticipation)`` input window to
    a single output symbol. Applying ``Phi`` to a point slides this window,
    emitting one output symbol per position.

    Examples
    --------
    >>> from sofic.shifts.sliding_block_code import SlidingBlockCode
    >>> nor = SlidingBlockCode(
    ...     {("0", "0"): "1", ("0", "1"): "0", ("1", "0"): "0", ("1", "1"): "0"},
    ...     memory=1,
    ...     anticipation=0,
    ... )
    >>> nor.apply_word(["0", "0", "1", "0"])
    ('1', '0', '0')
    """

    def __init__(
        self,
        block_map: Mapping[Sequence[Any], Any],
        *,
        memory: int = 0,
        anticipation: int = 0,
        input_alphabet: Iterable[Any] | None = None,
        output_alphabet: Iterable[Any] | None = None,
    ) -> None:
        if memory < 0 or anticipation < 0:
            raise ValueError("memory and anticipation must be non-negative")
        self.memory = memory
        self.anticipation = anticipation
        self.window = memory + anticipation + 1
        self.block_map: dict[tuple[Any, ...], Any] = {tuple(key): value for key, value in block_map.items()}
        for key in self.block_map:
            if len(key) != self.window:
                raise ValueError(f"block_map key {key!r} has length {len(key)}, expected window {self.window}")
        self.input_alphabet = frozenset(
            input_alphabet if input_alphabet is not None else {symbol for key in self.block_map for symbol in key}
        )
        self.output_alphabet = frozenset(output_alphabet if output_alphabet is not None else self.block_map.values())

    def __repr__(self) -> str:
        return f"SlidingBlockCode(window={self.window}, memory={self.memory}, anticipation={self.anticipation})"

    def apply_word(self, word: Sequence[Any]) -> tuple[Any, ...]:
        """Return ``Phi`` applied to a finite word (shrinks by ``window - 1``)."""
        word = tuple(word)
        if len(word) < self.window:
            return ()
        return tuple(self.block_map[word[i : i + self.window]] for i in range(len(word) - self.window + 1))

    def apply(self, shift: Any) -> SoficShift:
        """Return the image subshift ``Phi(shift)`` as a sofic presentation.

        Vertices pair a vertex ``q`` of ``shift``'s presentation with the last
        ``window - 1`` symbols read along a path into ``q``; each edge of
        ``shift`` out of ``q`` reading ``a`` emits ``Phi(context + a)``. Tracking
        the presentation vertex keeps every constraint of ``shift``, not just
        those visible in ``window``-blocks. Raises :class:`ValueError` when a
        ``window``-block of ``shift`` is missing from ``block_map``.
        """
        outgoing: dict[Any, list[tuple[Any, Any]]] = {state: [] for state in shift.states()}
        for transition in shift.transitions():
            symbol = transition.data.get(ATTR_SYMBOL)
            if symbol is not None:
                outgoing[transition.source].append((symbol, transition.target))

        context_length = self.window - 1
        frontier = {(state, ()) for state in outgoing}
        for _ in range(context_length):
            frontier = {
                (target, (*context, symbol)) for state, context in frontier for symbol, target in outgoing[state]
            }

        image = SoficShift(symbol_alphabet=frozenset(self.output_alphabet))
        seen = set(frontier)
        queue = list(frontier)
        used_outputs: set[Any] = set()
        while queue:
            state, context = queue.pop()
            image.graph.add_state((state, context))
            for symbol, target in outgoing[state]:
                block = (*context, symbol)
                if block not in self.block_map:
                    raise ValueError(f"block {block!r} of the shift is not in block_map")
                successor = (target, block[1:])
                image.add_transition((state, context), successor, self.block_map[block])
                used_outputs.add(self.block_map[block])
                if successor not in seen:
                    seen.add(successor)
                    queue.append(successor)
        image.symbol_alphabet = frozenset(used_outputs)
        return image.trim_transient()

    def is_right_resolving(self, shift: Any | None = None) -> bool:
        """Return whether the induced image presentation is right-resolving."""
        source = shift if shift is not None else full_shift(self.input_alphabet)
        return self.apply(source).is_unifilar()

    def compose(self, other: SlidingBlockCode) -> SlidingBlockCode:
        """Return ``other`` applied after ``self`` (function composition ``other ∘ self``)."""
        window = self.window + other.window - 1
        composed: dict[tuple[Any, ...], Any] = {}
        for word in product(sorted(self.input_alphabet, key=repr), repeat=window):
            middle = self.apply_word(word)
            if len(middle) != other.window:
                continue
            output = other.block_map.get(tuple(middle))
            if output is None:
                continue
            composed[word] = output
        return SlidingBlockCode(
            composed,
            memory=self.memory + other.memory,
            anticipation=self.anticipation + other.anticipation,
            input_alphabet=self.input_alphabet,
            output_alphabet=other.output_alphabet,
        )

    def to_transducer(self) -> Any:
        """Return a Mealy machine realizing this code (output delayed by anticipation)."""
        from sofic.automata.transducers import MealyMachine

        graph = TransitionGraph()
        context_length = self.window - 1
        contexts = list(product(sorted(self.input_alphabet, key=repr), repeat=context_length))
        for context in contexts:
            graph.add_state(context)
        used_outputs: set[Any] = set()
        for context in contexts:
            for symbol in sorted(self.input_alphabet, key=repr):
                window = (*context, symbol)
                output = self.block_map.get(window)
                if output is None:
                    continue
                target = window[1:]
                graph.add_transition(context, target, **{ATTR_SYMBOL: symbol, ATTR_OUTPUT: output})
                used_outputs.add(output)
        machine = MealyMachine(
            input_alphabet=frozenset(self.input_alphabet),
            output_alphabet=frozenset(used_outputs),
            initial_states=frozenset(contexts),
            graph=graph,
        )
        machine.validate()
        return machine


def full_shift(alphabet: Iterable[Any]) -> SoficShift:
    """Return the full shift over ``alphabet`` as a one-state sofic presentation."""
    symbols = frozenset(alphabet)
    shift = SoficShift(symbol_alphabet=symbols)
    shift.graph.add_state("*")
    for symbol in symbols:
        shift.add_transition("*", "*", symbol)
    return shift
