"""Deterministic and weighted finite-state transducers.

The finite-automata reading of the transducer stack (Mohri, *Weighted Automata
Algorithms*, in *Handbook of Weighted Automata*, 2009; Roche & Schabes, *Finite-
State Language Processing*, 1997):

* :class:`SubsequentialTransducer` -- an input-deterministic (sequential)
  transducer augmented with a per-state final-output string, the classic
  subsequential transducer of Schutzenberger/Mohri.
* :class:`WeightedFiniteStateTransducer` -- a transducer whose edges carry
  weights in a semiring (probability or tropical), generalizing the stochastic
  Mealy machine used by the ε-transducer.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Hashable, Sequence
from typing import Any

from sofic.automata.transducers import MealyMachine
from sofic.exceptions import SoficValidationError
from sofic.graph import ATTR_OUTPUT, ATTR_PROB, ATTR_SYMBOL, EPSILON

SEMIRINGS = ("probability", "tropical", "log")


class SubsequentialTransducer(MealyMachine):
    """Input-deterministic transducer with a per-state final output string.

    A subsequential transducer reads its input left to right along the unique
    matching path and, on reaching the end of input, appends the ``final_output``
    string of the state it stops in.

    Examples
    --------
    >>> from sofic.automata.subsequential import SubsequentialTransducer
    >>> t = SubsequentialTransducer(
    ...     input_alphabet=frozenset("ab"),
    ...     output_alphabet=frozenset("xy"),
    ...     initial_states=frozenset({"q0"}),
    ...     final_output={"q0": ("y",)},
    ... )
    >>> t.graph.add_state("q0")
    >>> _ = t.add_transition("q0", "q0", "a", "x")
    >>> sorted(t.transduce("aa"))
    [('x', 'x', 'y')]
    """

    final_output: dict[Hashable, tuple[Any, ...]]

    def __init__(
        self,
        input_alphabet: frozenset[Any] | None = None,
        output_alphabet: frozenset[Any] | None = None,
        initial_states: frozenset[Hashable] | None = None,
        final_output: dict[Hashable, Sequence[Any]] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            input_alphabet=input_alphabet,
            output_alphabet=output_alphabet,
            initial_states=initial_states,
            **kwargs,
        )
        self.final_output = {state: tuple(word) for state, word in (final_output or {}).items()}

    def validate(self) -> None:
        super().validate()
        from sofic.properties import is_sequential_transducer

        self._require(is_sequential_transducer(self), "subsequential transducer must be input-deterministic")
        for state, word in self.final_output.items():
            self._require(self.graph.has_state(state), f"unknown final-output state {state!r}")
            for symbol in word:
                if symbol is EPSILON:
                    continue
                self._require(symbol in self.output_alphabet, f"final output {symbol!r} not in output alphabet")

    def is_subsequential(self) -> bool:
        """Return whether this is a valid subsequential transducer."""
        from sofic.properties import is_subsequential_transducer

        return is_subsequential_transducer(self)

    def transduce(self, word: Sequence[Any]) -> set[tuple[Any, ...]]:
        results: set[tuple[Any, ...]] = set()
        for state, output in self._walk(word):
            results.add(output + self.final_output.get(state, ()))
        return results

    def _walk(self, word: Sequence[Any]) -> set[tuple[Hashable, tuple[Any, ...]]]:
        configs: set[tuple[Hashable, tuple[Any, ...]]] = {(state, ()) for state in self.initial_states}
        for symbol in word:
            nxt: set[tuple[Hashable, tuple[Any, ...]]] = set()
            for state, output in configs:
                for transition in self.graph.out_transitions(state):
                    if transition.data.get(ATTR_SYMBOL) != symbol:
                        continue
                    emitted = transition.data.get(ATTR_OUTPUT)
                    extended = output + ((emitted,) if emitted is not None and emitted is not EPSILON else ())
                    nxt.add((transition.target, extended))
            configs = nxt
        return configs


class WeightedFiniteStateTransducer(MealyMachine):
    """Finite-state transducer with edge weights in a semiring.

    The weight of an ``(input, output)`` pair is the semiring sum over all
    matching paths of the semiring product of their edge weights. The
    ``probability`` semiring uses ``(+, x, 0, 1)`` (edge weight = probability);
    the ``tropical`` semiring uses ``(min, +, +inf, 0)`` (edge weight = cost),
    yielding shortest-path / Viterbi weights.
    """

    semiring: str

    def __init__(
        self,
        input_alphabet: frozenset[Any] | None = None,
        output_alphabet: frozenset[Any] | None = None,
        initial_states: frozenset[Hashable] | None = None,
        semiring: str = "probability",
        **kwargs: Any,
    ) -> None:
        super().__init__(
            input_alphabet=input_alphabet,
            output_alphabet=output_alphabet,
            initial_states=initial_states,
            **kwargs,
        )
        self.semiring = semiring

    def validate(self) -> None:
        super().validate()
        if self.semiring not in SEMIRINGS:
            raise SoficValidationError(f"unknown semiring {self.semiring!r}; expected one of {SEMIRINGS}")

    @classmethod
    def from_transducer(
        cls,
        transducer: MealyMachine,
        *,
        semiring: str = "probability",
    ) -> WeightedFiniteStateTransducer:
        """Build a WFST from a (probability-weighted) Mealy machine.

        For the ``tropical`` and ``log`` semirings each edge probability ``p`` is
        converted to the cost ``-log p``.
        """
        graph = transducer.graph.copy()
        if semiring in {"tropical", "log"}:
            for _s, _t, data in graph.nx.edges(data=True):
                prob = float(data.get(ATTR_PROB, 1.0))
                data[ATTR_PROB] = math.inf if prob <= 0.0 else -math.log(prob)
        result = cls(
            input_alphabet=frozenset(transducer.input_alphabet),
            output_alphabet=frozenset(transducer.output_alphabet),
            initial_states=frozenset(transducer.initial_states),
            semiring=semiring,
            graph=graph,
        )
        result.validate()
        return result

    def weight(self, inputs: Sequence[Any], outputs: Sequence[Any]) -> float:
        """Return the total semiring weight of reading ``inputs`` emitting ``outputs``."""
        if len(inputs) != len(outputs):
            return self._zero()
        frontier: dict[Hashable, float] = {state: self._one() for state in self.initial_states}
        for input_symbol, output_symbol in zip(inputs, outputs, strict=True):
            nxt: dict[Hashable, float] = defaultdict(self._zero)
            for state, accumulated in frontier.items():
                for transition in self.graph.out_transitions(state):
                    if transition.data.get(ATTR_SYMBOL) != input_symbol:
                        continue
                    if transition.data.get(ATTR_OUTPUT) != output_symbol:
                        continue
                    edge = float(transition.data.get(ATTR_PROB, self._one()))
                    nxt[transition.target] = self._plus(nxt[transition.target], self._times(accumulated, edge))
            frontier = dict(nxt)
            if not frontier:
                return self._zero()
        total = self._zero()
        for value in frontier.values():
            total = self._plus(total, value)
        return total

    def _one(self) -> float:
        return 0.0 if self.semiring in {"tropical", "log"} else 1.0

    def _zero(self) -> float:
        return math.inf if self.semiring in {"tropical", "log"} else 0.0

    def _times(self, left: float, right: float) -> float:
        return left + right if self.semiring in {"tropical", "log"} else left * right

    def _plus(self, left: float, right: float) -> float:
        return min(left, right) if self.semiring in {"tropical", "log"} else left + right
