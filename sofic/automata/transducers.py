"""Finite-state transducers."""

from __future__ import annotations

from abc import abstractmethod
from collections import defaultdict
from collections.abc import Hashable, Mapping, Sequence
from typing import Any

import numpy as np

from sofic.base import StateMachine
from sofic.exceptions import StochasticValidationError
from sofic.graph import ATTR_EMISSION, ATTR_OUTPUT, ATTR_PROB, ATTR_SYMBOL, EPSILON

ERROR_SYMBOL = "?"
ERROR_STATE = "?"


class Transducer(StateMachine):
    """Non-probabilistic input-to-output machine."""

    input_alphabet: frozenset[Any]
    output_alphabet: frozenset[Any]
    initial_states: frozenset[Hashable]

    def __init__(
        self,
        input_alphabet: frozenset[Any] | None = None,
        output_alphabet: frozenset[Any] | None = None,
        initial_states: frozenset[Hashable] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.input_alphabet = input_alphabet if input_alphabet is not None else frozenset()
        self.output_alphabet = output_alphabet if output_alphabet is not None else frozenset()
        self.initial_states = initial_states if initial_states is not None else frozenset()

    def validate(self) -> None:
        for state in self.initial_states:
            self._require(self.graph.has_state(state), f"missing initial state {state!r}")

    def is_deterministic(self) -> bool:
        """Return whether each state has at most one transition per input symbol."""
        from sofic.properties import is_deterministic_transducer

        return is_deterministic_transducer(self)

    def alphabets(
        self,
        *,
        node: bool = False,
    ) -> tuple[frozenset[Any], frozenset[Any]] | tuple[dict[Hashable, frozenset[Any]], dict[Hashable, frozenset[Any]]]:
        """Return effective input and output alphabets.

        ``EPSILON`` transitions are operationally meaningful but are not part
        of the external alphabets returned here.
        """
        input_by_state: dict[Hashable, set[Any]] = {state: set() for state in self.states()}
        output_by_state: dict[Hashable, set[Any]] = {state: set() for state in self.states()}
        inputs: set[Any] = set(self.input_alphabet)
        outputs: set[Any] = set(self.output_alphabet)
        for transition in self.transitions():
            symbol = transition.data.get(ATTR_SYMBOL)
            if symbol is not None and symbol is not EPSILON:
                inputs.add(symbol)
                input_by_state.setdefault(transition.source, set()).add(symbol)
            output = transition.data.get(ATTR_OUTPUT)
            if output is not None and output is not EPSILON:
                outputs.add(output)
                output_by_state.setdefault(transition.source, set()).add(output)
        if node:
            return (
                {state: frozenset(symbols) for state, symbols in input_by_state.items()},
                {state: frozenset(symbols) for state, symbols in output_by_state.items()},
            )
        return frozenset(inputs), frozenset(outputs)

    def is_complete(self, alphabet: frozenset[Any] | None = None) -> bool:
        """Return whether every state has an outgoing edge for each input symbol."""
        inputs = alphabet if alphabet is not None else self.alphabets()[0]
        input_by_state, _ = self.alphabets(node=True)
        return all(inputs <= symbols for symbols in input_by_state.values())

    @abstractmethod
    def transduce(self, word: Sequence[Any]) -> set[tuple[Any, ...]]:
        """Return possible output sequences for ``word``."""


class MealyMachine(Transducer):
    """Output symbols on transitions."""

    def add_transition(
        self,
        source: Hashable,
        target: Hashable,
        symbol: Any,
        output: Any | None = None,
        *,
        prob: float | None = None,
        **attrs: Any,
    ) -> int:
        """Add a transition with input ``symbol`` and optional output.

        Use :data:`sofic.graph.EPSILON` for empty input or empty output.
        Omitting ``prob`` keeps the transition topological; probability-aware
        helpers treat missing probabilities as weight ``1``.
        """
        data = {ATTR_SYMBOL: symbol, **attrs}
        if output is not None:
            data[ATTR_OUTPUT] = output
        if prob is not None:
            data[ATTR_PROB] = float(prob)
        return self.graph.add_transition(source, target, **data)

    def transduce(self, word: Sequence[Any]) -> set[tuple[Any, ...]]:
        from sofic.automata.transducer_simulation import transduce_mealy

        return transduce_mealy(self, word)

    def complete(
        self,
        alphabet: frozenset[Any] | None = None,
        *,
        reject: Hashable = ERROR_STATE,
        error_output: Any = ERROR_SYMBOL,
        copy: bool = True,
    ) -> MealyMachine:
        """Return a complete transducer by adding reject/error transitions."""
        result = self.copy() if copy else self
        symbols = alphabet if alphabet is not None else result.alphabets()[0]
        if not symbols:
            return result

        result.graph.add_state(reject)
        for state in list(result.states()):
            outgoing = {
                transition.data.get(ATTR_SYMBOL)
                for transition in result.graph.out_transitions(state)
                if transition.data.get(ATTR_SYMBOL) is not EPSILON
            }
            for symbol in symbols - outgoing:
                result.add_transition(state, reject, symbol, error_output, prob=1.0)

        for symbol in symbols:
            if not any(
                transition.data.get(ATTR_SYMBOL) == symbol for transition in result.graph.out_transitions(reject)
            ):
                result.add_transition(reject, reject, symbol, error_output, prob=1.0)

        result.input_alphabet = result.input_alphabet | frozenset(symbols)
        result.output_alphabet = result.output_alphabet | frozenset({error_output})
        return result

    def input_machine(self, *, build: bool = False) -> Any:
        """Return a Mealy HMM for the topological input language."""
        return _partial_machine(self, ATTR_SYMBOL, build=build)

    def output_machine(self, *, build: bool = False) -> Any:
        """Return a Mealy HMM for the topological output language."""
        return _partial_machine(self, ATTR_OUTPUT, build=build)

    def compose(self, other: MealyMachine, **kwargs: Any) -> MealyMachine:
        """Return the serial composition ``other`` after this transducer."""
        from sofic.automata.transducer_operations import compose_tt

        return compose_tt((self, other), **kwargs)

    def joint_machine(self, generator: Any, **kwargs: Any) -> Any:
        """Return the joint input/output generator induced by ``generator``."""
        from sofic.automata.transducer_operations import compose_tg

        return compose_tg(self, generator, joint=True, **kwargs)

    def transduce_generator(self, generator: Any, **kwargs: Any) -> Any:
        """Return the output generator induced by driving this transducer."""
        from sofic.automata.transducer_operations import transduce_generator

        return transduce_generator(self, generator, **kwargs)

    def labeled_transition_matrices(
        self,
        node_ordering: Sequence[Hashable] | None = None,
        *,
        mode: str = "zero",
    ) -> tuple[dict[tuple[Any, Any], np.ndarray], list[Hashable]] | dict[tuple[Any, Any], np.ndarray]:
        """Return matrices keyed by ``(input, output)`` symbol pairs."""
        norder = list(self.states()) if node_ordering is None else list(node_ordering)
        missing = [state for state in norder if not self.graph.has_state(state)]
        if missing:
            raise ValueError(f"unknown states in node_ordering: {missing!r}")

        if mode not in {"zero", "nan"}:
            raise ValueError("mode must be 'zero' or 'nan'")

        idx = {state: i for i, state in enumerate(norder)}
        n = len(norder)
        input_by_state, _ = self.alphabets(node=True)
        fill = 0.0 if mode == "zero" else np.nan

        def default_matrix(input_symbol: Any) -> np.ndarray:
            matrix = np.zeros((n, n), dtype=float)
            if mode == "nan":
                for state in norder:
                    if input_symbol not in input_by_state.get(state, frozenset()):
                        matrix[idx[state]] = fill
            return matrix

        matrices: dict[tuple[Any, Any], np.ndarray] = {}
        for transition in self.transitions():
            if transition.source not in idx or transition.target not in idx:
                continue
            input_symbol = transition.data.get(ATTR_SYMBOL, EPSILON)
            output_symbol = transition.data.get(ATTR_OUTPUT, EPSILON)
            key = (input_symbol, output_symbol)
            if key not in matrices:
                matrices[key] = default_matrix(input_symbol)
            matrices[key][idx[transition.source], idx[transition.target]] += _transition_probability(transition.data)

        if node_ordering is None:
            return matrices, norder
        return matrices

    def validate_stochastic(self) -> None:
        """Validate probability rows grouped by ``(state, input symbol)``."""
        rows: dict[tuple[Hashable, Any], float] = defaultdict(float)
        for transition in self.transitions():
            prob = _transition_probability(transition.data)
            if prob < 0:
                raise StochasticValidationError(f"negative transducer probability on {transition}")
            rows[(transition.source, transition.data.get(ATTR_SYMBOL, EPSILON))] += prob
        for (state, symbol), total in rows.items():
            if not np.isclose(total, 1.0):
                raise StochasticValidationError(f"transducer row ({state!r}, {symbol!r}) sums to {total}, not 1")

    def validate(self) -> None:
        super().validate()
        for transition in self.transitions():
            if ATTR_OUTPUT in transition.data:
                out = transition.data[ATTR_OUTPUT]
                if out is not EPSILON:
                    self._require(out in self.output_alphabet, f"output {out!r} not in output alphabet")
            if ATTR_SYMBOL in transition.data:
                sym = transition.data[ATTR_SYMBOL]
                if sym is not EPSILON:
                    self._require(sym in self.input_alphabet, f"symbol {sym!r} not in input alphabet")


class MooreMachine(Transducer):
    """Output symbols on states."""

    def add_transition(
        self,
        source: Hashable,
        target: Hashable,
        symbol: Any,
        *,
        prob: float | None = None,
        **attrs: Any,
    ) -> int:
        """Add a transition with input ``symbol``."""
        data = {ATTR_SYMBOL: symbol, **attrs}
        if prob is not None:
            data[ATTR_PROB] = float(prob)
        return self.graph.add_transition(source, target, **data)

    def set_output(self, state: Hashable, output: Any) -> None:
        """Set a state's Moore output."""
        self.graph.nx.nodes[state][ATTR_OUTPUT] = output

    def transduce(self, word: Sequence[Any]) -> set[tuple[Any, ...]]:
        from sofic.automata.transducer_simulation import transduce_moore

        return transduce_moore(self, word)

    def validate(self) -> None:
        super().validate()
        for state in self.states():
            out = self.graph.state_attrs(state).get(ATTR_OUTPUT)
            if out is not None and out is not EPSILON:
                self._require(out in self.output_alphabet, f"output {out!r} not in output alphabet")
        for transition in self.transitions():
            if ATTR_SYMBOL in transition.data:
                sym = transition.data[ATTR_SYMBOL]
                if sym is not EPSILON:
                    self._require(sym in self.input_alphabet, f"symbol {sym!r} not in input alphabet")


def _transition_probability(data: Mapping[str, Any]) -> float:
    return float(data.get(ATTR_PROB, 1.0))


def _partial_machine(machine: MealyMachine, label_attr: str, *, build: bool) -> Any:
    from sofic.generators.epsilon_machine import EpsilonMachine
    from sofic.generators.mealy import MealyHMM

    states = tuple(machine.states())
    initial_states = tuple(machine.initial_states) or states
    initial = {state: 1.0 / len(initial_states) for state in initial_states} if initial_states else {}
    alphabet = frozenset(
        transition.data[label_attr]
        for transition in machine.transitions()
        if label_attr in transition.data and transition.data[label_attr] is not EPSILON
    )
    result = MealyHMM(initial_distribution=initial, observation_alphabet=alphabet)
    for state in states:
        result.graph.add_state(state)

    row_counts: dict[Hashable, int] = defaultdict(int)
    for transition in machine.transitions():
        symbol = transition.data.get(label_attr)
        if symbol is None or symbol is EPSILON:
            continue
        row_counts[transition.source] += 1
    for transition in machine.transitions():
        symbol = transition.data.get(label_attr)
        if symbol is None or symbol is EPSILON:
            continue
        prob = 1.0 / row_counts[transition.source] if row_counts[transition.source] else 0.0
        result.graph.add_transition(transition.source, transition.target, **{ATTR_EMISSION: symbol, ATTR_PROB: prob})

    if build:
        return EpsilonMachine.from_hmm(result)
    return result
