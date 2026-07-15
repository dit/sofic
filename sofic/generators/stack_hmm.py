"""Hidden Markov generators with visibly pushdown stack state."""

from __future__ import annotations

from collections.abc import Hashable, Mapping, Sequence
from typing import Any, Literal

import numpy as np

from sofic.exceptions import StochasticValidationError
from sofic.generators.base import StochasticModel
from sofic.generators.stationary import stationary_distribution_from_transition
from sofic.graph import (
    ATTR_KIND,
    ATTR_PROB,
    ATTR_SYMBOL,
    KIND_CALL,
    KIND_INTERNAL,
    KIND_RETURN,
    Transition,
    TransitionGraph,
)
from sofic.shifts.sofic_dyck import MatchedEdge, SoficDyckShift, TransitionRef, transition_ref

Configuration = tuple[Hashable, tuple[TransitionRef, ...]]
_KINDS = frozenset({KIND_CALL, KIND_RETURN, KIND_INTERNAL})
_TOL = 1e-15


class HiddenMarkovStackModel(StochasticModel):
    """Stochastic visibly-pushdown generator over a finite control graph.

    The hidden configuration is a finite control state together with an
    unbounded stack of call-edge references. Outgoing edge probabilities are
    interpreted as weights over the transitions enabled by the current stack.
    The enabled weights are normalized at each step, so return transitions that
    are disabled by the stack do not make the stochastic row invalid.
    """

    call_alphabet: frozenset[Any]
    return_alphabet: frozenset[Any]
    internal_alphabet: frozenset[Any]
    symbol_alphabet: frozenset[Any]
    matched_edges: frozenset[MatchedEdge]
    allow_empty_stack_returns: bool

    def __init__(
        self,
        call_alphabet: frozenset[Any] | None = None,
        return_alphabet: frozenset[Any] | None = None,
        internal_alphabet: frozenset[Any] | None = None,
        matched_edges: set[MatchedEdge] | frozenset[MatchedEdge] | None = None,
        symbol_alphabet: frozenset[Any] | None = None,
        allow_empty_stack_returns: bool = True,
        **kwargs: Any,
    ) -> None:
        self.call_alphabet = call_alphabet if call_alphabet is not None else frozenset()
        self.return_alphabet = return_alphabet if return_alphabet is not None else frozenset()
        self.internal_alphabet = internal_alphabet if internal_alphabet is not None else frozenset()
        inferred_alphabet = self.call_alphabet | self.return_alphabet | self.internal_alphabet
        self.symbol_alphabet = symbol_alphabet if symbol_alphabet is not None else inferred_alphabet
        self.matched_edges = frozenset(matched_edges or frozenset())
        self.allow_empty_stack_returns = allow_empty_stack_returns
        super().__init__(**kwargs)

    def validate_stochastic(self) -> None:
        super().validate_stochastic()
        role_alphabet = self.call_alphabet | self.return_alphabet | self.internal_alphabet
        self._require(
            len(self.call_alphabet) + len(self.return_alphabet) + len(self.internal_alphabet) == len(role_alphabet),
            "call, return, and internal alphabets must be disjoint",
        )
        self._require(self.symbol_alphabet == role_alphabet, "symbol_alphabet must equal the visible role alphabets")

        call_edges: set[TransitionRef] = set()
        return_edges: set[TransitionRef] = set()
        all_edges: set[TransitionRef] = set()
        outgoing_mass = dict.fromkeys(self.states(), 0.0)
        for transition in self.transitions():
            ref = transition_ref(transition)
            all_edges.add(ref)
            kind = transition.data.get(ATTR_KIND)
            symbol = transition.data.get(ATTR_SYMBOL)
            prob = float(transition.data.get(ATTR_PROB, 0.0))
            if prob < 0.0:
                raise StochasticValidationError(f"negative transition probability on {transition}")
            outgoing_mass[transition.source] = outgoing_mass.get(transition.source, 0.0) + prob
            self._require(kind in _KINDS, f"invalid stack edge kind {kind!r}")
            self._require(symbol is not None, "stack generator transitions require a symbol")
            if kind == KIND_CALL:
                self._require(symbol in self.call_alphabet, f"{symbol!r} not in call alphabet")
                call_edges.add(ref)
            elif kind == KIND_RETURN:
                self._require(symbol in self.return_alphabet, f"{symbol!r} not in return alphabet")
                return_edges.add(ref)
            else:
                self._require(symbol in self.internal_alphabet, f"{symbol!r} not in internal alphabet")

        for state, mass in outgoing_mass.items():
            if mass <= 0.0:
                raise StochasticValidationError(f"outgoing transition mass from {state!r} is not positive")

        for call_ref, return_ref in self.matched_edges:
            self._require(call_ref in all_edges, f"matched call edge {call_ref!r} is missing")
            self._require(return_ref in all_edges, f"matched return edge {return_ref!r} is missing")
            self._require(call_ref in call_edges, f"matched edge {call_ref!r} is not a call transition")
            self._require(return_ref in return_edges, f"matched edge {return_ref!r} is not a return transition")

    def add_call_transition(
        self,
        source: Hashable,
        target: Hashable,
        symbol: Any,
        prob: float,
        **attrs: Any,
    ) -> TransitionRef:
        """Add a call transition and return its stable edge reference."""
        data = {**attrs, ATTR_KIND: KIND_CALL, ATTR_SYMBOL: symbol, ATTR_PROB: prob}
        key = self.graph.add_transition(source, target, **data)
        return source, target, key

    def add_return_transition(
        self,
        source: Hashable,
        target: Hashable,
        symbol: Any,
        prob: float,
        **attrs: Any,
    ) -> TransitionRef:
        """Add a return transition and return its stable edge reference."""
        data = {**attrs, ATTR_KIND: KIND_RETURN, ATTR_SYMBOL: symbol, ATTR_PROB: prob}
        key = self.graph.add_transition(source, target, **data)
        return source, target, key

    def add_internal_transition(
        self,
        source: Hashable,
        target: Hashable,
        symbol: Any,
        prob: float,
        **attrs: Any,
    ) -> TransitionRef:
        """Add an internal transition and return its stable edge reference."""
        data = {**attrs, ATTR_KIND: KIND_INTERNAL, ATTR_SYMBOL: symbol, ATTR_PROB: prob}
        key = self.graph.add_transition(source, target, **data)
        return source, target, key

    def add_matched_pair(self, call_ref: TransitionRef, return_ref: TransitionRef) -> None:
        """Mark ``call_ref`` and ``return_ref`` as a legal call-return pair."""
        self.matched_edges = frozenset({*self.matched_edges, (call_ref, return_ref)})

    def sample(
        self,
        n: int,
        rng: np.random.Generator | None = None,
    ) -> tuple[list[Any], list[Configuration]]:
        """Generate up to ``n`` symbols and the pre-emission configurations."""
        if n < 0:
            raise ValueError("n must be nonnegative")
        generator = rng if rng is not None else np.random.default_rng()
        initial_states = [state for state, mass in self.initial_distribution.items() if mass > 0.0]
        if not initial_states:
            return [], []
        initial_probs = np.array([float(self.initial_distribution[state]) for state in initial_states], dtype=float)
        state = initial_states[int(generator.choice(len(initial_states), p=initial_probs / initial_probs.sum()))]
        config: Configuration = (state, ())

        observations: list[Any] = []
        configurations: list[Configuration] = []
        for _ in range(n):
            successors = self._normalized_successors(config)
            if not successors:
                break
            probs = np.array([prob for _transition, prob, _next_config in successors], dtype=float)
            transition, _prob, next_config = successors[int(generator.choice(len(successors), p=probs))]
            symbol = transition.data.get(ATTR_SYMBOL)
            if symbol is None:
                break
            configurations.append(config)
            observations.append(symbol)
            config = next_config
        return observations, configurations

    def word_probability(self, word: Sequence[Any]) -> float:
        """Return the probability of emitting ``word`` from the initial law."""
        word = tuple(word)
        if not word:
            return float(sum(self.initial_distribution.values()))
        if any(symbol not in self.symbol_alphabet for symbol in word):
            return 0.0

        current: dict[Configuration, float] = {
            (state, ()): float(prob) for state, prob in self.initial_distribution.items() if prob > _TOL
        }
        for symbol in word:
            next_masses: dict[Configuration, float] = {}
            for config, mass in current.items():
                for transition, prob, next_config in self._normalized_successors(config):
                    if transition.data.get(ATTR_SYMBOL) != symbol:
                        continue
                    next_masses[next_config] = next_masses.get(next_config, 0.0) + mass * prob
            current = {config: mass for config, mass in next_masses.items() if mass > _TOL}
            if not current:
                return 0.0
        return float(sum(current.values()))

    def words_of_length(self, length: int) -> dict[tuple[Any, ...], float]:
        """Return emitted words of ``length`` and their probabilities."""
        if length < 0:
            raise ValueError("length must be nonnegative")
        if length == 0:
            total = float(sum(self.initial_distribution.values()))
            return {(): total} if total > _TOL else {}

        layers: dict[tuple[Any, ...], dict[Configuration, float]] = {
            (): {(state, ()): float(prob) for state, prob in self.initial_distribution.items() if prob > _TOL}
        }
        for _ in range(length):
            next_layers: dict[tuple[Any, ...], dict[Configuration, float]] = {}
            for prefix, configs in layers.items():
                for config, mass in configs.items():
                    for transition, prob, next_config in self._normalized_successors(config):
                        symbol = transition.data.get(ATTR_SYMBOL)
                        if symbol is None:
                            continue
                        next_prefix = prefix + (symbol,)
                        bucket = next_layers.setdefault(next_prefix, {})
                        bucket[next_config] = bucket.get(next_config, 0.0) + mass * prob
            layers = next_layers
            if not layers:
                break

        distribution: dict[tuple[Any, ...], float] = {}
        for word, configs in layers.items():
            probability = float(sum(configs.values()))
            if probability > _TOL:
                distribution[word] = probability
        return distribution

    def reachable_configurations(self, max_stack_depth: int) -> tuple[Configuration, ...]:
        """Return configurations reachable from the initial law up to stack depth."""
        if max_stack_depth < 0:
            raise ValueError("max_stack_depth must be nonnegative")
        starts = [(state, ()) for state, prob in self.initial_distribution.items() if prob > _TOL]
        configurations: list[Configuration] = []
        seen: set[Configuration] = set()
        queue = list(starts)
        for config in starts:
            seen.add(config)

        while queue:
            config = queue.pop(0)
            configurations.append(config)
            for _transition, _prob, next_config in self._normalized_successors(
                config,
                max_stack_depth=max_stack_depth,
            ):
                if next_config in seen:
                    continue
                seen.add(next_config)
                queue.append(next_config)
        return tuple(configurations)

    def configuration_transition_matrix(self, max_stack_depth: int) -> np.ndarray:
        """Return the finite-depth transition matrix over reachable configurations."""
        matrix, _configs = self._configuration_transition_matrix_and_configs(max_stack_depth)
        return matrix

    def stationary_distribution(
        self,
        max_stack_depth: int,
        marginal: Literal["control", "configuration"] = "control",
    ) -> np.ndarray:
        """Return a finite-depth stationary distribution.

        ``marginal="configuration"`` returns the distribution over truncated
        stack configurations. ``marginal="control"`` sums those masses over the
        finite control states in this model's normal state order.
        """
        matrix, configs = self._configuration_transition_matrix_and_configs(max_stack_depth)
        pi = stationary_distribution_from_transition(matrix)
        if marginal == "configuration":
            return pi
        if marginal != "control":
            raise ValueError("marginal must be 'control' or 'configuration'")

        idx = self.reindex()
        control = np.zeros(len(idx), dtype=float)
        for mass, (state, _stack) in zip(pi, configs, strict=True):
            control[idx.index(state)] += float(mass)
        return control

    def to_sofic_dyck_shift(self) -> SoficDyckShift:
        """Strip probabilities and return the positive-probability Dyck support."""
        graph = TransitionGraph()
        for state in self.states():
            graph.add_state(state, **self.graph.state_attrs(state))

        edge_map: dict[TransitionRef, TransitionRef] = {}
        for transition in self.transitions():
            if float(transition.data.get(ATTR_PROB, 0.0)) <= 0.0:
                continue
            data = {key: value for key, value in transition.data.items() if key != ATTR_PROB}
            key = graph.add_transition(transition.source, transition.target, **data)
            edge_map[transition_ref(transition)] = (transition.source, transition.target, key)

        matched_edges = frozenset(
            (edge_map[call_ref], edge_map[return_ref])
            for call_ref, return_ref in self.matched_edges
            if call_ref in edge_map and return_ref in edge_map
        )
        return SoficDyckShift(
            graph=graph,
            call_alphabet=self.call_alphabet,
            return_alphabet=self.return_alphabet,
            internal_alphabet=self.internal_alphabet,
            matched_edges=matched_edges,
            symbol_alphabet=self.symbol_alphabet,
        )

    @classmethod
    def from_sofic_dyck_shift(
        cls,
        shift: SoficDyckShift,
        probabilities: Mapping[TransitionRef, float],
        initial_distribution: Mapping[Hashable, float] | None = None,
        **kwargs: Any,
    ) -> HiddenMarkovStackModel:
        """Build a stochastic stack model by assigning probabilities to a Dyck shift."""
        graph = TransitionGraph()
        for state in shift.states():
            graph.add_state(state, **shift.graph.state_attrs(state))

        edge_map: dict[TransitionRef, TransitionRef] = {}
        for transition in shift.transitions():
            ref = transition_ref(transition)
            if ref not in probabilities:
                raise ValueError(f"missing probability for transition {ref!r}")
            data = dict(transition.data)
            data[ATTR_PROB] = float(probabilities[ref])
            key = graph.add_transition(transition.source, transition.target, **data)
            edge_map[ref] = (transition.source, transition.target, key)

        matched_edges = frozenset(
            (edge_map[call_ref], edge_map[return_ref]) for call_ref, return_ref in shift.matched_edges
        )
        initial = (
            dict(initial_distribution) if initial_distribution is not None else _uniform_initial_distribution(shift)
        )
        return cls(
            graph=graph,
            initial_distribution=initial,
            call_alphabet=shift.call_alphabet,
            return_alphabet=shift.return_alphabet,
            internal_alphabet=shift.internal_alphabet,
            matched_edges=matched_edges,
            symbol_alphabet=shift.symbol_alphabet,
            **kwargs,
        )

    def _configuration_transition_matrix_and_configs(
        self,
        max_stack_depth: int,
    ) -> tuple[np.ndarray, tuple[Configuration, ...]]:
        configs = self.reachable_configurations(max_stack_depth)
        matrix = np.zeros((len(configs), len(configs)), dtype=float)
        config_index = {config: i for i, config in enumerate(configs)}
        for i, config in enumerate(configs):
            successors = self._normalized_successors(config, max_stack_depth=max_stack_depth)
            if not successors:
                matrix[i, i] = 1.0
                continue
            for _transition, prob, next_config in successors:
                matrix[i, config_index[next_config]] += prob
        return matrix, configs

    def _normalized_successors(
        self,
        config: Configuration,
        max_stack_depth: int | None = None,
    ) -> tuple[tuple[Transition, float, Configuration], ...]:
        state, stack = config
        weighted: list[tuple[Transition, float, Configuration]] = []
        total = 0.0
        for transition in self.graph.out_transitions(state):
            weight = float(transition.data.get(ATTR_PROB, 0.0))
            if weight <= 0.0:
                continue
            next_stack = self._next_stack(transition, stack, max_stack_depth=max_stack_depth)
            if next_stack is None:
                continue
            weighted.append((transition, weight, (transition.target, next_stack)))
            total += weight
        if total <= 0.0:
            return ()
        return tuple((transition, weight / total, next_config) for transition, weight, next_config in weighted)

    def _next_stack(
        self,
        transition: Transition,
        stack: tuple[TransitionRef, ...],
        max_stack_depth: int | None = None,
    ) -> tuple[TransitionRef, ...] | None:
        kind = transition.data.get(ATTR_KIND)
        ref = transition_ref(transition)
        if kind == KIND_CALL:
            if max_stack_depth is not None and len(stack) >= max_stack_depth:
                return None
            return stack + (ref,)
        if kind == KIND_RETURN:
            if not stack:
                return stack if self.allow_empty_stack_returns else None
            if (stack[-1], ref) in self.matched_edges:
                return stack[:-1]
            return None
        if kind == KIND_INTERNAL:
            return stack
        return None


def _uniform_initial_distribution(shift: SoficDyckShift) -> dict[Hashable, float]:
    states = tuple(shift.states())
    if not states:
        return {}
    probability = 1.0 / len(states)
    return dict.fromkeys(states, probability)
