"""Counting utilities for conjugate Bayesian process inference."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Hashable, Sequence
from dataclasses import dataclass
from typing import Any

from pensive.generators.mealy import MealyHMM
from pensive.graph import ATTR_EMISSION


class BayesianInferenceError(ValueError):
    """Raised when Bayesian inference inputs are inconsistent."""


def pretty_symbol(symbol: Any) -> str:
    """Format a symbol like cmpy's Bayesian inference utilities."""
    if isinstance(symbol, str):
        return symbol
    try:
        iterator = iter(symbol)
    except TypeError:
        return str(symbol)
    return ":".join(map(str, iterator))


def pretty_word(word: Sequence[Any]) -> str:
    """Format a word as comma-separated symbols."""
    return ",".join(pretty_symbol(symbol) for symbol in word)


def split_word(word: Sequence[Any]) -> tuple[tuple[Any, ...], Any]:
    """Split a word into ``(history, next_symbol)``."""
    word = tuple(word)
    return tuple(word[:-1]), word[-1]


class WordCountsMC:
    """Counts of length-``order`` contexts and following symbols."""

    def __init__(self, data: Sequence[Any], order: int):
        if order < 0:
            raise ValueError("order must be nonnegative")
        self.counts: defaultdict[tuple[tuple[Any, ...], Any], float] = defaultdict(float)
        self.order = int(order)
        self.add_counts_from(data)

    def __str__(self) -> str:
        if not self.counts:
            return "No counts."
        formatted = {
            f"{pretty_word(context)} -> {pretty_symbol(symbol)}": (context, symbol)
            for context, symbol in self.counts
        }
        integer_counts = all(float(value).is_integer() for value in self.counts.values())
        lines = []
        for label in sorted(formatted):
            value = self.counts[formatted[label]]
            value_repr = str(int(value)) if integer_counts else str(value)
            lines.append(f"n({label}) = {value_repr}")
        return "\n".join(lines) + "\n"

    def add_counts_from(self, data: Sequence[Any]) -> None:
        data = tuple(data)
        for index in range(0, max(0, len(data) - self.order)):
            context = data[index : index + self.order]
            symbol = data[index + self.order]
            self.counts[(context, symbol)] += 1
            self.counts[(context, "*")] += 1

    def clear_word_counts(self) -> None:
        self.counts = defaultdict(float)

    def get_word_count(self, word: Sequence[Any]) -> float:
        return self.counts.get(split_word(word), 0.0)

    def set_word_count(self, word: Sequence[Any], value: float) -> None:
        context, symbol = split_word(word)
        previous = self.counts.get((context, symbol), 0.0)
        self.counts[(context, symbol)] = float(value)
        self.counts[(context, "*")] = self.counts.get((context, "*"), 0.0) - previous + float(value)


@dataclass(frozen=True)
class PathTrace:
    """Counts and terminal state for one assumed start state."""

    counts: dict[Hashable | tuple[Hashable, Any], int]
    last_state: Hashable | None
    state_path: tuple[Hashable, ...] = ()


class PathCountEM:
    """State and edge counts for a unifilar candidate topology."""

    def __init__(self, machine: MealyHMM, data: Sequence[Any] | None, state_path: bool = False):
        self.machine = machine
        self.collect_state_path = state_path
        self.edges: list[tuple[Hashable, Any]] = []
        self.nodes: list[Hashable] = list(machine.states())
        self.trace: dict[tuple[Hashable, Any], Hashable] = {}
        self.counts: dict[Hashable, PathTrace] = {}
        self.possible_start_nodes: list[Hashable] = []
        self._process_machine()
        self._generate_counts(tuple(data or ()))

    def _process_machine(self) -> None:
        for transition in self.machine.transitions():
            symbol = transition.data.get(ATTR_EMISSION)
            key = (transition.source, symbol)
            if key in self.trace:
                raise BayesianInferenceError("non-unifilar topology is not allowed")
            self.trace[key] = transition.target
            if key not in self.edges:
                self.edges.append(key)
        self.edges.sort(key=repr)

    def _generate_counts(self, data: tuple[Any, ...]) -> None:
        for start in self.nodes:
            state = start
            counts: dict[Hashable | tuple[Hashable, Any], int] = {}
            path = [state]
            valid = True
            for symbol in data:
                counts[state] = counts.get(state, 0) + 1
                edge = (state, symbol)
                counts[edge] = counts.get(edge, 0) + 1
                if edge not in self.trace:
                    valid = False
                    state = None
                    path = []
                    counts = {}
                    break
                state = self.trace[edge]
                path.append(state)
            if valid:
                self.possible_start_nodes.append(start)
            self.counts[start] = PathTrace(
                counts=counts,
                last_state=state,
                state_path=tuple(path) if self.collect_state_path else (),
            )

    def get_edges(self) -> list[tuple[Hashable, Any]]:
        return list(self.edges)

    def get_nodes(self) -> list[Hashable]:
        return list(self.nodes)

    def get_edge_count(self, start_node: Hashable, edge: tuple[Hashable, Any]) -> int | None:
        return self.counts[start_node].counts.get(edge)

    def get_node_count(self, start_node: Hashable, node: Hashable) -> int | None:
        return self.counts[start_node].counts.get(node)

    def get_possible_start_nodes(self) -> list[Hashable]:
        return list(self.possible_start_nodes)

    def get_last_node(self, start_node: Hashable) -> Hashable | None:
        return self.counts[start_node].last_state

    def get_state_path(self, start_node: Hashable) -> tuple[Hashable, ...]:
        return self.counts[start_node].state_path
