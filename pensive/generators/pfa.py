"""Probabilistic finite automata."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from pensive.generators.base import StochasticModel
from pensive.generators.edge_emissions import validate_stochastic_edge_emissions
from pensive.generators.mealy import MealyHMM
from pensive.graph import ATTR_EMISSION, ATTR_PROB


class ProbabilisticFiniteAutomaton(StochasticModel):
    """String generator with joint symbol+probability on edges."""

    output_alphabet: frozenset[Any]

    def __init__(self, output_alphabet: frozenset[Any] | None = None, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.output_alphabet = output_alphabet if output_alphabet is not None else frozenset()

    def validate_stochastic(self) -> None:
        super().validate_stochastic()
        validate_stochastic_edge_emissions(
            self,
            alphabet=self.output_alphabet,
            alphabet_name="output",
            row_mass_label="outgoing masses",
            negative_probability_label="negative probability",
        )

    def is_unifilar(self) -> bool:
        """Return whether each state emits at most one edge per symbol."""
        from pensive.properties import is_unifilar_emissions

        return is_unifilar_emissions(self)

    def to_mealy_hmm(self) -> MealyHMM:
        from pensive.generators.conversions import pfa_to_mealy_hmm

        return pfa_to_mealy_hmm(self)

    def string_probability(self, word: Sequence[Any]) -> float:
        if not word:
            return sum(self.initial_distribution.values())
        idx = self.reindex()
        n = len(idx)
        mass = np.zeros(n, dtype=float)
        for state, prob in self.initial_distribution.items():
            mass[idx.index(state)] = float(prob)
        for symbol in word:
            updated = np.zeros(n, dtype=float)
            for transition in self.transitions():
                if transition.data.get(ATTR_EMISSION) != symbol:
                    continue
                i = idx.index(transition.source)
                j = idx.index(transition.target)
                updated[j] += mass[i] * float(transition.data.get(ATTR_PROB, 0.0))
            mass = updated
        return float(mass.sum())

    def words_of_length(self, length: int) -> dict[tuple[Any, ...], float]:
        """Return output words of ``length`` and their probabilities."""
        from pensive.generators.words import pfa_words_of_length

        return pfa_words_of_length(self, length)

    def sample(self, n: int, rng: np.random.Generator | None = None) -> list[Any]:
        generator = rng if rng is not None else np.random.default_rng()
        idx = self.reindex()
        probs = np.array([self.initial_distribution.get(s, 0.0) for s in idx.states], dtype=float)
        if probs.sum() <= 0.0:
            return []
        state = int(generator.choice(len(idx), p=probs / probs.sum()))
        output: list[Any] = []
        for _ in range(n):
            outgoing = list(self.graph.out_transitions(idx.state(state)))
            if not outgoing:
                break
            edge_probs = np.array([float(t.data.get(ATTR_PROB, 0.0)) for t in outgoing], dtype=float)
            if edge_probs.sum() <= 0.0:
                break
            edge_probs /= edge_probs.sum()
            edge = outgoing[int(generator.choice(len(outgoing), p=edge_probs))]
            emission = edge.data.get(ATTR_EMISSION)
            if emission is not None:
                output.append(emission)
            state = idx.index(edge.target)
        return output
