"""Markov chains (visible-state generators)."""

from __future__ import annotations

from collections.abc import Hashable, Sequence
from typing import Any, Self

import numpy as np

from pensive.exceptions import StochasticValidationError
from pensive.generators.base import StochasticModel
from pensive.graph import ATTR_PROB


class MarkovChain(StochasticModel):
    """Visible-state Markov process without an observation layer."""

    def is_deterministic(self) -> bool:
        """Return whether each state has a single successor with probability 1."""
        from pensive.properties import is_deterministic_markov

        return is_deterministic_markov(self)

    def validate_stochastic(self) -> None:
        super().validate_stochastic()
        for state in self.states():
            outgoing = list(self.graph.out_transitions(state))
            total = sum(t.data.get(ATTR_PROB, 0.0) for t in outgoing)
            if outgoing and not np.isclose(total, 1.0):
                raise StochasticValidationError(f"transition probabilities from {state!r} sum to {total}")
            for transition in outgoing:
                prob = transition.data.get(ATTR_PROB, 0.0)
                if prob < 0:
                    raise StochasticValidationError(f"negative transition probability on {transition}")

    def stationary_distribution(self) -> np.ndarray:
        idx = self.reindex()
        n = len(idx)
        if n == 0:
            return np.array([], dtype=float)

        transition = np.zeros((n, n), dtype=float)
        for source in idx.states:
            i = idx.index(source)
            for edge in self.graph.out_transitions(source):
                j = idx.index(edge.target)
                transition[i, j] += float(edge.data.get(ATTR_PROB, 0.0))

        distribution = np.full(n, 1.0 / n, dtype=float)
        for _ in range(10_000):
            updated = distribution @ transition
            if np.allclose(updated, distribution, rtol=1e-10, atol=1e-12):
                distribution = updated
                break
            distribution = updated

        total = distribution.sum()
        if total <= 0.0:
            raise StochasticValidationError("failed to compute a positive stationary distribution")
        return distribution / total

    def entropy_rate(self) -> float:
        from pensive.generators.measures import entropy_rate_markov

        return entropy_rate_markov(self)

    def reverse(self) -> Self:
        from pensive.generators.reversal import time_reverse_stochastic

        return time_reverse_stochastic(self)

    def sample_path(self, n: int, rng: np.random.Generator | None = None) -> list[Hashable]:
        generator = rng if rng is not None else np.random.default_rng()
        idx = self.reindex()
        pi = self.stationary_distribution()
        state = int(generator.choice(len(idx), p=pi))

        path: list[Hashable] = []
        for _ in range(n):
            path.append(idx.state(state))
            outgoing = list(self.graph.out_transitions(idx.state(state)))
            probs = np.array([float(t.data.get(ATTR_PROB, 0.0)) for t in outgoing], dtype=float)
            if probs.sum() <= 0.0:
                break
            probs /= probs.sum()
            choice = int(generator.choice(len(outgoing), p=probs))
            state = idx.index(outgoing[choice].target)
        return path
