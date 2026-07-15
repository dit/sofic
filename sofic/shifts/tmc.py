"""Topological Markov chains."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from sofic.generators.mealy import MealyHMM
from sofic.graph import ATTR_MULTIPLICITY
from sofic.shifts.base import SymbolicModel
from sofic.shifts.sofic import SoficShift


class TopologicalMarkovChain(SymbolicModel):
    """Adjacency-matrix presentation with edge multiplicities."""

    @classmethod
    def from_adjacency(
        cls, matrix: np.ndarray, symbol_alphabet: frozenset[Any] | None = None, **kwargs: Any
    ) -> TopologicalMarkovChain:
        from sofic.shifts.tmc_construction import from_adjacency

        return from_adjacency(matrix, symbol_alphabet)

    def to_sofic_shift(self) -> SoficShift:
        from sofic.shifts.tmc_construction import to_sofic_shift

        return to_sofic_shift(self)

    def topological_entropy(self) -> float:
        from sofic.shifts.tmc_construction import topological_entropy

        return topological_entropy(self)

    def validate(self) -> None:
        super().validate()
        for transition in self.transitions():
            mult = transition.data.get(ATTR_MULTIPLICITY, 1)
            self._require(isinstance(mult, (int, float)) and mult >= 1, "multiplicity must be >= 1")

    def parry_measure(self) -> MealyHMM:
        from sofic.shifts.parry_construction import parry_measure

        return parry_measure(self)
