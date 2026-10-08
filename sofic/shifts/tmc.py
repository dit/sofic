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

        return from_adjacency(matrix, symbol_alphabet, cls=cls, **kwargs)

    def to_sofic_shift(self) -> SoficShift:
        from sofic.shifts.tmc_construction import to_sofic_shift

        return to_sofic_shift(self)

    def topological_entropy(self) -> float:
        """Edge-shift entropy in bits: ``log2`` spectral radius counting edge multiplicities.

        A TMC is an edge shift, so parallel edges are distinct points even when
        they share a label; :meth:`to_sofic_shift` followed by
        :meth:`~sofic.shifts.sofic.SoficShift.topological_entropy` gives the
        entropy of the labeled (sofic) shift instead.
        """
        from sofic.shifts.tmc_construction import topological_entropy

        return topological_entropy(self)

    def periodic_points(self, n: int) -> int:
        """Number of points of period ``n``: ``tr A^n`` counting edge multiplicities.

        See :func:`~sofic.shifts.algorithms.periodic_points` :cite:`LindMarcus1995`.
        """
        from sofic.shifts.algorithms import periodic_points

        return periodic_points(self, n)

    def zeta_function(self, t: Any = None) -> Any:
        """Zeta function ``1 / det(I - tA)`` as a sympy rational function.

        See :func:`~sofic.shifts.algorithms.zeta_function` :cite:`LindMarcus1995`.
        """
        from sofic.shifts.algorithms import zeta_function

        return zeta_function(self, t)

    def validate(self) -> None:
        super().validate()
        for transition in self.transitions():
            mult = transition.data.get(ATTR_MULTIPLICITY, 1)
            self._require(isinstance(mult, (int, float)) and mult >= 1, "multiplicity must be >= 1")

    def parry_measure(self) -> MealyHMM:
        from sofic.shifts.parry_construction import parry_measure

        return parry_measure(self)
