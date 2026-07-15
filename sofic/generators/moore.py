"""Moore-type hidden Markov models."""

from __future__ import annotations

from collections.abc import Hashable, Mapping
from typing import TYPE_CHECKING, Any

import numpy as np

from sofic.exceptions import StochasticValidationError
from sofic.generators.base import HiddenMarkovModel
from sofic.generators.mealy import MealyHMM
from sofic.graph import ATTR_EMISSION_DIST, ATTR_PROB

if TYPE_CHECKING:
    from sofic.generators.lumping import LabelsLike, PartitionLike


class MooreHMM(HiddenMarkovModel):
    """HMM with P(o | q) on states and P(q' | q) on edges."""

    def set_emission_distribution(self, state: Hashable, distribution: Mapping[Any, float]) -> None:
        """Set the state emission law ``P(observation | state)``."""
        self.graph.nx.nodes[state][ATTR_EMISSION_DIST] = dict(distribution)

    def add_transition(self, source: Hashable, target: Hashable, prob: float, **attrs: Any) -> int:
        """Add an edge carrying transition probability ``P(target | source)``."""
        return self.graph.add_transition(source, target, **{ATTR_PROB: float(prob), **attrs})

    def is_unifilar(self) -> bool:
        """Return whether the Mealy conversion is row-unifilar."""
        return self.to_mealy().is_unifilar()

    def to_mealy(self) -> MealyHMM:
        from sofic.generators.conversions import moore_to_mealy

        return moore_to_mealy(self)

    def is_lumpable(self, partition: PartitionLike, *, rtol: float = 1e-8, atol: float = 1e-10) -> bool:
        """Return whether ``partition`` is strongly lumpable for this HMM."""
        from sofic.generators.lumping import is_lumpable

        return is_lumpable(self, partition, rtol=rtol, atol=atol)

    def lump(
        self,
        partition: PartitionLike,
        *,
        check: bool = True,
        labels: LabelsLike | None = None,
        rtol: float = 1e-8,
        atol: float = 1e-10,
    ) -> MooreHMM:
        """Aggregate states into blocks, returning the lumped Moore HMM."""
        from sofic.generators.lumping import lump

        return lump(self, partition, check=check, labels=labels, rtol=rtol, atol=atol)

    def validate_stochastic(self) -> None:
        super().validate_stochastic()
        for state in self.states():
            attrs = self.graph.state_attrs(state)
            emission_dist = attrs.get(ATTR_EMISSION_DIST)
            if emission_dist is not None:
                total = sum(emission_dist.values())
                if not np.isclose(total, 1.0):
                    raise StochasticValidationError(f"emission distribution at {state!r} sums to {total}")
                for symbol, prob in emission_dist.items():
                    if prob < 0:
                        raise StochasticValidationError(f"negative emission probability at {state!r}")
                    self._require(symbol in self.observation_alphabet, f"unknown emission {symbol!r}")
            outgoing = list(self.graph.out_transitions(state))
            trans_total = sum(t.data.get(ATTR_PROB, 0.0) for t in outgoing)
            if outgoing and not np.isclose(trans_total, 1.0):
                raise StochasticValidationError(f"transition probabilities from {state!r} sum to {trans_total}")
