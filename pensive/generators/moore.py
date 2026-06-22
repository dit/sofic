"""Moore-type hidden Markov models."""

from __future__ import annotations

import numpy as np

from pensive.exceptions import StochasticValidationError
from pensive.generators.base import HiddenMarkovModel
from pensive.generators.mealy import MealyHMM
from pensive.graph import ATTR_EMISSION_DIST, ATTR_PROB


class MooreHMM(HiddenMarkovModel):
    """HMM with P(o | q) on states and P(q' | q) on edges."""

    def is_unifilar(self) -> bool:
        """Return whether the Mealy conversion is row-unifilar."""
        return self.to_mealy().is_unifilar()

    def to_mealy(self) -> MealyHMM:
        from pensive.generators.conversions import moore_to_mealy

        return moore_to_mealy(self)

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
