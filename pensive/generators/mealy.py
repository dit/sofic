"""Mealy-type hidden Markov models."""

from __future__ import annotations

from collections.abc import Hashable, Mapping, Sequence
from typing import TYPE_CHECKING, Any

import numpy as np

from pensive.exceptions import StochasticValidationError, UnifilarityError
from pensive.generators.base import HiddenMarkovModel
from pensive.graph import ATTR_EMISSION, ATTR_PROB

if TYPE_CHECKING:
    from pensive.generators.mixed_state import MixedState, MixedStatePresentation


class MealyHMM(HiddenMarkovModel):
    """HMM with joint transition-emission law P(q', o | q) on edges.

    Each outgoing edge carries an emission symbol and a probability. Row sums at
    every state must equal 1. Use :meth:`mixed_state_presentation` to obtain
    belief-state dynamics, or :meth:`~pensive.generators.epsilon_machine.EpsilonMachine.from_generator`
    for the causal ε-machine presentation.

    Examples
    --------
    >>> from pensive.examples import golden_mean
    >>> eps = golden_mean(0.5)
    >>> eps.entropy_rate() > 0
    True
    """

    def validate_stochastic(self) -> None:
        super().validate_stochastic()
        for state in self.states():
            outgoing = list(self.graph.out_transitions(state))
            total = sum(t.data.get(ATTR_PROB, 0.0) for t in outgoing)
            if outgoing and not np.isclose(total, 1.0):
                raise StochasticValidationError(f"joint masses from {state!r} sum to {total}")
            for transition in outgoing:
                prob = transition.data.get(ATTR_PROB, 0.0)
                if prob < 0:
                    raise StochasticValidationError(f"negative joint probability on {transition}")
                emission = transition.data.get(ATTR_EMISSION)
                if emission is not None:
                    self._require(
                        emission in self.observation_alphabet,
                        f"emission {emission!r} not in observation alphabet",
                    )

    def _check_unifilar(self) -> None:
        if self.is_unifilar():
            return
        seen: set[tuple[Hashable, Any]] = set()
        for transition in self.transitions():
            emission = transition.data.get(ATTR_EMISSION)
            if emission is None:
                continue
            key = (transition.source, emission)
            if key in seen:
                raise UnifilarityError(f"duplicate emission {emission!r} from state {transition.source!r}")
            seen.add(key)

    def is_unifilar(self) -> bool:
        """Return whether each state emits at most one edge per symbol."""
        from pensive.properties import is_unifilar_emissions

        return is_unifilar_emissions(self)

    def mixed_state_presentation(
        self,
        *,
        initial_mixed_state: MixedState | Mapping[Hashable, float] | Sequence[float] | None = None,
    ) -> MixedStatePresentation:
        """Build the mixed-state presentation (observer belief dynamics)."""
        from pensive.generators.mixed_state import MixedStatePresentation

        return MixedStatePresentation.from_presentation(self, initial_mixed_state=initial_mixed_state)

    def to_edge_machine(self) -> MealyHMM:
        from pensive.generators.edge_machine import edge_machine_from_hmm

        return edge_machine_from_hmm(self)
