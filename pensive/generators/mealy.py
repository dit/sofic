"""Mealy-type hidden Markov models."""

from __future__ import annotations

from collections.abc import Hashable, Mapping, Sequence
from typing import TYPE_CHECKING, Any

from pensive.exceptions import UnifilarityError
from pensive.generators.base import HiddenMarkovModel
from pensive.generators.edge_emissions import validate_stochastic_edge_emissions
from pensive.graph import ATTR_EMISSION

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
        validate_stochastic_edge_emissions(
            self,
            alphabet=self.observation_alphabet,
            alphabet_name="observation",
            row_mass_label="joint masses",
            negative_probability_label="negative joint probability",
        )

    def to_mealy(self) -> MealyHMM:
        """Return this already-Mealy presentation."""
        return self

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
