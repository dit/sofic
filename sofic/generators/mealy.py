"""Mealy-type hidden Markov models."""

from __future__ import annotations

from collections.abc import Hashable, Mapping, Sequence
from typing import TYPE_CHECKING, Any

from sofic.exceptions import UnifilarityError
from sofic.generators.base import HiddenMarkovModel
from sofic.generators.edge_emissions import validate_stochastic_edge_emissions
from sofic.graph import ATTR_EMISSION, ATTR_PROB

if TYPE_CHECKING:
    from sofic.generators.lumping import LabelsLike, PartitionLike
    from sofic.generators.mixed_state import MixedState, MixedStatePresentation


class MealyHMM(HiddenMarkovModel):
    """HMM with joint transition-emission law P(q', o | q) on edges.

    Each outgoing edge carries an emission symbol and a probability. Row sums at
    every state must equal 1. Use :meth:`mixed_state_presentation` to obtain
    belief-state dynamics, or :meth:`~sofic.generators.epsilon_machine.EpsilonMachine.from_hmm`
    for the causal ε-machine presentation.

    Examples
    --------
    >>> from sofic.examples import golden_mean
    >>> eps = golden_mean(0.5)
    >>> eps.entropy_rate() > 0
    True
    """

    def add_transition(self, source: Hashable, target: Hashable, symbol: Any, prob: float, **attrs: Any) -> int:
        """Add an edge carrying joint emission probability ``P(target, symbol | source)``.

        ``prob`` may be a Python float or an exact sympy expression (see
        :mod:`sofic.generators.prob`).
        """
        from sofic.generators.prob import as_prob

        return self.graph.add_transition(
            source,
            target,
            **{ATTR_EMISSION: symbol, ATTR_PROB: as_prob(prob), **attrs},
        )

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
        from sofic.properties import is_unifilar_emissions

        return is_unifilar_emissions(self)

    def is_counifilar(self) -> bool:
        """Return whether each ``(target, emission)`` identifies a unique source."""
        from sofic.properties import is_counifilar_emissions

        return is_counifilar_emissions(self)

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
    ) -> MealyHMM:
        """Aggregate states into blocks, returning the lumped Mealy HMM."""
        from sofic.generators.lumping import lump

        return lump(self, partition, check=check, labels=labels, rtol=rtol, atol=atol)

    def is_irreducible(self) -> bool:
        """Return whether the internal state graph is strongly connected."""
        from sofic.properties import is_irreducible

        return is_irreducible(self)

    def is_ergodic(self, *, weak: bool = True) -> bool:
        """Return weak/strong ergodicity of the internal finite-state dynamics."""
        from sofic.properties import is_ergodic

        return is_ergodic(self, weak=weak)

    def is_stationary(self, *, rtol: float = 1e-8, atol: float = 1e-10) -> bool:
        """Return whether the initial distribution is internally stationary."""
        from sofic.properties import is_stationary

        return is_stationary(self, rtol=rtol, atol=atol)

    def is_detailed_balance(self, *, rtol: float = 1e-8, atol: float = 1e-10) -> bool:
        """Return whether stationary labeled flows satisfy detailed balance."""
        from sofic.properties import is_detailed_balance

        return is_detailed_balance(self, rtol=rtol, atol=atol)

    def is_periodic(self) -> bool:
        """Return whether terminal internal components have graph period greater than one."""
        from sofic.properties import is_periodic

        return is_periodic(self)

    def is_strictly_sofic(self) -> bool:
        """Return whether this generator's support is strictly sofic."""
        from sofic.properties import is_strictly_sofic

        return is_strictly_sofic(self)

    def mixed_state_presentation(
        self,
        *,
        initial_mixed_state: MixedState | Mapping[Hashable, float] | Sequence[float] | None = None,
    ) -> MixedStatePresentation:
        """Build the mixed-state presentation (observer belief dynamics)."""
        from sofic.generators.mixed_state import MixedStatePresentation

        return MixedStatePresentation.from_presentation(self, initial_mixed_state=initial_mixed_state)

    def to_edge_machine(self, iterations: int = 1, style: int = 0) -> MealyHMM:
        from sofic.generators.edge_machine import hmm_to_edge_machine

        return hmm_to_edge_machine(self, iterations=iterations, style=style)
