"""Bidirectional ε-machine — joint forward/reverse causal presentation."""

from __future__ import annotations

from collections.abc import Hashable
from typing import Any, Self

from pensive.exceptions import PensiveValidationError
from pensive.generators.epsilon_machine import EpsilonMachine
from pensive.generators.mealy import MealyHMM


class BidirectionalEpsilonMachine(MealyHMM):
    """Non-unifilar generator over joint causal states (S⁺, S⁻).

    Pairs forward and reverse ε-machines into a single presentation whose states
    are ``(forward, reverse)`` tuples. Supports excess entropy, crypticity, and
    information anatomy when ``dit`` is installed.

    Examples
    --------
    >>> from pensive.examples import golden_mean_bidirectional
    >>> bidir = golden_mean_bidirectional(0.5)
    >>> bidir.entropy_rate() > 0
    True
    """

    forward_machine: EpsilonMachine
    reverse_machine: EpsilonMachine
    _joint_pi: dict[tuple[Hashable, Hashable], float] | None = None

    def __init__(
        self,
        forward_machine: EpsilonMachine | None = None,
        reverse_machine: EpsilonMachine | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        if forward_machine is None or reverse_machine is None:
            raise PensiveValidationError("forward_machine and reverse_machine are required")
        self.forward_machine = forward_machine
        self.reverse_machine = reverse_machine

    def validate(self) -> None:
        super().validate_stochastic()
        for state in self.states():
            if not isinstance(state, tuple) or len(state) != 2:
                raise PensiveValidationError(
                    f"bidirectional state must be (forward, reverse) pair, got {state!r}"
                )

    def is_unifilar(self) -> bool:
        """Return whether joint emissions are row-unifilar (usually ``False``)."""
        from pensive.properties import is_unifilar_emissions

        return is_unifilar_emissions(self)

    def entropy_rate(self) -> float:
        """Process entropy rate h_μ (same as the forward ε-machine)."""
        return self.forward_machine.entropy_rate()

    def copy(self) -> Self:
        cloned = super().copy()
        cloned._joint_pi = None
        return cloned

    @classmethod
    def from_epsilon_machines(
        cls,
        forward: EpsilonMachine,
        reverse: EpsilonMachine,
    ) -> Self:
        from pensive.generators.bidirectional_construction import build_bidirectional_epsilon_machine

        return build_bidirectional_epsilon_machine(forward, reverse)

    @classmethod
    def from_epsilon_machine(cls, forward: EpsilonMachine) -> Self:
        from pensive.generators.bidirectional_construction import infer_reverse_epsilon_machine

        reverse = infer_reverse_epsilon_machine(forward)
        return cls.from_epsilon_machines(forward, reverse)

    def joint_distribution(self) -> dict[tuple[Hashable, Hashable], float]:
        if self._joint_pi is not None:
            return dict(self._joint_pi)
        from pensive.generators.bidirectional_construction import joint_distribution

        return joint_distribution(self)

    def marginalize_forward(self) -> EpsilonMachine:
        from pensive.generators.bidirectional_construction import marginalize_forward

        return marginalize_forward(self)

    def marginalize_reverse(self) -> EpsilonMachine:
        from pensive.generators.bidirectional_construction import marginalize_reverse

        return marginalize_reverse(self)

    def step_distribution(self) -> Any:
        from pensive.generators.bidirectional_construction import bidirectional_step_distribution

        return bidirectional_step_distribution(self)

    def predicted_information(self) -> float:
        from pensive.dit_bridge import predicted_information

        return predicted_information(self)

    def bound_information(self) -> float:
        from pensive.dit_bridge import bound_information

        return bound_information(self)

    def ephemeral_information(self) -> float:
        from pensive.dit_bridge import ephemeral_information

        return ephemeral_information(self)

    def excess_entropy(self) -> float:
        from pensive.dit_bridge import excess_entropy_bidirectional

        return excess_entropy_bidirectional(self)

    def statistical_complexity(self) -> float:
        from pensive.dit_bridge import bidirectional_statistical_complexity

        return bidirectional_statistical_complexity(self)

    def crypticity(self) -> float:
        from pensive.dit_bridge import crypticity

        return crypticity(self)

    def information_anatomy(self) -> dict[str, float]:
        from pensive.dit_bridge import information_anatomy

        return information_anatomy(self)
