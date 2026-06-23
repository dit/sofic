"""Bidirectional ε-machine — joint forward/reverse causal presentation."""

from __future__ import annotations

from collections.abc import Hashable
from typing import Any, Self

from pensive.exceptions import PensiveValidationError
from pensive.generators.epsilon_machine import EpsilonMachine
from pensive.generators.mealy import MealyHMM

_STEP_S_PLUS_0 = 0
_STEP_S_MINUS_0 = 1
_STEP_X_0 = 2
_STEP_S_PLUS_1 = 3
_STEP_S_MINUS_1 = 4


def _require_dit():
    try:
        import dit
    except ImportError as exc:
        raise ImportError("dit is required for entropy measures; install with `pip install pensive[measures]`") from exc
    return dit


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
                raise PensiveValidationError(f"bidirectional state must be (forward, reverse) pair, got {state!r}")

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
    def from_pair(
        cls,
        forward: EpsilonMachine,
        reverse: EpsilonMachine,
    ) -> Self:
        from pensive.generators.bidirectional_construction import build_bidirectional_epsilon_machine

        return build_bidirectional_epsilon_machine(forward, reverse)

    @classmethod
    def from_forward(cls, forward: EpsilonMachine) -> Self:
        from pensive.generators.bidirectional_construction import infer_reverse_epsilon_machine

        reverse = infer_reverse_epsilon_machine(forward)
        return cls.from_pair(forward, reverse)

    def joint_distribution(self) -> dict[tuple[Hashable, Hashable], float]:
        if self._joint_pi is not None:
            return dict(self._joint_pi)
        from pensive.generators.bidirectional_construction import joint_distribution

        return joint_distribution(self)

    def forward_epsilon_machine(self) -> EpsilonMachine:
        from pensive.generators.bidirectional_construction import forward_epsilon_machine

        return forward_epsilon_machine(self)

    def reverse_epsilon_machine(self) -> EpsilonMachine:
        from pensive.generators.bidirectional_construction import reverse_epsilon_machine

        return reverse_epsilon_machine(self)

    def step_distribution(self) -> Any:
        from pensive.generators.bidirectional_construction import bidirectional_step_distribution

        return bidirectional_step_distribution(self)

    def predicted_information(self) -> float:
        """ρ_μ = I[X₀ : S⁺₀] — predicted information rate (James et al., 2013)."""
        dit = _require_dit()
        dist = self.step_distribution()
        return float(dit.shannon.mutual_information(dist, [_STEP_X_0], [_STEP_S_PLUS_0]))

    def bound_information(self) -> float:
        """b_μ = H[X₀ | S⁺₀, S⁻₁] — bound information rate (James et al., 2013)."""
        dit = _require_dit()
        dist = self.step_distribution()
        return float(
            dit.shannon.conditional_entropy(
                dist,
                [_STEP_X_0],
                [_STEP_S_PLUS_0, _STEP_S_MINUS_1],
            )
        )

    def ephemeral_information(self) -> float:
        """r_μ = I[X₀ : S⁻₁ | S⁺₀] — ephemeral information rate (James et al., 2013)."""
        return float(self.entropy_rate() - self.bound_information())

    def excess_entropy(self) -> float:
        """Exact excess entropy E = I[S⁺; S⁻] from the bidirectional joint distribution."""
        dit = _require_dit()
        joint = self.joint_distribution()
        if not joint:
            return 0.0

        pi_plus: dict[Any, float] = {}
        pi_minus: dict[Any, float] = {}
        for (alpha, gamma), mass in joint.items():
            pi_plus[alpha] = pi_plus.get(alpha, 0.0) + mass
            pi_minus[gamma] = pi_minus.get(gamma, 0.0) + mass

        plus_outcomes = list(pi_plus.keys())
        minus_outcomes = list(pi_minus.keys())
        plus_dist = dit.Distribution(plus_outcomes, [pi_plus[s] for s in plus_outcomes])
        minus_dist = dit.Distribution(minus_outcomes, [pi_minus[s] for s in minus_outcomes])
        joint_outcomes = list(joint.keys())
        joint_dist = dit.Distribution(joint_outcomes, [joint[outcome] for outcome in joint_outcomes])
        return float(dit.shannon.entropy(plus_dist) + dit.shannon.entropy(minus_dist) - dit.shannon.entropy(joint_dist))

    def statistical_complexity(self) -> float:
        """C± = H[S⁺, S⁻] under the bidirectional stationary distribution."""
        dit = _require_dit()
        joint = self.joint_distribution()
        if not joint:
            return 0.0
        outcomes = list(joint.keys())
        probs = [joint[outcome] for outcome in outcomes]
        return float(dit.shannon.entropy(dit.Distribution(outcomes, probs)))

    def crypticity(self) -> float:
        """χ = C± − E for a bidirectional presentation."""
        return self.statistical_complexity() - self.excess_entropy()

    def minimal_generative_model(self, **kwargs: Any) -> Any:
        """Construct the minimal-state-entropy generative presentation."""
        from pensive.generators.minimal_generative_model import minimal_generative_model

        return minimal_generative_model(self, **kwargs)

    def wyner_generative_model(self, **kwargs: Any) -> Any:
        """Construct the Wyner-common-information generative presentation."""
        from pensive.generators.minimal_generative_model import wyner_generative_model

        return wyner_generative_model(self, **kwargs)

    def generative_complexity(self, **kwargs: Any) -> float:
        """C_g = H[G] for the minimal generative model."""
        return self.minimal_generative_model(**kwargs).generative_complexity()

    def information_anatomy(self) -> dict[str, float]:
        """Return ρ_μ, b_μ, r_μ, h_μ, E, and χ for this bidirectional presentation."""
        h_mu = self.entropy_rate()
        b_mu = self.bound_information()
        return {
            "rho_mu": self.predicted_information(),
            "bound_mu": b_mu,
            "ephemeral_mu": h_mu - b_mu,
            "entropy_rate": h_mu,
            "excess_entropy": self.excess_entropy(),
            "crypticity": self.crypticity(),
        }
