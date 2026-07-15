"""Bidirectional ε-machine — joint forward/reverse causal presentation."""

from __future__ import annotations

from collections.abc import Hashable
from typing import Any, Self

from sofic.exceptions import SoficValidationError
from sofic.generators.epsilon_machine import EpsilonMachine
from sofic.generators.mealy import MealyHMM

_STEP_S_PLUS_0 = 0
_STEP_S_MINUS_0 = 1
_STEP_X_0 = 2
_STEP_S_PLUS_1 = 3
_STEP_S_MINUS_1 = 4


def _require_dit():
    from sofic.generators.measures import require_dit

    return require_dit("entropy measures")


def _measure_value(dist: Any, value: Any) -> Any:
    """Return a dit measure result as Expr when ``dist`` is symbolic, else float."""
    if hasattr(dist, "is_symbolic") and dist.is_symbolic():
        return value
    return float(value)


class BidirectionalEpsilonMachine(MealyHMM):
    """Non-unifilar generator over joint causal states (S⁺, S⁻).

    Pairs forward and reverse ε-machines into a single presentation whose states
    are ``(forward, reverse)`` tuples. Supports excess entropy, crypticity, and
    information anatomy when ``dit`` is installed.

    Examples
    --------
    >>> from sofic.examples import golden_mean_bidirectional
    >>> bidir = golden_mean_bidirectional(0.5)
    >>> bidir.entropy_rate() > 0
    True
    """

    forward_machine: EpsilonMachine
    reverse_machine: EpsilonMachine
    _joint_pi: dict[tuple[Hashable, Hashable], Any] | None = None

    def __init__(
        self,
        forward_machine: EpsilonMachine | None = None,
        reverse_machine: EpsilonMachine | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        if forward_machine is None or reverse_machine is None:
            raise SoficValidationError("forward_machine and reverse_machine are required")
        self.forward_machine = forward_machine
        self.reverse_machine = reverse_machine

    def validate(self) -> None:
        super().validate_stochastic()
        for state in self.states():
            if not isinstance(state, tuple) or len(state) != 2:
                raise SoficValidationError(f"bidirectional state must be (forward, reverse) pair, got {state!r}")

    def is_unifilar(self) -> bool:
        """Return whether joint emissions are row-unifilar (usually ``False``)."""
        from sofic.properties import is_unifilar_emissions

        return is_unifilar_emissions(self)

    def entropy_rate(self) -> Any:
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
    ) -> BidirectionalEpsilonMachine:
        from sofic.generators.bidirectional_construction import build_bidirectional_epsilon_machine

        return build_bidirectional_epsilon_machine(forward, reverse)

    @classmethod
    def from_forward(cls, forward: EpsilonMachine) -> BidirectionalEpsilonMachine:
        from sofic.generators.bidirectional_construction import infer_reverse_epsilon_machine

        reverse = infer_reverse_epsilon_machine(forward)
        return cls.from_pair(forward, reverse)

    def joint_distribution(self) -> dict[tuple[Hashable, Hashable], Any]:
        if self._joint_pi is not None:
            return dict(self._joint_pi)
        from sofic.generators.bidirectional_construction import joint_distribution

        return joint_distribution(self)

    def forward_epsilon_machine(self) -> EpsilonMachine:
        from sofic.generators.bidirectional_construction import forward_epsilon_machine

        return forward_epsilon_machine(self)

    def reverse_epsilon_machine(self) -> EpsilonMachine:
        from sofic.generators.bidirectional_construction import reverse_epsilon_machine

        return reverse_epsilon_machine(self)

    def step_distribution(self) -> Any:
        from sofic.generators.bidirectional_construction import bidirectional_step_distribution

        return bidirectional_step_distribution(self)

    def predicted_information(self) -> Any:
        """ρ_μ = I[X₀ : S⁺₀] — predicted information rate (James et al., 2013)."""
        dit = _require_dit()
        dist = self.step_distribution()
        return _measure_value(dist, dit.shannon.mutual_information(dist, [_STEP_X_0], [_STEP_S_PLUS_0]))

    def bound_information(self) -> Any:
        """b_μ = I[X₀ : S⁻₁ | S⁺₀] — bound information rate (James et al., 2013)."""
        dit = _require_dit()
        dist = self.step_distribution()
        return _measure_value(
            dist,
            dit.shannon.conditional_entropy(dist, [_STEP_X_0], [_STEP_S_PLUS_0])
            - dit.shannon.conditional_entropy(
                dist,
                [_STEP_X_0],
                [_STEP_S_PLUS_0, _STEP_S_MINUS_1],
            ),
        )

    def ephemeral_information(self) -> Any:
        """r_μ = H[X₀ | S⁺₀, S⁻₁] — ephemeral information rate (James et al., 2013)."""
        dit = _require_dit()
        dist = self.step_distribution()
        return _measure_value(
            dist,
            dit.shannon.conditional_entropy(
                dist,
                [_STEP_X_0],
                [_STEP_S_PLUS_0, _STEP_S_MINUS_1],
            ),
        )

    def structural_ephemeral_information(self) -> Any:
        """r_μ^struct = H[S⁺₁ | S⁺₀, S⁻₁] — structural (branching) part of r_μ.

        The next forward causal state S⁺₁ is a deterministic function of S⁺₀ and
        X₀, so this equals I[X₀ : S⁺₁ | S⁺₀, S⁻₁]: the ephemeral randomness that
        selects among transitions to *different* next states (edges with
        structural consequence) and is not resolved by the future S⁻₁. Together
        with :meth:`parallel_edge_information` it partitions
        :meth:`ephemeral_information` (r_μ = r_μ^struct + r_μ^par).

        This refinement of the information anatomy (James et al., 2013) has no
        separate canonical source; it follows from the determinism of the
        forward transition function.
        """
        dit = _require_dit()
        dist = self.step_distribution()
        return _measure_value(
            dist,
            dit.shannon.conditional_entropy(
                dist,
                [_STEP_S_PLUS_1],
                [_STEP_S_PLUS_0, _STEP_S_MINUS_1],
            ),
        )

    def parallel_edge_information(self) -> Any:
        """r_μ^par = H[X₀ | S⁺₀, S⁺₁, S⁻₁] — parallel-edge (gauge) part of r_μ.

        Once the source S⁺₀ and destination S⁺₁ forward causal states are both
        fixed, the residual symbol uncertainty is pure output relabeling on edges
        "from the same state to the same state" — no structural consequence. It
        is the gauge component of the ephemeral information, complementary to
        :meth:`structural_ephemeral_information` (r_μ = r_μ^struct + r_μ^par).

        Refinement of the information anatomy (James et al., 2013); no separate
        canonical source.
        """
        dit = _require_dit()
        dist = self.step_distribution()
        return _measure_value(
            dist,
            dit.shannon.conditional_entropy(
                dist,
                [_STEP_X_0],
                [_STEP_S_PLUS_0, _STEP_S_PLUS_1, _STEP_S_MINUS_1],
            ),
        )

    def bound_structural_information(self) -> Any:
        """b_μ^struct = I[S⁺₁ : S⁻₁ | S⁺₀] — structural (branching) part of b_μ.

        Companion to :meth:`structural_ephemeral_information`: the transition
        randomness that changes the next forward causal state *and* is shared
        with the future S⁻₁. Together with :meth:`bound_parallel_edge_information`
        it partitions :meth:`bound_information` (b_μ = b_μ^struct + b_μ^par).

        Refinement of the information anatomy (James et al., 2013); no separate
        canonical source.
        """
        dit = _require_dit()
        dist = self.step_distribution()
        return _measure_value(
            dist,
            dit.shannon.conditional_entropy(dist, [_STEP_S_PLUS_1], [_STEP_S_PLUS_0])
            - dit.shannon.conditional_entropy(
                dist,
                [_STEP_S_PLUS_1],
                [_STEP_S_PLUS_0, _STEP_S_MINUS_1],
            ),
        )

    def bound_parallel_edge_information(self) -> Any:
        """b_μ^par = I[X₀ : S⁻₁ | S⁺₀, S⁺₁] — parallel-edge (gauge) part of b_μ.

        Companion to :meth:`parallel_edge_information`: the output-relabeling
        randomness on a fixed transition (source and destination forward states
        held constant) that is nonetheless shared with the future S⁻₁. Together
        with :meth:`bound_structural_information` it partitions
        :meth:`bound_information` (b_μ = b_μ^struct + b_μ^par).

        Refinement of the information anatomy (James et al., 2013); no separate
        canonical source.
        """
        dit = _require_dit()
        dist = self.step_distribution()
        return _measure_value(
            dist,
            dit.shannon.conditional_entropy(dist, [_STEP_X_0], [_STEP_S_PLUS_0, _STEP_S_PLUS_1])
            - dit.shannon.conditional_entropy(
                dist,
                [_STEP_X_0],
                [_STEP_S_PLUS_0, _STEP_S_PLUS_1, _STEP_S_MINUS_1],
            ),
        )

    def caekl_causal_information(self) -> Any:
        """J[S⁺₀ : X₀ : S⁻₁] — CAEKL mutual information among past, present, and future.

        Chan-AlBashabsheh-Ebrahimi-Kaced-Liu multivariate mutual information
        (:cite:`chan2015multivariate`) over the information-anatomy triple
        (:cite:`James2013`): the forward causal state S⁺₀ (past), the present
        symbol X₀, and the reverse causal state S⁻₁ (future). Finite and
        closed-form since the causal states are finite sufficient statistics of
        the semi-infinite past and future.
        """
        _require_dit()
        from dit.multivariate import caekl_mutual_information

        dist = self.step_distribution()
        return _measure_value(
            dist,
            caekl_mutual_information(dist, rvs=[[_STEP_S_PLUS_0], [_STEP_X_0], [_STEP_S_MINUS_1]]),
        )

    def excess_entropy(self) -> Any:
        """Exact excess entropy E = I[S⁺; S⁻] from the bidirectional joint distribution."""
        dit = _require_dit()
        joint = self.joint_distribution()
        if not joint:
            return 0.0

        from sofic.generators.prob import as_prob, has_symbolic, sum_probs

        pi_plus: dict[Any, Any] = {}
        pi_minus: dict[Any, Any] = {}
        for (alpha, gamma), mass in joint.items():
            pi_plus[alpha] = sum_probs([pi_plus.get(alpha, 0), mass])
            pi_minus[gamma] = sum_probs([pi_minus.get(gamma, 0), mass])

        plus_outcomes = list(pi_plus.keys())
        minus_outcomes = list(pi_minus.keys())
        joint_outcomes = list(joint.keys())
        plus_pmf = [as_prob(pi_plus[s]) for s in plus_outcomes]
        minus_pmf = [as_prob(pi_minus[s]) for s in minus_outcomes]
        joint_pmf = [as_prob(joint[outcome]) for outcome in joint_outcomes]
        if has_symbolic(joint_pmf):
            from dit.symbolic import symbolic_distribution

            plus_dist = symbolic_distribution(plus_outcomes, plus_pmf)
            minus_dist = symbolic_distribution(minus_outcomes, minus_pmf)
            joint_dist = symbolic_distribution(joint_outcomes, joint_pmf)
            return (
                dit.shannon.entropy(plus_dist)
                + dit.shannon.entropy(minus_dist)
                - dit.shannon.entropy(joint_dist)
            )
        plus_dist = dit.Distribution(plus_outcomes, plus_pmf)
        minus_dist = dit.Distribution(minus_outcomes, minus_pmf)
        joint_dist = dit.Distribution(joint_outcomes, joint_pmf)
        return float(
            dit.shannon.entropy(plus_dist) + dit.shannon.entropy(minus_dist) - dit.shannon.entropy(joint_dist)
        )

    def statistical_complexity(self) -> Any:
        """C± = H[S⁺, S⁻] under the bidirectional stationary distribution."""
        dit = _require_dit()
        joint = self.joint_distribution()
        if not joint:
            return 0.0
        outcomes = list(joint.keys())
        probs = [joint[outcome] for outcome in outcomes]
        from sofic.generators.prob import has_symbolic

        if has_symbolic(probs):
            from dit.symbolic import symbolic_distribution

            return dit.shannon.entropy(symbolic_distribution(outcomes, probs))
        return float(dit.shannon.entropy(dit.Distribution(outcomes, probs)))

    def crypticity(self) -> Any:
        """χ = C± − E for a bidirectional presentation."""
        return self.statistical_complexity() - self.excess_entropy()

    def minimal_generative_model(self, **kwargs: Any) -> Any:
        """Construct the minimal-state-entropy generative presentation."""
        from sofic.generators.minimal_generative_model import minimal_generative_model

        return minimal_generative_model(self, **kwargs)

    def wyner_generative_model(self, **kwargs: Any) -> Any:
        """Construct the Wyner-common-information generative presentation."""
        from sofic.generators.minimal_generative_model import wyner_generative_model

        return wyner_generative_model(self, **kwargs)

    def functional_generative_model(self, **kwargs: Any) -> Any:
        """Construct the functional-common-information generative presentation."""
        from sofic.generators.minimal_generative_model import functional_generative_model

        return functional_generative_model(self, **kwargs)

    def gacs_korner_generative_model(self, **kwargs: Any) -> Any:
        """Construct the Gács-Körner (deterministic meet) generative presentation."""
        from sofic.generators.minimal_generative_model import gacs_korner_generative_model

        return gacs_korner_generative_model(self, **kwargs)

    def generative_complexity(self, **kwargs: Any) -> float:
        """C_g = H[G] for the minimal generative model."""
        return self.minimal_generative_model(**kwargs).generative_complexity()

    def information_anatomy(self) -> dict[str, Any]:
        """Return ρ_μ, b_μ, r_μ, h_μ, E, χ, and the structural/gauge refinement.

        The ``*_structural`` / ``*_gauge`` keys split the ephemeral (r_μ) and
        bound (b_μ) rates along whether the randomness changes the next forward
        causal state (structural) or merely relabels the output on a fixed
        transition (gauge); each pair sums to its parent.
        """
        h_mu = self.entropy_rate()
        return {
            "rho_mu": self.predicted_information(),
            "bound_mu": self.bound_information(),
            "ephemeral_mu": self.ephemeral_information(),
            "ephemeral_structural": self.structural_ephemeral_information(),
            "ephemeral_gauge": self.parallel_edge_information(),
            "bound_structural": self.bound_structural_information(),
            "bound_gauge": self.bound_parallel_edge_information(),
            "entropy_rate": h_mu,
            "excess_entropy": self.excess_entropy(),
            "crypticity": self.crypticity(),
        }
