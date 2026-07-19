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

    def reverse_structural_ephemeral_information(self) -> Any:
        """r̄_μ^struct = H[S⁻₀ | S⁺₀, S⁻₁] — reverse structural (branching) part of r_μ.

        Time-reversed mirror of :meth:`structural_ephemeral_information`. The
        previous reverse causal state S⁻₀ is a deterministic function of S⁻₁ and
        X₀, so this equals I[X₀ : S⁻₀ | S⁺₀, S⁻₁]: the ephemeral randomness that
        the *past* cannot foresee yet which selects among *different* retrodictive
        (reverse) states — the branch the reverse ε-machine must resolve. It
        partitions as r̄_μ^struct = r_μ^rev + r_μ^joint (see
        :meth:`reverse_only_structural_ephemeral` and
        :meth:`joint_structural_ephemeral`).

        Refinement of the bidirectional information taxonomy
        (:cite:`jurgens2026taxonomy`; James et al., 2013); follows from the
        determinism of the reverse transition function.
        """
        dit = _require_dit()
        dist = self.step_distribution()
        return _measure_value(
            dist,
            dit.shannon.conditional_entropy(
                dist,
                [_STEP_S_MINUS_0],
                [_STEP_S_PLUS_0, _STEP_S_MINUS_1],
            ),
        )

    def forward_only_structural_ephemeral(self) -> Any:
        """r_μ^fwd = H[S⁺₁ | S⁺₀, S⁻₁, S⁻₀] — forward-only ephemeral branch.

        One of the four atoms of the five-variable ephemeral partition
        r_μ = r_μ^fwd + r_μ^rev + r_μ^joint + r_μ^gauge. It is the present
        randomness that changes the *next forward* causal state S⁺₁ while leaving
        the *previous reverse* causal state S⁻₀ (and the future S⁻₁) unresolved:
        a branch the future forgets but that the reverse presentation never even
        sees. In the taxonomy of :cite:`jurgens2026taxonomy` this is the
        persistent forward ephemeral rate ``p.r⁺_μ``.
        """
        dit = _require_dit()
        dist = self.step_distribution()
        return _measure_value(
            dist,
            dit.shannon.conditional_entropy(
                dist,
                [_STEP_S_PLUS_1],
                [_STEP_S_PLUS_0, _STEP_S_MINUS_1, _STEP_S_MINUS_0],
            ),
        )

    def reverse_only_structural_ephemeral(self) -> Any:
        """r_μ^rev = H[S⁻₀ | S⁺₀, S⁻₁, S⁺₁] — reverse-only ephemeral branch.

        Time-reversed mirror of :meth:`forward_only_structural_ephemeral` and one
        of the four atoms of r_μ = r_μ^fwd + r_μ^rev + r_μ^joint + r_μ^gauge. It
        is the present randomness that changes the *previous reverse* causal state
        S⁻₀ while leaving the *next forward* state S⁺₁ (and the future S⁻₁)
        unresolved — the branch only the retrodictor must resolve. In the taxonomy
        of :cite:`jurgens2026taxonomy` this is the persistent reverse ephemeral
        rate ``p.r⁻_μ`` and is a clean arrow-of-time diagnostic: it can be nonzero
        while r_μ^fwd vanishes (e.g. the noisy random phase-slip process).
        """
        dit = _require_dit()
        dist = self.step_distribution()
        return _measure_value(
            dist,
            dit.shannon.conditional_entropy(
                dist,
                [_STEP_S_MINUS_0],
                [_STEP_S_PLUS_0, _STEP_S_MINUS_1, _STEP_S_PLUS_1],
            ),
        )

    def joint_structural_ephemeral(self) -> Any:
        """r_μ^joint = I[S⁺₁ : S⁻₀ | S⁺₀, S⁻₁] — joint ephemeral branch.

        One of the four atoms of r_μ = r_μ^fwd + r_μ^rev + r_μ^joint + r_μ^gauge:
        the present randomness that *simultaneously* selects the next forward
        state S⁺₁ and the previous reverse state S⁻₀ — a single coin flip both
        presentations must branch on but which the future S⁻₁ still forgets. It is
        shared by the forward and reverse structural ephemeral rates
        (r_μ^struct = r_μ^fwd + r_μ^joint and r̄_μ^struct = r_μ^rev + r_μ^joint),
        and equals the persistent bidirectional ephemeral rate ``p.r±_μ`` of
        :cite:`jurgens2026taxonomy`. Nonzero for the golden mean.
        """
        dit = _require_dit()
        dist = self.step_distribution()
        return _measure_value(
            dist,
            dit.shannon.conditional_entropy(dist, [_STEP_S_PLUS_1], [_STEP_S_PLUS_0, _STEP_S_MINUS_1])
            - dit.shannon.conditional_entropy(
                dist,
                [_STEP_S_PLUS_1],
                [_STEP_S_PLUS_0, _STEP_S_MINUS_1, _STEP_S_MINUS_0],
            ),
        )

    def pure_gauge_information(self) -> Any:
        """r_μ^gauge = H[X₀ | S⁺₀, S⁻₁, S⁺₁, S⁻₀] — pure-gauge ephemeral branch.

        The fourth atom of r_μ = r_μ^fwd + r_μ^rev + r_μ^joint + r_μ^gauge: once
        the source S⁺₀, both next states S⁺₁ and S⁻₀, and the future S⁻₁ are all
        fixed, the residual symbol uncertainty is pure output relabeling on edges
        with *no* structural consequence in either time direction — parallel edges
        from the same state to the same state. It is the transient ephemeral rate
        ``t.r_μ`` of :cite:`jurgens2026taxonomy` and is the whole of r_μ for an
        i.i.d. fair coin. Sharper than :meth:`parallel_edge_information`, which
        still lumps in the reverse-only branch (r_μ^par = r_μ^rev + r_μ^gauge).
        """
        dit = _require_dit()
        dist = self.step_distribution()
        return _measure_value(
            dist,
            dit.shannon.conditional_entropy(
                dist,
                [_STEP_X_0],
                [_STEP_S_PLUS_0, _STEP_S_MINUS_1, _STEP_S_PLUS_1, _STEP_S_MINUS_0],
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

    def reverse_bound_information(self) -> Any:
        """b̄_μ = I[X₀ : S⁺₀ | S⁻₁] — reverse (retrodictive) bound information.

        Time-reversed mirror of :meth:`bound_information`: the present information
        shared with the *past* causal state S⁺₀ but not already carried by the
        future S⁻₁. Bound information is time-reversal invariant, so b̄_μ = b_μ
        (:cite:`James2011`); this method computes it from the reverse triple and
        the equality is asserted in the test-suite as a symmetry check.
        """
        dit = _require_dit()
        dist = self.step_distribution()
        return _measure_value(
            dist,
            dit.shannon.conditional_entropy(dist, [_STEP_X_0], [_STEP_S_MINUS_1])
            - dit.shannon.conditional_entropy(
                dist,
                [_STEP_X_0],
                [_STEP_S_MINUS_1, _STEP_S_PLUS_0],
            ),
        )

    def reverse_bound_structural_information(self) -> Any:
        """b̄_μ^struct = I[S⁻₀ : S⁺₀ | S⁻₁] — structural part of the reverse bound.

        Time-reversed mirror of :meth:`bound_structural_information`: the branch
        that changes the previous reverse causal state S⁻₀ *and* is shared with
        the past S⁺₀. By Theorem A′ the reverse bound has no gauge part
        (:meth:`reverse_bound_gauge_information` ≈ 0), so b̄_μ^struct = b̄_μ = b_μ.

        Refinement of the bidirectional information taxonomy
        (:cite:`jurgens2026taxonomy`; James et al., 2013).
        """
        dit = _require_dit()
        dist = self.step_distribution()
        return _measure_value(
            dist,
            dit.shannon.conditional_entropy(dist, [_STEP_S_MINUS_0], [_STEP_S_MINUS_1])
            - dit.shannon.conditional_entropy(
                dist,
                [_STEP_S_MINUS_0],
                [_STEP_S_MINUS_1, _STEP_S_PLUS_0],
            ),
        )

    def reverse_bound_gauge_information(self) -> Any:
        """b̄_μ^gauge = I[X₀ : S⁺₀ | S⁻₁, S⁻₀] — reverse bound-gauge (Theorem A′).

        Time-reversed mirror of :meth:`bound_parallel_edge_information`. Theorem A′
        (the reverse of James et al.'s bound-gauge vanishing, Theorem A) asserts
        this is identically zero: once the previous reverse state S⁻₀ is fixed,
        the present symbol X₀ carries nothing further about the past S⁺₀ that the
        future S⁻₁ did not already supply. Computed here to verify the theorem
        numerically; the test-suite pins it to ≈ 0.

        Refinement of the bidirectional information taxonomy
        (:cite:`jurgens2026taxonomy`; James et al., 2013).
        """
        dit = _require_dit()
        dist = self.step_distribution()
        return _measure_value(
            dist,
            dit.shannon.conditional_entropy(dist, [_STEP_X_0], [_STEP_S_MINUS_1, _STEP_S_MINUS_0])
            - dit.shannon.conditional_entropy(
                dist,
                [_STEP_X_0],
                [_STEP_S_MINUS_1, _STEP_S_MINUS_0, _STEP_S_PLUS_0],
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
            return dit.shannon.entropy(plus_dist) + dit.shannon.entropy(minus_dist) - dit.shannon.entropy(joint_dist)
        plus_dist = dit.Distribution(plus_outcomes, plus_pmf)
        minus_dist = dit.Distribution(minus_outcomes, minus_pmf)
        joint_dist = dit.Distribution(joint_outcomes, joint_pmf)
        return float(dit.shannon.entropy(plus_dist) + dit.shannon.entropy(minus_dist) - dit.shannon.entropy(joint_dist))

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

        See :meth:`five_variable_anatomy` for the finer four-atom ephemeral
        partition and the reverse-time bound mirror over the full joint
        ``Pr(S⁺₀, S⁻₀, X₀, S⁺₁, S⁻₁)``.
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

    def five_variable_anatomy(self) -> dict[str, Any]:
        """Full five-variable anatomy over ``Pr(S⁺₀, S⁻₀, X₀, S⁺₁, S⁻₁)``.

        Extends :meth:`information_anatomy` with the two unifilarity relations
        (forward ``S⁺₁ = φ⁺(S⁺₀, X₀)`` and reverse ``S⁻₀ = φ⁻(S⁻₁, X₀)``) that let
        both next-states be read off the present. This yields:

        * the four-atom ephemeral partition
          ``r_μ = r_μ^fwd + r_μ^rev + r_μ^joint + r_μ^gauge`` (keys
          ``ephemeral_forward``, ``ephemeral_reverse``, ``ephemeral_joint``,
          ``ephemeral_pure_gauge``), all ≥ 0, refining the coarse
          structural/gauge split (``r_μ^struct = r_μ^fwd + r_μ^joint``,
          ``r_μ^par = r_μ^rev + r_μ^gauge``);
        * the reverse structural ephemeral rate
          ``r̄_μ^struct = r_μ^rev + r_μ^joint`` (key
          ``ephemeral_structural_reverse``);
        * the reverse-time bound mirror ``b̄_μ``, ``b̄_μ^struct`` and the
          Theorem-A′ gauge residual ``b̄_μ^gauge`` (≈ 0).

        These atoms map onto the fourteen measures of the prediction taxonomy of
        :cite:`jurgens2026taxonomy`; the structural/gauge (unifilarity) reading is
        the refinement layered on top.
        """
        anatomy = self.information_anatomy()
        anatomy.update(
            {
                "ephemeral_forward": self.forward_only_structural_ephemeral(),
                "ephemeral_reverse": self.reverse_only_structural_ephemeral(),
                "ephemeral_joint": self.joint_structural_ephemeral(),
                "ephemeral_pure_gauge": self.pure_gauge_information(),
                "ephemeral_structural_reverse": self.reverse_structural_ephemeral_information(),
                "bound_reverse": self.reverse_bound_information(),
                "bound_structural_reverse": self.reverse_bound_structural_information(),
                "bound_gauge_reverse": self.reverse_bound_gauge_information(),
            }
        )
        return anatomy
