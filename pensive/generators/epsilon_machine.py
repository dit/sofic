"""Epsilon machines (unifilar causal presentations)."""

from __future__ import annotations

from collections.abc import Hashable, Sequence
from typing import TYPE_CHECKING, Any, Literal, Self

from pensive.generators.mealy import MealyHMM
from pensive.generators.moore import MooreHMM

if TYPE_CHECKING:
    from pensive.generators.bidirectional_epsilon_machine import BidirectionalEpsilonMachine
    from pensive.generators.block_entropy import BlockEntropyDiagram
    from pensive.generators.minimal_generative_model import (
        FunctionalGenerativeModel,
        GacsKornerGenerativeModel,
        MinimalGenerativeModel,
        WynerGenerativeModel,
    )


class EpsilonMachine(MealyHMM):
    """Unifilar Mealy HMM: minimal causal presentation of a stationary process.

    Validates row-unifilarity on emissions and exposes computational mechanics
    quantities (entropy rate, statistical complexity, synchronization orders).

    Examples
    --------
    >>> from pensive.examples import golden_mean
    >>> eps = golden_mean(0.5)
    >>> eps.markov_order()
    1
    """

    _bidirectional_machine: BidirectionalEpsilonMachine | None = None
    _bidirectional_machine_fingerprint: tuple[Any, ...] | None = None

    def validate(self) -> None:
        super().validate()
        self._check_unifilar()

    @classmethod
    def from_hmm(cls, hmm: MealyHMM | MooreHMM, **kwargs: Any) -> EpsilonMachine:
        from pensive.generators.epsilon_construction import build_epsilon_machine

        return build_epsilon_machine(hmm)

    @classmethod
    def from_sequence(
        cls,
        sequence: Sequence[Any],
        *,
        method: Literal["cssr", "subtree"] = "cssr",
        **kwargs: Any,
    ) -> EpsilonMachine:
        """Reconstruct an ε-machine from an observed symbol sequence.

        Parameters
        ----------
        sequence
            Observed process realization.
        method
            ``"cssr"`` for Causal-State Splitting Reconstruction, or
            ``"subtree"`` for depth-``L`` subtree merging (pass ``L=...``).
        **kwargs
            Forwarded to :func:`~pensive.generators.epsilon_inference.cssr` or
            :func:`~pensive.generators.epsilon_inference.subtree_merge`.
        """
        if method == "cssr":
            from pensive.generators.epsilon_inference import cssr

            return cssr(sequence, **kwargs)
        if method == "subtree":
            from pensive.generators.epsilon_inference import subtree_merge

            return subtree_merge(sequence, **kwargs)
        raise ValueError(f"unknown inference method {method!r}")

    def copy(self) -> Self:
        cloned = super().copy()
        cloned._bidirectional_machine = None
        cloned._bidirectional_machine_fingerprint = None
        return cloned

    def invalidate_bidirectional_cache(self) -> None:
        """Clear the cached bidirectional presentation."""
        self._bidirectional_machine = None
        self._bidirectional_machine_fingerprint = None

    def to_bidirectional(self) -> BidirectionalEpsilonMachine:
        """Return the bidirectional presentation, building and caching on first use."""
        fingerprint = self._bidirectional_cache_fingerprint()
        if self._bidirectional_machine is None or self._bidirectional_machine_fingerprint != fingerprint:
            from pensive.generators.bidirectional_epsilon_machine import BidirectionalEpsilonMachine

            self._bidirectional_machine = BidirectionalEpsilonMachine.from_forward(self)
            self._bidirectional_machine_fingerprint = fingerprint
        return self._bidirectional_machine

    def _bidirectional_cache_fingerprint(self) -> tuple[Any, ...]:
        states = tuple(
            (
                repr(state),
                tuple(sorted((repr(key), repr(value)) for key, value in self.graph.state_attrs(state).items())),
            )
            for state in sorted(self.states(), key=repr)
        )
        transitions = tuple(
            sorted(
                (
                    repr(transition.source),
                    repr(transition.target),
                    tuple(sorted((repr(key), repr(value)) for key, value in transition.data.items())),
                )
                for transition in self.transitions()
            )
        )
        initial = tuple(sorted((repr(state), float(mass)) for state, mass in self.initial_distribution.items()))
        alphabet = tuple(sorted(repr(symbol) for symbol in self.observation_alphabet))
        return (states, transitions, initial, alphabet)

    def statistical_complexity(self) -> float:
        """C_mu = H[causal state] under the stationary distribution."""
        return self.state_entropy()

    def bidirectional_statistical_complexity(self) -> float:
        """C± = H[S⁺, S⁻] under the bidirectional stationary distribution."""
        return self.to_bidirectional().statistical_complexity()

    def excess_entropy(self) -> float:
        """Excess entropy ``E`` from the bidirectional machine when available."""
        from pensive.generators.block_entropy import _excess_entropy

        return _excess_entropy(self)

    def predicted_information(self) -> float:
        """ρ_μ = I[X₀ : S⁺₀] — predicted information rate (James et al., 2013)."""
        return self.to_bidirectional().predicted_information()

    def bound_information(self) -> float:
        """b_μ = I[X₀ : S⁻₁ | S⁺₀] — bound information rate (James et al., 2013)."""
        return self.to_bidirectional().bound_information()

    def ephemeral_information(self) -> float:
        """r_μ = H[X₀ | S⁺₀, S⁻₁] — ephemeral information rate (James et al., 2013)."""
        return self.to_bidirectional().ephemeral_information()

    def information_anatomy(self) -> dict[str, float]:
        """Return ρ_μ, b_μ, r_μ, h_μ, E, and bidirectional χ for this ε-machine."""
        return self.to_bidirectional().information_anatomy()

    def caekl_causal_information(self) -> float:
        """J[S⁺₀ : X₀ : S⁻₁] — CAEKL mutual info among past, present, and future causal states."""
        return self.to_bidirectional().caekl_causal_information()

    def block_entropy_diagram(self, max_length: int) -> BlockEntropyDiagram:
        """Compute finite-block entropy convergence curves up to ``max_length``."""
        from pensive.generators.block_entropy import block_entropy_diagram

        return block_entropy_diagram(self, max_length)

    def block_entropy_estimates(
        self,
        max_length: int,
        *,
        entropy_rate: float | None = None,
        use_exact: bool = True,
    ) -> Any:
        """Approximate information quantities from finite block entropies.

        With ``use_exact`` (the default) the asymptotic ``h_mu`` and excess entropy
        use the exact closed-form/bidirectional values, matching the diagram and
        block-convergence entry points. Pass ``use_exact=False`` for genuinely
        finite-length estimates (see :meth:`approximate_entropy_rate`).
        """
        from pensive.generators.block_entropy import block_entropy_estimates

        return block_entropy_estimates(self, max_length, entropy_rate=entropy_rate, use_exact=use_exact)

    def approximate_entropy_rate(self, max_length: int) -> float:
        """Approximate ``h_mu`` as the last finite-block entropy difference."""
        return self.block_entropy_estimates(max_length, use_exact=False).entropy_rate

    def approximate_excess_entropy(self, max_length: int, *, entropy_rate: float | None = None) -> float:
        """Approximate ``E`` from finite-block entropy lower/upper estimates."""
        return self.block_entropy_estimates(max_length, entropy_rate=entropy_rate, use_exact=False).excess_entropy

    def approximate_information_anatomy(
        self,
        max_length: int,
        *,
        entropy_rate: float | None = None,
    ) -> dict[str, float]:
        """Approximate anatomy rates without constructing a bidirectional machine."""
        return self.block_convergence_estimates(max_length, entropy_rate=entropy_rate).information_anatomy()

    def plot_block_entropy_diagram(self, max_length: int, ax: Any | None = None, **kwargs: Any) -> Any:
        """Compute and plot finite-block entropy convergence curves."""
        from pensive.generators.block_entropy import plot_block_entropy_diagram

        return plot_block_entropy_diagram(self, max_length, ax=ax, **kwargs)

    def block_convergence_diagram(self, max_length: int, **kwargs: Any) -> Any:
        """James et al. (2011) block convergence curves up to ``max_length``."""
        from pensive.generators.block_convergence import block_convergence_diagram

        return block_convergence_diagram(self, max_length, **kwargs)

    def block_convergence_estimates(
        self,
        max_length: int,
        *,
        entropy_rate: float | None = None,
        use_exact: bool = True,
        max_caekl_length: int | None = None,
    ) -> Any:
        """Finite-block anatomy estimates including TC, DTC, coinformation, and CAEKL."""
        from pensive.generators.block_convergence import block_convergence_estimates

        return block_convergence_estimates(
            self,
            max_length,
            entropy_rate=entropy_rate,
            use_exact=use_exact,
            max_caekl_length=max_caekl_length,
        )

    def caekl_block_information(self, length: int) -> float:
        """Block CAEKL mutual information ``J(ℓ)`` from exact word probabilities."""
        from pensive.generators.block_convergence import block_caekl

        return block_caekl(self, length)

    def caekl_rate(self, max_length: int, **kwargs: Any) -> float:
        """Asymptotic CAEKL rate ``j_μ`` from finite-block convergence.

        When :attr:`~pensive.generators.block_convergence.BlockConvergenceEstimates.caekl_rate_converged`
        is ``True``, the returned rate equals ``J(ℓ) - J(ℓ-1)`` for all sufficiently
        large ``ℓ`` in the affine tail.
        """
        return float(self.block_convergence_estimates(max_length, **kwargs).caekl_rate)

    def caekl_intercept(self, max_length: int, **kwargs: Any) -> float:
        """Subextensive intercept ``J_∞`` in ``J(ℓ) ≈ J_∞ + j_μ ℓ``."""
        return float(self.block_convergence_estimates(max_length, **kwargs).caekl_intercept_scalar)

    def caekl_rate_converged(self, max_length: int, **kwargs: Any) -> bool:
        """Whether ``j_μ`` is certified from a stable affine tail of ``J(ℓ)``."""
        return bool(self.block_convergence_estimates(max_length, **kwargs).caekl_rate_converged)

    def plot_block_convergence_diagram(self, max_length: int, ax: Any | None = None, **kwargs: Any) -> Any:
        """Compute and plot James et al. (2011) block convergence curves."""
        from pensive.generators.block_convergence import plot_block_convergence_diagram

        return plot_block_convergence_diagram(self, max_length, ax=ax, **kwargs)

    def bidirectional_crypticity(self) -> float:
        """χ = C± − E (bidirectional statistical complexity minus excess entropy)."""
        return self.to_bidirectional().crypticity()

    def minimal_generative_model(self, **kwargs: Any) -> MinimalGenerativeModel:
        """Construct the minimal-state-entropy generative presentation."""
        return self.to_bidirectional().minimal_generative_model(**kwargs)

    def wyner_generative_model(self, **kwargs: Any) -> WynerGenerativeModel:
        """Construct the Wyner-common-information generative presentation."""
        return self.to_bidirectional().wyner_generative_model(**kwargs)

    def functional_generative_model(self, **kwargs: Any) -> FunctionalGenerativeModel:
        """Construct the functional-common-information generative presentation."""
        return self.to_bidirectional().functional_generative_model(**kwargs)

    def gacs_korner_generative_model(self, **kwargs: Any) -> GacsKornerGenerativeModel:
        """Construct the Gács-Körner (deterministic meet) generative presentation."""
        return self.to_bidirectional().gacs_korner_generative_model(**kwargs)

    def generative_complexity(self, **kwargs: Any) -> float:
        """C_g = H[G] for the minimal generative model."""
        return self.to_bidirectional().generative_complexity(**kwargs)

    def crypticity(self) -> float:
        """χ = C_μ − E (forward statistical complexity minus excess entropy)."""
        return self.statistical_complexity() - self.excess_entropy()

    def causal_irreversibility(self) -> float:
        """ΔC_μ = C_μ − C_μ^rev (Crutchfield et al., 2009)."""
        reverse = self.from_time_reversed(self)
        return self.statistical_complexity() - reverse.statistical_complexity()

    def stored_information_decomposition(self) -> dict[str, float]:
        """Bidirectional stored-information quantities (Ellison et al., 2009)."""
        bidir = self.to_bidirectional()
        forward = self.statistical_complexity()
        reverse = self.from_time_reversed(self).statistical_complexity()
        bidirectional = bidir.statistical_complexity()
        return {
            "forward_complexity": forward,
            "reverse_complexity": reverse,
            "bidirectional_complexity": bidirectional,
            "causal_irreversibility": forward - reverse,
            "excess_entropy": bidir.excess_entropy(),
            "crypticity": bidir.crypticity(),
        }

    def transient_information(self, max_length: int) -> float:
        """Finite-block transient information TI(L) (Ellison et al., 2009)."""
        estimates = self.block_entropy_estimates(max_length)
        return float(estimates.transient_information_estimate[max_length])

    def oracular_information(self, max_length: int) -> float:
        """Finite-block oracular information Ω(L) (Ellison et al., 2009)."""
        estimates = self.block_entropy_estimates(max_length)
        return float(estimates.oracular_information_estimate[max_length])

    def gauge_information(self, max_length: int) -> float:
        """Finite-block gauge information Γ(L) (Ellison et al., 2009)."""
        estimates = self.block_entropy_estimates(max_length)
        return float(estimates.gauge_information_estimate[max_length])

    def predictability_gain(self, max_length: int) -> float:
        """Finite-block predictability gain ``PG(L) = h_mu(L) - h_mu`` (Bialek et al., 2001)."""
        if max_length < 1:
            raise ValueError("max_length must be at least 1 for predictability gain")
        estimates = self.block_entropy_estimates(max_length)
        return float(estimates.predictability_gain_estimate[max_length])

    def structural_information(self) -> float:
        """Asymptotic structural information (Feldman & Crutchfield, 1998)."""
        from pensive.generators.alternative_complexity import structural_information

        return structural_information(self)

    def thermodynamic_depth(self) -> float:
        """Thermodynamic depth of causal states (Shalizi & Crutchfield, 1999)."""
        from pensive.generators.alternative_complexity import thermodynamic_depth

        return thermodynamic_depth(self)

    def spectral_complexity(self) -> float:
        """Spectral entropy of mixed-state transition eigenvalues (Riechers & Crutchfield, 2017)."""
        from pensive.generators.alternative_complexity import spectral_complexity

        return spectral_complexity(self)

    def markov_order(self) -> int | float:
        """Markov order ``R``: longest prefix-free synchronizing word length.

        Topological (probability-independent); see James et al., arXiv:1010.5545.
        Returns ``math.inf`` when no finite ``R`` exists.
        """
        from pensive.generators.synchronization import graph_from_epsilon_machine, markov_order_from_graph

        return markov_order_from_graph(graph_from_epsilon_machine(self))

    def is_markov(self) -> bool:
        """Return whether the process has finite Markov order."""
        import math

        order = self.markov_order()
        return not isinstance(order, float) or math.isfinite(order)

    def cryptic_order(self) -> int | float:
        """Cryptic order ``k_chi``: retrodiction depth after synchronization.

        Distinct from :meth:`crypticity` (``χ = C_μ − E``) and from
        :meth:`~pensive.generators.bidirectional_epsilon_machine.BidirectionalEpsilonMachine.crypticity`
        (``χ = C± − E``). See James et al., arXiv:1010.5545.
        """
        from pensive.generators.synchronization import cryptic_order_from_graph, graph_from_epsilon_machine

        return cryptic_order_from_graph(graph_from_epsilon_machine(self))

    def is_exactly_synchronizable(self) -> bool:
        """Return whether the presentation has finite Markov order."""
        from pensive.generators.synchronization import graph_from_epsilon_machine, is_exactly_synchronizable

        return is_exactly_synchronizable(graph_from_epsilon_machine(self))

    @classmethod
    def from_time_reversed(cls, forward: EpsilonMachine) -> EpsilonMachine:
        """Build a reverse ε-machine presentation from ``forward``."""
        from pensive.exceptions import StochasticValidationError, UnifilarityError
        from pensive.generators.reversal import time_reverse_stochastic

        rev_hmm = time_reverse_stochastic(forward)
        try:
            return cls.from_hmm(rev_hmm)
        except (StochasticValidationError, UnifilarityError):
            return _row_normalized_presentation(rev_hmm)


def _row_normalized_presentation(hmm: MealyHMM) -> EpsilonMachine:
    from pensive.generators.stochastic import normalize_row_weights
    from pensive.graph import ATTR_EMISSION, ATTR_PROB, TransitionGraph

    graph = TransitionGraph()
    for state in hmm.states():
        graph.add_state(state)
    for state in hmm.states():
        outgoing = list(hmm.graph.out_transitions(state))
        merged: dict[tuple[Hashable, Any], float] = {}
        for transition in outgoing:
            prob = float(transition.data.get(ATTR_PROB, 0.0))
            emission = transition.data.get(ATTR_EMISSION)
            key = (transition.target, emission)
            merged[key] = merged.get(key, 0.0) + prob
        merged = normalize_row_weights(merged)
        for (target, emission), prob in merged.items():
            attrs = {ATTR_PROB: prob}
            if emission is not None:
                attrs[ATTR_EMISSION] = emission
            graph.add_transition(state, target, **attrs)

    eps = EpsilonMachine(
        graph=graph,
        initial_distribution=dict(hmm.initial_distribution),
        observation_alphabet=hmm.observation_alphabet,
    )
    eps.validate_stochastic()
    return eps
