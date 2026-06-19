"""Epsilon machines (unifilar causal presentations)."""

from __future__ import annotations

from collections.abc import Hashable, Sequence
from typing import TYPE_CHECKING, Any, Literal, Self

from pensive.generators.mealy import MealyHMM
from pensive.generators.moore import MooreHMM

if TYPE_CHECKING:
    from pensive.generators.bidirectional_epsilon_machine import BidirectionalEpsilonMachine


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
    def from_generator(cls, hmm: MealyHMM | MooreHMM, **kwargs: Any) -> EpsilonMachine:
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

    def bidirectional_epsilon_machine(self) -> BidirectionalEpsilonMachine:
        """Return the bidirectional presentation, building and caching on first use."""
        fingerprint = self._bidirectional_cache_fingerprint()
        if self._bidirectional_machine is None or self._bidirectional_machine_fingerprint != fingerprint:
            from pensive.generators.bidirectional_epsilon_machine import BidirectionalEpsilonMachine

            self._bidirectional_machine = BidirectionalEpsilonMachine.from_epsilon_machine(self)
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
        return self.bidirectional_epsilon_machine().statistical_complexity()

    def excess_entropy(self) -> float:
        """Excess entropy E = I[S⁺; S⁻] via the bidirectional ε-machine."""
        return self.bidirectional_epsilon_machine().excess_entropy()

    def predicted_information(self) -> float:
        """ρ_μ = I[X₀ : S⁺₀] — predicted information rate (James et al., 2013)."""
        return self.bidirectional_epsilon_machine().predicted_information()

    def bound_information(self) -> float:
        """b_μ = H[X₀ | S⁺₀, S⁻₁] — bound information rate (James et al., 2013)."""
        return self.bidirectional_epsilon_machine().bound_information()

    def ephemeral_information(self) -> float:
        """r_μ = I[X₀ : S⁻₁ | S⁺₀] — ephemeral information rate (James et al., 2013)."""
        return self.bidirectional_epsilon_machine().ephemeral_information()

    def information_anatomy(self) -> dict[str, float]:
        """Return ρ_μ, b_μ, r_μ, h_μ, E, and bidirectional χ for this ε-machine."""
        return self.bidirectional_epsilon_machine().information_anatomy()

    def bidirectional_crypticity(self) -> float:
        """χ = C± − E (bidirectional statistical complexity minus excess entropy)."""
        return self.bidirectional_epsilon_machine().crypticity()

    def crypticity(self) -> float:
        """χ = C_μ − E (forward statistical complexity minus excess entropy)."""
        return self.statistical_complexity() - self.excess_entropy()

    def markov_order(self) -> int | float:
        """Markov order ``R``: longest prefix-free synchronizing word length.

        Topological (probability-independent); see James et al., arXiv:1010.5545.
        Returns ``math.inf`` when no finite ``R`` exists.
        """
        from pensive.generators.synchronization import graph_from_epsilon_machine, markov_order_from_graph

        return markov_order_from_graph(graph_from_epsilon_machine(self))

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
            return cls.from_generator(rev_hmm)
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
