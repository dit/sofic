"""Stochastic generator base classes."""

from __future__ import annotations

from collections.abc import Hashable, Mapping, Sequence
from typing import TYPE_CHECKING, Any, Self

import numpy as np

from pensive.base import StateMachine
from pensive.exceptions import QuasiStochasticValidationError, StochasticValidationError

if TYPE_CHECKING:
    from pensive.automata.dfa import DFA
    from pensive.automata.nfa import NFA
    from pensive.generators.mealy import MealyHMM
    from pensive.generators.prob import SymbolConstraints
    from pensive.shifts.sofic import SoficShift


class StochasticModel(StateMachine):
    """Generator with a probability distribution over initial states."""

    initial_distribution: dict[Hashable, float]
    symbol_constraints: SymbolConstraints | None

    def __init__(
        self,
        initial_distribution: Mapping[Hashable, float] | None = None,
        *,
        symbol_constraints: SymbolConstraints | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.initial_distribution = dict(initial_distribution or {})
        self.symbol_constraints = symbol_constraints

    def validate(self) -> None:
        self.validate_stochastic()

    def validate_stochastic(self) -> None:
        from pensive.generators.prob import is_symbolic, row_sums_to_one

        probs = list(self.initial_distribution.values())
        if probs and not row_sums_to_one(probs):
            total = sum(probs)
            raise StochasticValidationError(f"initial distribution sums to {total}, not 1")
        for state, prob in self.initial_distribution.items():
            if is_symbolic(prob):
                if getattr(prob, "is_negative", None) is True:
                    raise StochasticValidationError(f"negative initial probability at {state!r}")
            elif prob < 0:
                raise StochasticValidationError(f"negative initial probability at {state!r}")
            self._require(self.graph.has_state(state), f"unknown initial state {state!r}")

    def stationary_distribution(self) -> np.ndarray:
        from pensive.generators.stationary import stationary_distribution_hmm

        return stationary_distribution_hmm(self)

    def state_distribution(self) -> Any:
        from pensive.generators.measures import state_distribution

        return state_distribution(self)

    def state_entropy(self) -> float:
        from pensive.generators.measures import state_entropy

        return state_entropy(self)

    def reverse(self) -> Self:
        """Return the time-reversed generator (stochastic reversal, not a graph transpose).

        Overrides :meth:`pensive.base.StateMachine.reverse` (which merely
        transposes the transition graph): for a stochastic process the reversal
        must reweight edges by the stationary distribution
        (:func:`~pensive.generators.reversal.time_reverse_stochastic`). Emission
        machines are routed through :meth:`EpsilonMachine.from_hmm`.
        """
        from pensive.generators.reversal import is_markov_like, time_reverse_stochastic

        if not is_markov_like(self):
            from pensive.generators.epsilon_machine import EpsilonMachine
            from pensive.generators.mealy import MealyHMM
            from pensive.generators.moore import MooreHMM

            if isinstance(self, (MealyHMM, MooreHMM)):
                from pensive.generators.epsilon_machine import EpsilonMachine

                if isinstance(self, EpsilonMachine):
                    return EpsilonMachine.from_time_reversed(self)
                return EpsilonMachine.from_hmm(time_reverse_stochastic(self))
            raise NotImplementedError("time-reversed generators with edge emissions require EpsilonMachine.from_hmm")
        return time_reverse_stochastic(self)


class HiddenMarkovModel(StochasticModel):
    """Hidden-state generator with an observation alphabet."""

    observation_alphabet: frozenset[Any]

    def __init__(self, observation_alphabet: frozenset[Any] | None = None, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.observation_alphabet = observation_alphabet if observation_alphabet is not None else frozenset()

    def sample(self, n: int, rng: np.random.Generator | None = None) -> tuple[list[Any], list[Hashable]]:
        from pensive.generators.hmm_inference import sample

        return sample(self, n, rng)

    def log_likelihood(self, observations: Sequence[Any]) -> float:
        from pensive.generators.hmm_inference import log_likelihood

        return log_likelihood(self, observations)

    def forward(self, observations: Sequence[Any], *, scaled: bool = False) -> np.ndarray:
        from pensive.generators.hmm_inference import forward

        return forward(self, observations, scaled=scaled)

    def backward(self, observations: Sequence[Any], *, scaled: bool = False) -> np.ndarray:
        from pensive.generators.hmm_inference import backward

        return backward(self, observations, scaled=scaled)

    def viterbi(self, observations: Sequence[Any]) -> list[Hashable]:
        from pensive.generators.hmm_inference import viterbi

        return viterbi(self, observations)

    def to_mealy(self) -> MealyHMM:
        """Return an equivalent Mealy-style presentation."""
        raise NotImplementedError(f"{type(self).__name__} must implement to_mealy()")

    def entropy_rate(self) -> float:
        from pensive.generators.measures import entropy_rate_hmm

        return entropy_rate_hmm(self)

    def joint_block_distribution(self, history_length: int = 1) -> Any:
        from pensive.generators.measures import joint_block_distribution

        return joint_block_distribution(self, history_length=history_length)

    def words_of_length(self, length: int) -> dict[tuple[Any, ...], float]:
        """Return observed words of ``length`` and their probabilities."""
        from pensive.generators.words import hmm_words_of_length

        return hmm_words_of_length(self, length)

    def word_probability(
        self,
        word: Sequence[Any],
        *,
        start: Hashable | Mapping[Hashable, float] | Sequence[float] | np.ndarray | None = None,
    ) -> float:
        """Return the probability of an observed finite word."""
        from pensive.generators.words import hmm_word_probability

        return hmm_word_probability(self, word, start=start)

    def log_word_probability(
        self,
        word: Sequence[Any],
        *,
        start: Hashable | Mapping[Hashable, float] | Sequence[float] | np.ndarray | None = None,
    ) -> float:
        """Return ``log2`` of an observed finite-word probability."""
        from pensive.generators.words import hmm_log_word_probability

        return hmm_log_word_probability(self, word, start=start)

    def word_probabilities(
        self,
        lengths: int | Sequence[int],
        *,
        start: Hashable | Mapping[Hashable, float] | Sequence[float] | np.ndarray | None = None,
        sparse: bool = True,
    ) -> dict[tuple[Any, ...], float]:
        """Return observed-word probabilities for one or more lengths."""
        from pensive.generators.words import hmm_word_probabilities

        return hmm_word_probabilities(self, lengths, start=start, sparse=sparse)

    def conditional_word_probability(
        self,
        word: Sequence[Any],
        condition: Sequence[Any],
        *,
        start: Hashable | Mapping[Hashable, float] | Sequence[float] | np.ndarray | None = None,
    ) -> float:
        """Return ``P(word | condition)``."""
        from pensive.generators.words import hmm_conditional_word_probability

        return hmm_conditional_word_probability(self, word, condition, start=start)

    def is_equal_process(
        self,
        other: HiddenMarkovModel,
        *,
        start1: Hashable | Mapping[Hashable, float] | Sequence[float] | np.ndarray | None = None,
        start2: Hashable | Mapping[Hashable, float] | Sequence[float] | np.ndarray | None = None,
        rtol: float | None = None,
        atol: float | None = None,
    ) -> bool:
        """Return whether two HMMs generate the same finite-word process."""
        from pensive.generators.process_equivalence import is_equal_process

        return is_equal_process(self, other, start1=start1, start2=start2, rtol=rtol, atol=atol)

    def to_sofic_shift(self) -> SoficShift:
        """Strip probabilities and return a sofic shift with the same support."""
        from pensive.generators.conversions import hmm_to_sofic_shift

        return hmm_to_sofic_shift(self)

    def to_support_nfa(self) -> NFA:
        """Return an NFA for the support language of this HMM."""
        from pensive.generators.conversions import hmm_to_support_nfa

        return hmm_to_support_nfa(self)

    def to_support_dfa(self) -> DFA:
        """Return a determinized automaton for the support language of this HMM."""
        from pensive.generators.conversions import hmm_to_support_dfa

        return hmm_to_support_dfa(self)


class QuasiStochasticModel(StateMachine):
    """Generator allowing signed quasiprobabilities on internal weights."""

    initial_quasidistribution: dict[Hashable, float]

    def __init__(self, initial_quasidistribution: Mapping[Hashable, float] | None = None, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.initial_quasidistribution = dict(initial_quasidistribution or {})

    def validate(self) -> None:
        self.validate_quasistochastic()

    def validate_quasistochastic(self) -> None:
        total = sum(self.initial_quasidistribution.values())
        if not np.isclose(total, 1.0):
            raise QuasiStochasticValidationError(f"initial quasidistribution sums to {total}, not 1")
        for state in self.initial_quasidistribution:
            self._require(self.graph.has_state(state), f"unknown initial state {state!r}")

    def word_probability(self, word: Sequence[Any]) -> float:
        from pensive.generators.quasi_inference import word_probability

        return word_probability(self, word)

    def stationary_quasidistribution(self) -> np.ndarray:
        from pensive.generators.quasi_inference import stationary_quasidistribution

        return stationary_quasidistribution(self)

    def transition_matrices(self) -> dict[Any, np.ndarray]:
        from pensive.generators.quasi_inference import transition_matrices

        return transition_matrices(self)

    def words_of_length(self, length: int) -> dict[tuple[Any, ...], float]:
        """Return words of ``length`` and their signed quasiprobabilities."""
        from pensive.generators.words import quasi_words_of_length

        return quasi_words_of_length(self, length)

    def collision_entropy(self) -> float:
        from pensive.generators.measures import collision_entropy

        return collision_entropy(self)

    def process_negativity(self) -> float:
        from pensive.generators.measures import process_negativity

        return process_negativity(self)
