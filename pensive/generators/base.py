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
    from pensive.shifts.sofic import SoficShift


class StochasticModel(StateMachine):
    """Generator with a probability distribution over initial states."""

    initial_distribution: dict[Hashable, float]

    def __init__(self, initial_distribution: Mapping[Hashable, float] | None = None, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.initial_distribution = dict(initial_distribution or {})

    def validate(self) -> None:
        self.validate_stochastic()

    def validate_stochastic(self) -> None:
        total = sum(self.initial_distribution.values())
        if not np.isclose(total, 1.0):
            raise StochasticValidationError(f"initial distribution sums to {total}, not 1")
        for state, prob in self.initial_distribution.items():
            if prob < 0:
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
        from pensive.generators.reversal import is_markov_like, time_reverse_stochastic

        if not is_markov_like(self):
            from pensive.generators.epsilon_machine import EpsilonMachine
            from pensive.generators.mealy import MealyHMM
            from pensive.generators.moore import MooreHMM

            if isinstance(self, (MealyHMM, MooreHMM)):
                from pensive.generators.epsilon_machine import EpsilonMachine

                if isinstance(self, EpsilonMachine):
                    return EpsilonMachine.from_time_reversed(self)
                return EpsilonMachine.from_generator(time_reverse_stochastic(self))
            raise NotImplementedError(
                "time-reversed generators with edge emissions require EpsilonMachine.from_generator"
            )
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

    def forward(self, observations: Sequence[Any]) -> np.ndarray:
        from pensive.generators.hmm_inference import forward

        return forward(self, observations)

    def backward(self, observations: Sequence[Any]) -> np.ndarray:
        from pensive.generators.hmm_inference import backward

        return backward(self, observations)

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

    def to_sofic_shift(self) -> SoficShift:
        """Strip probabilities and return a sofic shift with the same support."""
        from pensive.generators.conversions import hmm_to_sofic_shift

        return hmm_to_sofic_shift(self)

    def to_automata(self) -> NFA:
        """Return an NFA for the support language of this HMM."""
        from pensive.generators.conversions import hmm_to_automata

        return hmm_to_automata(self)

    def to_dfa(self) -> DFA:
        """Return a determinized automaton for the support language of this HMM."""
        from pensive.generators.conversions import hmm_to_dfa

        return hmm_to_dfa(self)

    def reverse(self) -> Self:
        return super().reverse()


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
