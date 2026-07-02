"""Conjugate Bayesian inference for finite-order Markov chains."""

from __future__ import annotations

from collections.abc import Hashable, Iterable, Sequence
from itertools import product
from typing import Any

import numpy as np
from scipy.special import polygamma

from pensive.generators.mealy import MealyHMM
from pensive.graph import ATTR_EMISSION, ATTR_PROB
from pensive.inference.bayesian.counts import (
    BayesianInferenceError,
    WordCountsMC,
    dirichlet_multinomial_log_evidence,
    pretty_symbol,
    pretty_word,
    split_word,
)


def words_iter(alphabet: Sequence[Any], length: int) -> Iterable[tuple[Any, ...]]:
    return product(tuple(alphabet), repeat=length)


class DirichletPriorMC:
    """Dirichlet row prior for an order-``k`` Markov chain."""

    def __init__(self, alphabet: Sequence[Any], order: int, uniform: bool = True):
        self.alphabet = tuple(alphabet)
        self.order = int(order)
        self.uniform = uniform
        self.alphas: dict[tuple[tuple[Any, ...], Any], float] = {}

    def __str__(self) -> str:
        lines = []
        for context in words_iter(self.alphabet, self.order):
            lines.append(f"alpha({pretty_word(context)} -> *) = {self.get_alpha((*context, '*'))}")
            for symbol in self.alphabet:
                lines.append(f"alpha({pretty_word(context)} -> {pretty_symbol(symbol)}) = {self.get_alpha((*context, symbol))}")
        return "\n".join(lines) + "\n"

    def create_random_prior(self, lower: int, upper: int, rng: np.random.Generator | None = None) -> None:
        generator = rng if rng is not None else np.random.default_rng()
        self.uniform = False
        for word in words_iter(self.alphabet, self.order + 1):
            self.set_alpha(word, int(generator.integers(lower, upper)))

    def get_alpha(self, word: Sequence[Any]) -> float | None:
        word = tuple(word)
        if len(word) != self.order + 1:
            return None
        if self.uniform:
            return float(len(self.alphabet) if word[-1] == "*" else 1.0)
        return self.alphas.get(split_word(word))

    def set_alpha(self, word: Sequence[Any], value: float) -> None:
        self.uniform = False
        context, symbol = split_word(word)
        previous = self.alphas.get((context, symbol), 0.0)
        self.alphas[(context, symbol)] = float(value)
        self.alphas[(context, "*")] = self.alphas.get((context, "*"), 0.0) - previous + float(value)


class MarkovChainPosterior:
    """Posterior over a finite-order Markov chain's transition rows."""

    def __init__(self, alphabet: Sequence[Any], data: Sequence[Any], order: int, prior_type: str = "uniform"):
        self.alphabet = tuple(alphabet)
        self.order = int(order)
        self.prior_type = prior_type
        self.counts = WordCountsMC(data, self.order)
        self.prior = DirichletPriorMC(self.alphabet, self.order)
        if prior_type == "random":
            self.prior.create_random_prior(1, 6)
        elif prior_type != "uniform":
            raise BayesianInferenceError("unknown Markov-chain prior type")

    @property
    def contexts(self) -> tuple[tuple[Any, ...], ...]:
        return tuple(words_iter(self.alphabet, self.order))

    def add_counts_from(self, data: Sequence[Any]) -> None:
        self.counts.add_counts_from(data)

    def transition_probability_mle(self, word: Sequence[Any], symbol: Any) -> tuple[float, float]:
        context = tuple(word)
        n = self.counts.get_word_count((*context, symbol))
        N = self.counts.get_word_count((*context, "*"))
        if N > 0:
            prob = n / N
            variance = n * (N - n) / N**3
        else:
            prob = variance = 0.0
        return float(prob), float(variance)

    def transition_probability_pme(self, word: Sequence[Any], symbol: Any) -> tuple[float, float]:
        context = tuple(word)
        n = self.counts.get_word_count((*context, symbol))
        N = self.counts.get_word_count((*context, "*"))
        a = self.prior.get_alpha((*context, symbol))
        A = self.prior.get_alpha((*context, "*"))
        if a is None or A is None:
            raise BayesianInferenceError("missing prior alpha")
        prob = (n + a) / (N + A)
        variance = ((n + a) * (N + A - n - a)) / ((N + A + 1) * (N + A) ** 2)
        return float(prob), float(variance)

    def posterior_alpha_matrix(self) -> np.ndarray:
        matrix = np.zeros((len(self.contexts), len(self.alphabet)), dtype=float)
        for i, context in enumerate(self.contexts):
            for j, symbol in enumerate(self.alphabet):
                alpha = self.prior.get_alpha((*context, symbol))
                if alpha is None:
                    raise BayesianInferenceError("missing prior alpha")
                matrix[i, j] = alpha + self.counts.get_word_count((*context, symbol))
        return matrix

    def posterior_mean_matrix(self) -> np.ndarray:
        alpha = self.posterior_alpha_matrix()
        return alpha / alpha.sum(axis=1, keepdims=True)

    def log_evidence(self) -> float:
        evidence = 0.0
        for context in self.contexts:
            root = (*context, "*")
            alpha_root = self.prior.get_alpha(root)
            if alpha_root is None:
                raise BayesianInferenceError("missing prior alpha")
            cells: list[tuple[float, float]] = []
            for symbol in self.alphabet:
                word = (*context, symbol)
                alpha = self.prior.get_alpha(word)
                if alpha is None:
                    raise BayesianInferenceError("missing prior alpha")
                cells.append((alpha, self.counts.get_word_count(word)))
            evidence += dirichlet_multinomial_log_evidence(
                alpha_root, self.counts.get_word_count(root), cells
            )
        return float(evidence)

    def average_relative_entropy_plus_entropy_rate(self) -> float:
        beta = 0.0
        for context in self.contexts:
            for symbol in self.alphabet:
                beta += self.counts.get_word_count((*context, symbol)) + self.prior.get_alpha((*context, symbol))

        result = 0.0
        invlog2 = 1 / np.log(2)
        for context in self.contexts:
            root = (*context, "*")
            n = self.counts.get_word_count(root)
            alpha = self.prior.get_alpha(root)
            prob = (n + alpha) / beta
            result += invlog2 * prob * polygamma(0, n + alpha)
            for symbol in self.alphabet:
                cond, _variance = self.transition_probability_pme(context, symbol)
                ws_count = self.counts.get_word_count((*context, symbol))
                ws_alpha = self.prior.get_alpha((*context, symbol))
                result -= invlog2 * prob * cond * polygamma(0, ws_count + ws_alpha)
        return float(result)

    def variance_relative_entropy_plus_entropy_rate(self) -> float:
        beta = 0.0
        for context in self.contexts:
            for symbol in self.alphabet:
                beta += self.counts.get_word_count((*context, symbol)) + self.prior.get_alpha((*context, symbol))

        result = 0.0
        invlog2 = 1 / np.log(2)
        for context in self.contexts:
            root = (*context, "*")
            n = self.counts.get_word_count(root)
            alpha = self.prior.get_alpha(root)
            prob = (n + alpha) / beta
            result -= invlog2 * prob**2 * polygamma(1, n + alpha)
            for symbol in self.alphabet:
                cond, _variance = self.transition_probability_pme(context, symbol)
                ws_count = self.counts.get_word_count((*context, symbol))
                ws_alpha = self.prior.get_alpha((*context, symbol))
                result += invlog2 * prob**2 * cond**2 * polygamma(1, ws_count + ws_alpha)
        return float(result)

    def transition_probability_mle_iter(self) -> Iterable[tuple[str, str, float, float]]:
        for context in self.contexts:
            for symbol in self.alphabet:
                prob, var = self.transition_probability_mle(context, symbol)
                yield pretty_word(context), pretty_symbol(symbol), prob, var

    def transition_probability_pme_iter(self) -> Iterable[tuple[str, str, float, float]]:
        for context in self.contexts:
            for symbol in self.alphabet:
                prob, var = self.transition_probability_pme(context, symbol)
                yield pretty_word(context), pretty_symbol(symbol), prob, var

    def _state_for_context(self, context: tuple[Any, ...]) -> Hashable:
        return context if self.order else "A"

    def _target_for(self, context: tuple[Any, ...], symbol: Any) -> Hashable:
        return (*context[1:], symbol) if self.order else "A"

    def generate_mealy_hmm(self, method: str = "PME", threshold: float = 0.0, reduce: bool = True) -> MealyHMM:
        del reduce
        if not 0 <= threshold <= 1:
            raise BayesianInferenceError("threshold must be between 0 and 1")
        if method == "PME":
            matrix = self.posterior_mean_matrix()
        elif method == "MLE":
            matrix = np.array(
                [[self.transition_probability_mle(context, symbol)[0] for symbol in self.alphabet] for context in self.contexts],
                dtype=float,
            )
        else:
            raise BayesianInferenceError("unknown inference method")

        hmm = MealyHMM(observation_alphabet=frozenset(self.alphabet))
        hmm.name = f"Inferred order-{self.order} Markov chain, {method}"
        states = [self._state_for_context(context) for context in self.contexts]
        initial = {state: 1.0 / len(states) for state in states} if states else {}
        hmm.initial_distribution = initial
        for state in states:
            hmm.graph.add_state(state)
        for i, context in enumerate(self.contexts):
            source = self._state_for_context(context)
            for j, symbol in enumerate(self.alphabet):
                prob = float(matrix[i, j])
                if prob > threshold:
                    hmm.graph.add_transition(source, self._target_for(context, symbol), **{ATTR_EMISSION: symbol, ATTR_PROB: prob})
        hmm.validate()
        return hmm

    def sample_mealy_hmms(
        self,
        n: int = 1,
        threshold: float = 0.0,
        reduce: bool = True,
        rng: np.random.Generator | None = None,
    ) -> Iterable[MealyHMM]:
        del reduce
        if not 0 <= threshold <= 1:
            raise BayesianInferenceError("threshold must be between 0 and 1")
        generator = rng if rng is not None else np.random.default_rng()
        alpha = self.posterior_alpha_matrix()
        samples = np.stack([generator.dirichlet(row, size=n) for row in alpha], axis=1)
        for sample in samples:
            hmm = MealyHMM(observation_alphabet=frozenset(self.alphabet))
            hmm.name = f"Sampled order-{self.order} Markov chain"
            states = [self._state_for_context(context) for context in self.contexts]
            hmm.initial_distribution = {state: 1.0 / len(states) for state in states}
            for state in states:
                hmm.graph.add_state(state)
            for i, context in enumerate(self.contexts):
                source = self._state_for_context(context)
                for j, symbol in enumerate(self.alphabet):
                    prob = float(sample[i, j])
                    if prob > threshold:
                        hmm.graph.add_transition(source, self._target_for(context, symbol), **{ATTR_EMISSION: symbol, ATTR_PROB: prob})
            hmm.validate()
            yield hmm

    def as_pymc_model(self, *, observed_as_counts: bool = False) -> Any:
        from pensive.inference.bayesian.pymc_backend import markov_chain_model

        return markov_chain_model(self, observed_as_counts=observed_as_counts)

    def counts_string(self) -> str:
        return str(self.counts)

    def prior_string(self) -> str:
        return str(self.prior)

    def transition_probability_mle_string(self) -> str:
        return "".join(
            f"Pr( {symbol} | {context} ) = {prob:.8f} , StdDev = {np.sqrt(var):.8f}\n"
            for context, symbol, prob, var in self.transition_probability_mle_iter()
        )

    def transition_probability_pme_string(self) -> str:
        return "".join(
            f"Pr( {symbol} | {context} ) = {prob:.8f} , StdDev = {np.sqrt(var):.8f}\n"
            for context, symbol, prob, var in self.transition_probability_pme_iter()
        )


InferMC = MarkovChainPosterior
