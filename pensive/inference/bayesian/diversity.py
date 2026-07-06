"""Posterior diversity diagnostics for Bayesian epsilon-machine model comparison.

Two complementary notions of posterior spread are tracked:

* **Machine diversity** — Shannon entropy of the topology weights returned by
  :meth:`~pensive.inference.bayesian.comparison.ModelComparisonEM.model_probabilities`.
  This measures uncertainty over *presentations* (topologies), not processes.

* **Process diversity** — weighted Jensen–Shannon divergence (JSD) over length-:math:`L`
  word distributions, one per posterior component. When many high-weight machines
  generate nearly the same stochastic process, process diversity can be much smaller
  than machine diversity.

Word length conventions
-----------------------
For an :math:`n`-state presentation, classical HMM identification (Paz 1971;
Finesso 1991) uses the full length-:math:`(2n-1)` word distribution. Some
minimal-realization algorithms use a conservative window of :math:`2n+1`.
Upper's rank-growing history/future word lists (see
:func:`~pensive.generators.process_equivalence.is_equal_process`) provide a
data-driven alternative. All topologies in a comparison share the same :math:`L`,
chosen from the largest state count in the posterior.

JSD is computed in bits via ``dit``.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from itertools import product
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from pensive.generators.base import HiddenMarkovModel
from pensive.generators.process_equivalence import _HistoryFutureWordList
from pensive.generators.words import hmm_words_of_length
from pensive.inference.bayesian.counts import BayesianInferenceError
from pensive.inference.bayesian.epsilon import EpsilonMachinePosterior

if TYPE_CHECKING:
    from pensive.inference.bayesian.comparison import ModelComparisonEM

_TOL = 1e-15
WordLengthConvention = Literal["paz", "conservative", "upper_list"]
ProcessDiversityMethod = Literal["posterior_mean", "monte_carlo"]


@dataclass(frozen=True)
class PosteriorDiversityResult:
    """Posterior machine and process diversity diagnostics."""

    process_diversity: float
    machine_diversity: float
    word_length: int
    method: ProcessDiversityMethod
    n_components: int


def _require_dit():
    try:
        from dit.divergences.jensen_shannon_divergence import jensen_shannon_divergence_pmf
    except ImportError as exc:
        raise ImportError(
            "dit is required for posterior process diversity; install with `pip install dit`"
        ) from exc
    return jensen_shannon_divergence_pmf


def _comparison_alphabet(comparison: ModelComparisonEM) -> tuple[Any, ...]:
    alphabets: set[Any] = set()
    for posterior in comparison.em_dict.values():
        alphabets.update(posterior.machine.observation_alphabet)
    if not alphabets:
        raise BayesianInferenceError("posterior has no viable machines")
    return tuple(sorted(alphabets, key=repr))


def _max_state_count(comparison: ModelComparisonEM) -> int:
    if not comparison.em_dict:
        raise BayesianInferenceError("posterior has no viable machines")
    return max(len(posterior.dirichlet.nodes) for posterior in comparison.em_dict.values())


def _upper_list_word_length(machine: HiddenMarkovModel) -> int:
    hf = _HistoryFutureWordList.from_hmm(machine)
    future_words = hf.future_word_list()
    history_words = hf.history_word_list()
    candidates = [len(word) for word in (*future_words, *history_words)]
    return max(candidates) if candidates else 0


def process_identification_word_length(
    comparison: ModelComparisonEM,
    *,
    convention: WordLengthConvention | str = "paz",
    word_length: int | None = None,
) -> int:
    """Return the word length used to compare processes in a model comparison."""
    if word_length is not None:
        length = int(word_length)
        if length < 0:
            raise BayesianInferenceError("word_length must be nonnegative")
        return length

    n_max = _max_state_count(comparison)
    if convention == "paz":
        return max(0, 2 * n_max - 1)
    if convention == "conservative":
        return 2 * n_max + 1
    if convention == "upper_list":
        lengths = []
        for posterior in comparison.em_dict.values():
            start_probs = posterior.start_node_probabilities()
            if not start_probs:
                continue
            start_node = max(start_probs, key=start_probs.get)
            mean_machine = posterior.posterior_mean_machine(start_node)
            if mean_machine is not None:
                lengths.append(_upper_list_word_length(mean_machine))
        if not lengths:
            return max(0, 2 * n_max - 1)
        return max(lengths)
    raise BayesianInferenceError(f"unknown word-length convention: {convention!r}")


def word_distribution_to_pmf(
    distribution: dict[tuple[Any, ...], float],
    alphabet: Sequence[Any],
    length: int,
) -> np.ndarray:
    """Align a sparse word distribution to a dense PMF over ``alphabet**length``."""
    symbols = tuple(alphabet)
    if length == 0:
        total = sum(float(prob) for prob in distribution.values())
        return np.array([total if total > _TOL else 0.0], dtype=float)

    outcomes = list(product(symbols, repeat=length))
    pmf = np.zeros(len(outcomes), dtype=float)
    for index, word in enumerate(outcomes):
        prob = distribution.get(word, 0.0)
        if prob > _TOL:
            pmf[index] = float(prob)

    total = float(pmf.sum())
    if total > _TOL and abs(total - 1.0) > _TOL:
        pmf /= total
    return pmf


def machine_diversity(comparison: ModelComparisonEM) -> float:
    """Shannon entropy (bits) of the topology posterior weights."""
    probs = np.array(list(comparison.model_probabilities().values()), dtype=float)
    if probs.size == 0:
        raise BayesianInferenceError("posterior has no viable machines")
    positive = probs[probs > 0.0]
    if positive.size <= 1:
        return 0.0
    return float(-np.sum(positive * np.log2(positive)))


def posterior_mean_word_distribution(
    posterior: EpsilonMachinePosterior,
    length: int,
) -> dict[tuple[Any, ...], float]:
    """Start-marginalized word distribution from posterior-mean transition probabilities."""
    if length < 0:
        raise BayesianInferenceError("length must be nonnegative")

    start_probs = posterior.start_node_probabilities()
    if not start_probs:
        return {}

    distribution: dict[tuple[Any, ...], float] = {}
    for start_node, start_weight in start_probs.items():
        if start_weight <= _TOL:
            continue
        machine = posterior.posterior_mean_machine(start_node)
        if machine is None:
            continue
        words = hmm_words_of_length(machine, length)
        for word, prob in words.items():
            distribution[word] = distribution.get(word, 0.0) + start_weight * float(prob)
    return distribution


def _jsd_from_word_distributions(
    distributions: Sequence[dict[tuple[Any, ...], float]],
    weights: Sequence[float],
    alphabet: Sequence[Any],
    length: int,
) -> float:
    jensen_shannon_divergence_pmf = _require_dit()
    if not distributions:
        raise BayesianInferenceError("no word distributions supplied")
    if len(distributions) != len(weights):
        raise BayesianInferenceError("number of weights must match number of distributions")

    pmfs = np.vstack([word_distribution_to_pmf(dist, alphabet, length) for dist in distributions])
    weight_array = np.asarray(weights, dtype=float)
    if weight_array.sum() <= _TOL:
        raise BayesianInferenceError("posterior weights must sum to a positive value")
    return float(jensen_shannon_divergence_pmf(pmfs, weight_array))


def posterior_process_diversity(
    comparison: ModelComparisonEM,
    *,
    method: ProcessDiversityMethod = "posterior_mean",
    n_samples: int = 500,
    rng: np.random.Generator | None = None,
    convention: WordLengthConvention | str = "paz",
    word_length: int | None = None,
) -> PosteriorDiversityResult:
    """Compute weighted JSD over posterior word distributions."""
    if not comparison.em_dict:
        raise BayesianInferenceError("posterior has no viable machines")

    length = process_identification_word_length(comparison, convention=convention, word_length=word_length)
    machine_div = machine_diversity(comparison)
    alphabet = _comparison_alphabet(comparison)

    if method == "posterior_mean":
        model_probs = comparison.model_probabilities()
        names = list(model_probs)
        distributions = [posterior_mean_word_distribution(comparison.em_dict[name], length) for name in names]
        weights = [model_probs[name] for name in names]
        process_div = _jsd_from_word_distributions(distributions, weights, alphabet, length)
        return PosteriorDiversityResult(
            process_diversity=process_div,
            machine_diversity=machine_div,
            word_length=length,
            method="posterior_mean",
            n_components=len(names),
        )

    if method == "monte_carlo":
        if n_samples <= 0:
            raise BayesianInferenceError("n_samples must be positive")
        generator = rng if rng is not None else np.random.default_rng()
        distributions = []
        for _ in range(n_samples):
            _start, machine = comparison.generate_sample(rng=generator)
            distributions.append(hmm_words_of_length(machine, length))
        weights = [1.0 / n_samples] * n_samples
        process_div = _jsd_from_word_distributions(distributions, weights, alphabet, length)
        return PosteriorDiversityResult(
            process_diversity=process_div,
            machine_diversity=machine_div,
            word_length=length,
            method="monte_carlo",
            n_components=n_samples,
        )

    raise BayesianInferenceError(f"unknown process diversity method: {method!r}")
