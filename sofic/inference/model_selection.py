"""Classical model-selection criteria for stochastic generators.

Point-estimate information criteria -- AIC :cite:`Akaike1974`, the
small-sample-corrected AICc :cite:`HurvichTsai1989`, BIC :cite:`Schwarz1978`,
and the two-part minimum description length :cite:`Rissanen1978` -- together
with cross-validated log-likelihood and the widely applicable information
criterion (WAIC) :cite:`Watanabe2010`. These complement the exact Bayesian
evidences of :mod:`sofic.inference.bayesian`: they score any fitted
:class:`~sofic.generators.base.HiddenMarkovModel` (ε-machine, Mealy HMM, Markov
chain) using the natural-log likelihood from
:func:`sofic.generators.hmm_inference.log_likelihood` and a free-parameter count
read off the transition graph, so they are likelihood-agnostic and apply
directly to discrete-emission models.

All information criteria follow the convention **lower is better**;
cross-validated and WAIC log scores follow **higher is better** for the raw log
score (WAIC itself is reported on the deviance scale, lower is better).
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from sofic.generators.base import HiddenMarkovModel
from sofic.generators.hmm_inference import free_parameter_labels, log_likelihood

__all__ = [
    "ModelScores",
    "WAICResult",
    "count_free_parameters",
    "cross_validated_log_likelihood",
    "information_criterion",
    "compare_information_criteria",
    "posterior_pointwise_log_likelihoods",
    "rank_topological_epsilon_machines",
    "score_model",
    "waic",
    "waic_epsilon_machine",
]

_CRITERIA = ("aic", "aicc", "bic", "mdl")


@dataclass(frozen=True)
class ModelScores:
    """Information-criterion scores for one fitted model (lower is better)."""

    log_likelihood: float
    num_parameters: int
    num_observations: int
    aic: float
    aicc: float
    bic: float
    mdl: float

    def value(self, criterion: str) -> float:
        """Return the score for ``criterion`` (one of ``aic``/``aicc``/``bic``/``mdl``)."""
        key = criterion.lower()
        if key not in _CRITERIA:
            raise ValueError(f"unknown criterion {criterion!r}; choose from {_CRITERIA}")
        return float(getattr(self, key))


@dataclass(frozen=True)
class WAICResult:
    """Widely applicable information criterion decomposition."""

    waic: float
    lppd: float
    p_waic: float
    standard_error: float


def _normalize_sequences(sequences: Iterable[Any]) -> list[list[Any]]:
    seqs = list(sequences)
    if not seqs:
        return []
    first = seqs[0]
    if isinstance(first, (list, tuple)) and not isinstance(first, (str, bytes)):
        return [list(seq) for seq in seqs]
    return [list(seqs)]


def count_free_parameters(model: HiddenMarkovModel, *, include_initial: bool = False) -> int:
    """Return the number of free real parameters of ``model``.

    Counts one free parameter per non-reference outgoing edge at each state (the
    multinomial free-parameterization of the joint emission-transition law used
    by :func:`sofic.generators.hmm_inference.observed_information`). With
    ``include_initial`` the ``n - 1`` free parameters of the initial
    distribution are added; for a stationary presentation the initial law is
    determined by the dynamics, so this defaults to ``False``.
    """
    transition_params = len(free_parameter_labels(model))
    if not include_initial:
        return transition_params
    support = sum(1 for mass in model.initial_distribution.values() if float(mass) > 0.0)
    return transition_params + max(0, support - 1)


def _total_length(sequences: Sequence[Sequence[Any]]) -> int:
    return int(sum(len(seq) for seq in sequences))


def _total_log_likelihood(model: HiddenMarkovModel, sequences: Sequence[Sequence[Any]]) -> float:
    total = 0.0
    for seq in sequences:
        contribution = log_likelihood(model, seq)
        if not np.isfinite(contribution):
            return float("-inf")
        total += contribution
    return total


def score_model(
    model: HiddenMarkovModel,
    data: Iterable[Any],
    *,
    include_initial: bool = False,
) -> ModelScores:
    """Score ``model`` on ``data`` with AIC, AICc, BIC, and MDL.

    ``data`` may be a single observation sequence or an iterable of sequences.
    The scores use natural-log likelihoods; the number of observations is the
    total symbol count. When the data has zero probability under the model the
    likelihood is ``-inf`` and every criterion is ``+inf``.
    """
    sequences = _normalize_sequences(data)
    n = _total_length(sequences)
    k = count_free_parameters(model, include_initial=include_initial)
    ll = _total_log_likelihood(model, sequences)

    if not np.isfinite(ll):
        inf = float("inf")
        return ModelScores(float("-inf"), k, n, inf, inf, inf, inf)

    aic = 2.0 * k - 2.0 * ll
    denom = n - k - 1
    aicc = aic + (2.0 * k * (k + 1)) / denom if denom > 0 else float("inf")
    log_n = np.log(n) if n > 0 else 0.0
    bic = k * log_n - 2.0 * ll
    mdl = 0.5 * k * log_n - ll
    return ModelScores(float(ll), k, n, float(aic), float(aicc), float(bic), float(mdl))


def information_criterion(
    model: HiddenMarkovModel,
    data: Iterable[Any],
    *,
    criterion: str = "bic",
    include_initial: bool = False,
) -> float:
    """Return a single information-criterion value for ``model`` (lower is better)."""
    return score_model(model, data, include_initial=include_initial).value(criterion)


def compare_information_criteria(
    models: Iterable[HiddenMarkovModel],
    data: Iterable[Any],
    *,
    criterion: str = "bic",
    include_initial: bool = False,
) -> dict[str, ModelScores]:
    """Score several models on shared ``data``, keyed by each model's ``name``.

    The data is materialized once and reused. Model keys fall back to
    ``Model-<index>`` when a model has no ``name`` attribute.
    """
    sequences = _normalize_sequences(data)
    _ = criterion  # accepted for symmetry; callers pick the field via ModelScores.value
    scores: dict[str, ModelScores] = {}
    for index, model in enumerate(models):
        name = str(getattr(model, "name", None) or f"Model-{index}")
        scores[name] = score_model(model, sequences, include_initial=include_initial)
    return scores


# --- Cross-validated log-likelihood ---------------------------------------


def _fold_indices(n_items: int, folds: int, rng: np.random.Generator) -> list[np.ndarray]:
    order = rng.permutation(n_items)
    return [np.sort(chunk) for chunk in np.array_split(order, folds)]


def cross_validated_log_likelihood(
    fit: Callable[[list[Any]], HiddenMarkovModel],
    data: Iterable[Any],
    *,
    folds: int = 5,
    rng: np.random.Generator | int | None = None,
) -> float:
    """Return the total held-out natural-log likelihood under ``folds``-fold CV.

    ``fit(train_sequences)`` must fit and return a model from a list of training
    sequences. When ``data`` is a collection of sequences the folds partition the
    sequences; a single long sequence is split into ``folds`` contiguous blocks.
    Each held-out block is scored under a model trained on the remaining data and
    the contributions are summed (higher is better). A fold whose held-out data
    has zero probability contributes ``-inf``.
    """
    generator = rng if isinstance(rng, np.random.Generator) else np.random.default_rng(rng)
    sequences = _normalize_sequences(data)
    if folds < 2:
        raise ValueError("folds must be at least 2")

    if len(sequences) >= folds:
        partition = _fold_indices(len(sequences), folds, generator)
        blocks = [[sequences[i] for i in idx] for idx in partition]
    else:
        # Single (or few) long sequence(s): split the concatenation into contiguous blocks.
        flat = [symbol for seq in sequences for symbol in seq]
        if len(flat) < folds:
            raise ValueError("not enough data for the requested number of folds")
        blocks = [list(chunk) for chunk in np.array_split(np.array(flat, dtype=object), folds)]
        blocks = [[list(block)] for block in blocks]

    total = 0.0
    for held_out_index in range(len(blocks)):
        train: list[Any] = []
        for index, block in enumerate(blocks):
            if index == held_out_index:
                continue
            train.extend(block)
        held_out = blocks[held_out_index]
        model = fit(train)
        total += _total_log_likelihood(model, _normalize_sequences(held_out))
    return float(total)


# --- WAIC ------------------------------------------------------------------


def waic(pointwise_log_likelihoods: np.ndarray) -> WAICResult:
    r"""Widely applicable information criterion from posterior samples.

    ``pointwise_log_likelihoods`` has shape ``(n_samples, n_points)`` with entry
    ``[s, i] = log p(y_i | theta_s)`` for posterior draw ``theta_s``. Returns the
    WAIC on the deviance scale (lower is better),
    ``WAIC = -2 (lppd - p_waic)`` with the log pointwise predictive density
    ``lppd = sum_i log mean_s p(y_i | theta_s)`` and effective parameter count
    ``p_waic = sum_i Var_s log p(y_i | theta_s)`` :cite:`Watanabe2010`.
    """
    from scipy.special import logsumexp

    matrix = np.asarray(pointwise_log_likelihoods, dtype=float)
    if matrix.ndim != 2 or matrix.size == 0:
        raise ValueError("pointwise_log_likelihoods must be a non-empty (n_samples, n_points) array")
    n_samples = matrix.shape[0]
    lppd_pointwise = logsumexp(matrix, axis=0) - np.log(n_samples)
    p_waic_pointwise = matrix.var(axis=0, ddof=1) if n_samples > 1 else np.zeros(matrix.shape[1])
    elpd_pointwise = lppd_pointwise - p_waic_pointwise
    waic_value = -2.0 * float(elpd_pointwise.sum())
    n_points = matrix.shape[1]
    standard_error = float(np.sqrt(n_points * np.var(-2.0 * elpd_pointwise, ddof=0))) if n_points > 1 else 0.0
    return WAICResult(
        waic=waic_value,
        lppd=float(lppd_pointwise.sum()),
        p_waic=float(p_waic_pointwise.sum()),
        standard_error=standard_error,
    )


def posterior_pointwise_log_likelihoods(
    posterior: Any,
    data: Iterable[Any],
    *,
    n_samples: int = 200,
    rng: np.random.Generator | int | None = None,
) -> np.ndarray:
    """Sample per-sequence log-likelihoods from an ε-machine/Markov posterior.

    ``posterior`` must expose ``generate_sample(rng=...) -> (start, model)`` (e.g.
    :class:`~sofic.inference.bayesian.epsilon.EpsilonMachinePosterior`). Each data
    sequence is one WAIC "point"; returns an array of shape
    ``(n_samples, n_sequences)`` suitable for :func:`waic`.
    """
    generator = rng if isinstance(rng, np.random.Generator) else np.random.default_rng(rng)
    sequences = _normalize_sequences(data)
    matrix = np.empty((n_samples, len(sequences)), dtype=float)
    for s in range(n_samples):
        _start, model = posterior.generate_sample(rng=generator)
        for i, seq in enumerate(sequences):
            matrix[s, i] = log_likelihood(model, seq)
    return matrix


def waic_epsilon_machine(
    posterior: Any,
    data: Iterable[Any],
    *,
    n_samples: int = 200,
    rng: np.random.Generator | int | None = None,
) -> WAICResult:
    """Compute WAIC for a posterior over machines by sampling parameters."""
    return waic(posterior_pointwise_log_likelihoods(posterior, data, n_samples=n_samples, rng=rng))


# --- Topological enumeration ranking --------------------------------------


@dataclass(frozen=True)
class TopologyScore:
    """A candidate topology, its fitted realization, and its scores."""

    machine: Any
    scores: ModelScores
    criterion_value: float


def _fit_topology(machine: Any, sequences: list[list[Any]], method: str) -> HiddenMarkovModel | None:
    if method == "bayesian":
        from sofic.inference.bayesian.epsilon import EpsilonMachinePosterior

        flat = [symbol for seq in sequences for symbol in seq]
        posterior = EpsilonMachinePosterior(machine, flat)
        return posterior.posterior_mean_machine()
    if method == "baum_welch":
        fitted, _trace = machine.baum_welch(sequences)
        return fitted
    raise ValueError(f"unknown fit method {method!r}; choose 'bayesian' or 'baum_welch'")


def rank_topological_epsilon_machines(
    data: Iterable[Any],
    *,
    alphabet: Sequence[Any],
    num_states: int | Iterable[int],
    criterion: str = "bic",
    fit: str = "bayesian",
    include_initial: bool = False,
    check_minimal: bool = True,
) -> list[TopologyScore]:
    """Rank enumerated topological ε-machines by an information criterion.

    Enumerates canonical topological ε-machines over ``alphabet`` for each
    requested state count (:func:`~sofic.generators.topological_epsilon_enumeration.iter_topological_epsilon_machines`),
    fits transition probabilities to ``data`` (Bayesian posterior mean by
    default, or Baum-Welch), scores each fitted model, and returns the
    candidates sorted best-first by ``criterion``. This is the frequentist
    counterpart to the Bayesian topology comparison of
    :class:`~sofic.inference.bayesian.comparison.ModelComparisonEM`.
    """
    from sofic.generators.topological_epsilon_enumeration import iter_topological_epsilon_machines

    sequences = _normalize_sequences(data)
    symbols = tuple(alphabet)
    k = len(symbols)
    counts = [num_states] if isinstance(num_states, int) else sorted({int(value) for value in num_states})

    results: list[TopologyScore] = []
    for n in counts:
        for topology in iter_topological_epsilon_machines(k, n, alphabet=symbols, check_minimal=check_minimal):
            fitted = _fit_topology(topology, sequences, fit)
            if fitted is None:
                continue
            scores = score_model(fitted, sequences, include_initial=include_initial)
            results.append(TopologyScore(fitted, scores, scores.value(criterion)))

    results.sort(key=lambda item: item.criterion_value)
    return results
