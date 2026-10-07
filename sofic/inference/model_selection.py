r"""Classical model-selection criteria for stochastic generators.

Point-estimate information criteria -- AIC :cite:`Akaike1974`, the
small-sample-corrected AICc :cite:`HurvichTsai1989`, BIC :cite:`Schwarz1978`,
and the two-part minimum description length :cite:`Rissanen1978` -- together
with cross-validated log-likelihood and the widely applicable information
criterion (WAIC) :cite:`Watanabe2010`. These complement the exact Bayesian
evidences of :mod:`sofic.inference.bayesian`: they score any fitted
:class:`~sofic.generators.base.HiddenMarkovModel` (ε-machine, Mealy HMM, Markov
chain) using the log-likelihood (in bits) from
:func:`sofic.inference.hmm.log_likelihood` and a free-parameter count
read off the transition graph, so they are likelihood-agnostic and apply
directly to discrete-emission models.

Log-likelihoods, log scores, and the MDL code length are reported in **bits**.
AIC, AICc, BIC, and WAIC keep their standard deviance scale: they are computed
from the natural log-likelihood :math:`\ln L = \ln 2 \cdot \log_2 L`.

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
from sofic.inference.hmm import free_parameter_labels, log_likelihood

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
    by :func:`sofic.inference.hmm.observed_information`). With
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


def _smoothed_log_likelihood(
    model: HiddenMarkovModel,
    sequence: Sequence[Any],
    *,
    smoothing: float,
    alphabet_size: int,
) -> float:
    """Log-likelihood (bits) with each one-step prediction mixed with the uniform law.

    ``P'(x_t | x_{0:t}) = (1 - smoothing) P(x_t | x_{0:t}) + smoothing / alphabet_size``,
    computed by forward filtering. After a symbol the model forbids, the belief is
    propagated without conditioning on it.
    """
    from sofic.generators.matrices import emission_tensors

    pi, joint = emission_tensors(model)
    total_step = sum(joint.values())
    belief = np.asarray(pi, dtype=float)
    belief = belief / belief.sum()
    total = 0.0
    for symbol in sequence:
        matrix = joint.get(symbol)
        unnormalized = belief @ matrix if matrix is not None else np.zeros_like(belief)
        predicted = float(unnormalized.sum())
        total += float(np.log2((1.0 - smoothing) * predicted + smoothing / alphabet_size))
        if predicted > 0.0:
            belief = unnormalized / predicted
        else:
            belief = belief @ total_step
            belief = belief / belief.sum()
    return total


def score_model(
    model: HiddenMarkovModel,
    data: Iterable[Any],
    *,
    include_initial: bool = False,
) -> ModelScores:
    """Score ``model`` on ``data`` with AIC, AICc, BIC, and MDL.

    ``data`` may be a single observation sequence or an iterable of sequences.
    ``log_likelihood`` and ``mdl`` are in bits; AIC, AICc, and BIC use the natural
    log-likelihood so they keep their usual scale. The number of observations is
    the total symbol count. When the data has zero probability under the model the
    likelihood is ``-inf`` and every criterion is ``+inf``.
    """
    sequences = _normalize_sequences(data)
    n = _total_length(sequences)
    k = count_free_parameters(model, include_initial=include_initial)
    ll = _total_log_likelihood(model, sequences)

    if not np.isfinite(ll):
        inf = float("inf")
        return ModelScores(float("-inf"), k, n, inf, inf, inf, inf)

    ln_l = ll * np.log(2.0)
    aic = 2.0 * k - 2.0 * ln_l
    denom = n - k - 1
    aicc = aic + (2.0 * k * (k + 1)) / denom if denom > 0 else float("inf")
    bic = k * (np.log(n) if n > 0 else 0.0) - 2.0 * ln_l
    mdl = 0.5 * k * (np.log2(n) if n > 0 else 0.0) - ll
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
    gap: int = 0,
    smoothing: float = 0.0,
) -> float:
    """Return the total held-out log-likelihood (bits) under ``folds``-fold CV.

    ``fit(train_sequences)`` must fit and return a model from a list of training
    sequences. When ``data`` holds at least ``folds`` sequences the folds partition
    the sequences. Otherwise (a single long sequence, or fewer sequences than
    folds) every sequence is split into ``folds`` contiguous blocks and fold ``f``
    holds out block ``f`` of each sequence; the remaining blocks are passed to
    ``fit`` as separate training sequences, so distinct sequences are never
    concatenated.
    Each held-out block is scored under a model trained on the remaining data and
    the contributions are summed (higher is better). A fold whose held-out data
    has zero probability contributes ``-inf`` unless ``smoothing > 0``.

    Parameters
    ----------
    gap
        For contiguous blocks, drop this many symbols from the training data on
        each side adjacent to the held-out block. Neighboring
        blocks of a dependent sequence are correlated, so without a gap the
        held-out score is optimistic (buffered or "h-block" cross-validation
        :cite:`Burman1994`). A gap of the order of the process's memory suffices.
    smoothing
        Mix each one-step held-out prediction with the uniform distribution over
        the observed alphabet, with this weight. Then a single transition that the
        fitted model forbids costs ``log2(smoothing / |A|)`` instead of making the
        whole fold ``-inf``, so models can still be compared.
    """
    if gap < 0:
        raise ValueError("gap must be non-negative")
    if not 0.0 <= smoothing < 1.0:
        raise ValueError("smoothing must be in [0, 1)")
    generator = rng if isinstance(rng, np.random.Generator) else np.random.default_rng(rng)
    sequences = _normalize_sequences(data)
    if folds < 2:
        raise ValueError("folds must be at least 2")

    alphabet_size = max(1, len({symbol for seq in sequences for symbol in seq}))
    if len(sequences) >= folds:
        partition = _fold_indices(len(sequences), folds, generator)
        splits = [
            (
                [sequences[i] for i in idx],
                [sequences[i] for j, other in enumerate(partition) if j != f for i in other],
            )
            for f, idx in enumerate(partition)
        ]
    else:
        if _total_length(sequences) < folds:
            raise ValueError("not enough data for the requested number of folds")
        # Few long sequences: split each one into ``folds`` contiguous blocks and hold out
        # block ``f`` of every sequence, so no fold joins two sequences end to end.
        bounds = [np.linspace(0, len(seq), folds + 1).astype(int) for seq in sequences]
        splits = []
        for f in range(folds):
            held: list[list[Any]] = []
            train: list[list[Any]] = []
            for seq, edges in zip(sequences, bounds, strict=True):
                for index in range(folds):
                    start, stop = int(edges[index]), int(edges[index + 1])
                    if index == f:
                        if stop > start:
                            held.append(seq[start:stop])
                        continue
                    if index == f - 1:
                        stop = max(start, stop - gap)
                    elif index == f + 1:
                        start = min(stop, start + gap)
                    if stop > start:
                        train.append(seq[start:stop])
            splits.append((held, train))

    total = 0.0
    for held_out, train in splits:
        model = fit(train)
        if smoothing > 0.0:
            total += sum(
                _smoothed_log_likelihood(model, seq, smoothing=smoothing, alphabet_size=alphabet_size)
                for seq in held_out
            )
        else:
            total += _total_log_likelihood(model, held_out)
    return float(total)


# --- WAIC ------------------------------------------------------------------


def waic(pointwise_log_likelihoods: np.ndarray) -> WAICResult:
    r"""Widely applicable information criterion from posterior samples.

    ``pointwise_log_likelihoods`` has shape ``(n_samples, n_points)`` with entry
    ``[s, i] = log2 p(y_i | theta_s)`` (bits) for posterior draw ``theta_s``, as
    returned by :func:`posterior_pointwise_log_likelihoods`. Returns the WAIC on
    the standard (natural-log) deviance scale (lower is better),
    ``WAIC = -2 (lppd - p_waic)`` with the log pointwise predictive density
    ``lppd = sum_i log mean_s p(y_i | theta_s)`` and effective parameter count
    ``p_waic = sum_i Var_s log p(y_i | theta_s)`` :cite:`Watanabe2010`. ``waic``,
    ``p_waic``, and ``standard_error`` use natural logs; ``lppd`` is in bits.
    """
    from scipy.special import logsumexp

    matrix = np.asarray(pointwise_log_likelihoods, dtype=float)
    if matrix.ndim != 2 or matrix.size == 0:
        raise ValueError("pointwise_log_likelihoods must be a non-empty (n_samples, n_points) array")
    matrix = matrix * np.log(2.0)
    n_samples = matrix.shape[0]
    lppd_pointwise = logsumexp(matrix, axis=0) - np.log(n_samples)
    p_waic_pointwise = matrix.var(axis=0, ddof=1) if n_samples > 1 else np.zeros(matrix.shape[1])
    elpd_pointwise = lppd_pointwise - p_waic_pointwise
    waic_value = -2.0 * float(elpd_pointwise.sum())
    n_points = matrix.shape[1]
    standard_error = float(np.sqrt(n_points * np.var(-2.0 * elpd_pointwise, ddof=0))) if n_points > 1 else 0.0
    return WAICResult(
        waic=waic_value,
        lppd=float(lppd_pointwise.sum() / np.log(2.0)),
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

    A sampled machine starts in the sampled start state of the posterior's own
    training sequence, which says nothing about where an arbitrary scored
    sequence starts; each sequence is therefore scored with its start state
    marginalized over the sampled machine's stationary distribution.
    """
    generator = rng if isinstance(rng, np.random.Generator) else np.random.default_rng(rng)
    sequences = _normalize_sequences(data)
    matrix = np.empty((n_samples, len(sequences)), dtype=float)
    for s in range(n_samples):
        _start, model = posterior.generate_sample(rng=generator)
        pi = model.stationary_distribution()
        states = model.reindex().states
        model.initial_distribution = {state: float(mass) for state, mass in zip(states, pi, strict=True) if mass > 0}
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
        from sofic.inference.bayesian.epsilon import EpsilonMachinePosterior, _pooled_posterior_mean_machine

        if len(sequences) == 1:
            return EpsilonMachinePosterior(machine, sequences[0]).posterior_mean_machine()
        return _pooled_posterior_mean_machine(machine, sequences)
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
