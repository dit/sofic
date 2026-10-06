"""Diagnostics for reconstructed machines: goodness of fit and structural stability.

:func:`goodness_of_fit` asks whether a fitted machine reproduces the observed
word statistics, via a parametric bootstrap from the machine itself
:cite:`Efron1993`. :func:`structure_stability` and :func:`reconstruction_sweep`
ask whether the reconstructed *structure* survives resampling the data or
changing the reconstruction's tuning parameters.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Hashable, Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np

from sofic.exceptions import StochasticValidationError
from sofic.graph import ATTR_EMISSION

__all__ = [
    "GoodnessOfFit",
    "StructureStability",
    "goodness_of_fit",
    "reconstruction_sweep",
    "structure_stability",
    "topology_key",
]


@dataclass(frozen=True)
class GoodnessOfFit:
    """The result of :func:`goodness_of_fit`.

    Attributes
    ----------
    statistic
        ``"g"`` or ``"entropy_rate"``.
    block_length
        The word length compared.
    value
        The statistic on the observed data.
    pvalue
        ``(1 + #{simulated >= observed}) / (1 + n_samples)``. Small values mean
        the machine does not reproduce the data's length-``block_length`` statistics.
    null
        The statistic on each sequence simulated from the machine.
    forbidden_words
        Observed words that the machine assigns probability zero.
    """

    statistic: str
    block_length: int
    value: float
    pvalue: float
    null: np.ndarray
    forbidden_words: tuple[tuple[Any, ...], ...] = ()


def _default_word_length(n: int, alphabet_size: int) -> int:
    """The longest ``L`` with about ten observations per possible word, from 1 to 6."""
    k = max(2, alphabet_size)
    return max(1, min(6, int(np.log(max(n, 1) / 10) / np.log(k))))


def _word_counts(sequence: Sequence[Any], L: int) -> Counter[tuple[Any, ...]]:
    seq = tuple(sequence)
    return Counter(seq[i : i + L] for i in range(len(seq) - L + 1))


def _conditional_entropy(counts: Counter[tuple[Any, ...]]) -> float:
    """Plug-in ``H[X_{L-1} | X_{0:L-1}]`` in bits from length-``L`` word counts."""
    total = sum(counts.values())
    prefixes: Counter[tuple[Any, ...]] = Counter()
    for word, count in counts.items():
        prefixes[word[:-1]] += count
    h = -sum(c * np.log2(c) for c in counts.values()) + sum(c * np.log2(c) for c in prefixes.values())
    return float(h / total)


def _model_conditional_entropy(probabilities: dict[tuple[Any, ...], float]) -> float:
    prefixes: Counter[tuple[Any, ...]] = Counter()
    for word, p in probabilities.items():
        prefixes[word[:-1]] += p
    joint = -sum(p * np.log2(p) for p in probabilities.values() if p > 0)
    marginal = -sum(p * np.log2(p) for p in prefixes.values() if p > 0)
    return float(joint - marginal)


def goodness_of_fit(
    machine: Any,
    data: Sequence[Any],
    *,
    block_length: int | None = None,
    statistic: Literal["g", "entropy_rate"] = "g",
    n_samples: int = 199,
    burn_in: int = 100,
    rng: np.random.Generator | int | None = None,
) -> GoodnessOfFit:
    """Parametric-bootstrap test that ``machine`` generated ``data``.

    Sequences as long as ``data`` are simulated from ``machine`` (after
    ``burn_in`` steps, so they start near stationarity), and a length-``block_length``
    word statistic of the data is compared with its distribution over the simulations.
    Because the null distribution is simulated, the overlap between successive
    words is accounted for; no chi-squared approximation is used.

    Parameters
    ----------
    machine
        A fitted generator with ``sample``, ``word_probabilities`` and
        ``stationary_distribution`` (e.g. an ε-machine from :func:`learn_epsilon_machine_cssr`).
    data
        The observed sequence the machine was fitted to.
    block_length
        Word length; by default the longest with about ten observations per
        possible word (between 1 and 6).
    statistic
        ``"g"`` is the G statistic of the observed length-``block_length`` word counts
        against the machine's stationary word probabilities. ``"entropy_rate"`` is
        ``|h_hat - h_L|``, the gap between the plug-in conditional entropy
        ``H[X_{L-1} | X_{0:L-1}]`` and the machine's value.
    n_samples
        Number of simulated sequences.
    burn_in
        Steps discarded at the start of each simulation.
    rng
        Seed or generator.

    Returns
    -------
    GoodnessOfFit

    Notes
    -----
    Fitting and testing on the same data makes the test conservative, as in any
    parametric bootstrap without refitting. A small p-value is still evidence
    that the reconstruction misses structure. For CSSR that usually means
    ``max_history`` is shorter than the source's synchronization length, which happens
    for strictly sofic sources. An observed word that the machine forbids gives
    ``G = inf`` and the smallest possible p-value.
    """
    generator = rng if isinstance(rng, np.random.Generator) else np.random.default_rng(rng)
    seq = tuple(data)
    n = len(seq)
    alphabet = set(seq) | set(machine.observation_alphabet)
    if block_length is None:
        block_length = _default_word_length(n, len(alphabet))
    if block_length < 1 or n < block_length:
        raise ValueError("block_length must be between 1 and len(data)")
    pi = np.asarray(machine.stationary_distribution(), dtype=float)
    probabilities = {tuple(w): float(p) for w, p in machine.word_probabilities(block_length, start=pi).items()}

    if statistic == "g":

        def compute(sample: Sequence[Any]) -> float:
            counts = _word_counts(sample, block_length)
            total = sum(counts.values())
            g = 0.0
            for word, count in counts.items():
                p = probabilities.get(word, 0.0)
                if p <= 0.0:
                    return float("inf")
                g += count * np.log(count / (total * p))
            return 2.0 * g
    elif statistic == "entropy_rate":
        target = _model_conditional_entropy(probabilities)

        def compute(sample: Sequence[Any]) -> float:
            return abs(_conditional_entropy(_word_counts(sample, block_length)) - target)
    else:
        raise ValueError(f"unknown statistic {statistic!r}")

    value = compute(seq)
    null = np.empty(n_samples)
    for i in range(n_samples):
        simulated, _ = machine.sample(n + burn_in, generator)
        null[i] = compute(simulated[burn_in:])
    tol = 1e-12 * max(1.0, abs(value)) if np.isfinite(value) else 0.0
    pvalue = float((1 + np.sum(null >= value - tol)) / (1 + n_samples))
    forbidden = tuple(
        sorted((w for w in _word_counts(seq, block_length) if probabilities.get(w, 0.0) <= 0.0), key=repr)
    )
    return GoodnessOfFit(statistic, int(block_length), float(value), pvalue, null, forbidden)


def topology_key(machine: Any) -> tuple[int, tuple[tuple[int, str, int], ...]]:
    """An isomorphism-invariant signature of a unifilar machine's labeled topology.

    States are relabeled in breadth-first order from each start state, following
    symbols in ``repr`` order, and the lexicographically smallest edge list is
    kept. Two unifilar machines have the same key exactly when their labeled
    transition graphs are isomorphic; transition probabilities are ignored.
    """
    states = list(machine.states())
    out = {
        state: sorted(
            ((repr(t.data.get(ATTR_EMISSION)), t.target) for t in machine.graph.out_transitions(state)),
            key=lambda item: (item[0], repr(item[1])),
        )
        for state in states
    }
    best: tuple[tuple[int, str, int], ...] | None = None
    for start in states:
        labels: dict[Hashable, int] = {start: 0}
        order = [start]
        index = 0
        while index < len(order):
            for _symbol, target in out[order[index]]:
                if target not in labels:
                    labels[target] = len(labels)
                    order.append(target)
            index += 1
        for state in sorted(states, key=repr):
            labels.setdefault(state, len(labels))
        key = tuple(sorted((labels[s], symbol, labels[t]) for s in states for symbol, t in out[s]))
        if best is None or key < best:
            best = key
    return len(states), best or ()


@dataclass
class StructureStability:
    """The result of :func:`structure_stability`.

    Attributes
    ----------
    reference
        :func:`topology_key` of the reconstruction from the full data.
    topologies
        How often each topology was reconstructed across resamples.
    state_counts
        How often each number of states was reconstructed.
    failures
        Resamples on which reconstruction raised an error.
    """

    reference: tuple[int, tuple[tuple[int, str, int], ...]]
    topologies: Counter = field(default_factory=Counter)
    state_counts: Counter = field(default_factory=Counter)
    failures: int = 0

    @property
    def n_resamples(self) -> int:
        return sum(self.topologies.values()) + self.failures

    @property
    def reference_fraction(self) -> float:
        """Fraction of resamples reproducing the full-data topology."""
        return self.topologies.get(self.reference, 0) / max(1, self.n_resamples)

    @property
    def modal_topology(self) -> tuple[int, tuple[tuple[int, str, int], ...]] | None:
        return self.topologies.most_common(1)[0][0] if self.topologies else None


Method = Literal["cssr", "subtree", "spectral"]


def _reconstruct(sequence: Sequence[Any], method: Method | Callable[..., Any], kwargs: dict[str, Any]) -> Any:
    if isinstance(method, str):
        from sofic.generators.epsilon_machine import EpsilonMachine

        return EpsilonMachine.from_sequence(sequence, method=method, **kwargs)
    return method(sequence, **kwargs)


def structure_stability(
    sequence: Sequence[Any],
    *,
    method: Method | Callable[..., Any] = "cssr",
    n_resamples: int = 50,
    resample: Literal["subsample", "block"] = "subsample",
    fraction: float = 0.5,
    mean_block_length: float | None = None,
    rng: np.random.Generator | int | None = None,
    **kwargs: Any,
) -> StructureStability:
    """How often the reconstructed topology survives resampling the data.

    Parameters
    ----------
    sequence
        The observed sequence.
    method
        Reconstruction method passed to
        :meth:`~sofic.generators.epsilon_machine.EpsilonMachine.from_sequence`,
        or a callable ``method(sequence, **kwargs)`` returning a machine.
    n_resamples
        Number of resampled reconstructions.
    resample
        ``"subsample"`` reconstructs from random contiguous segments of
        ``fraction * len(sequence)`` symbols, which contain no artificial
        junctions :cite:`Politis1999`. ``"block"`` uses the stationary bootstrap
        :cite:`Politis1994` (via :func:`dit.inference.stationary_bootstrap`).
        Block junctions create words the source never emits, which can add
        spurious states.
    fraction
        Segment length for ``"subsample"``, as a fraction of the data.
    mean_block_length
        Mean block length for ``"block"``.
    rng
        Seed or generator.
    **kwargs
        Forwarded to the reconstruction (e.g. ``max_history``, ``alpha``).

    Returns
    -------
    StructureStability

    Notes
    -----
    Subsamples are shorter than the data, so they detect less structure; a
    topology that appears in most half-length subsamples is well supported, while
    one that rarely reappears reflects the particular sample. Statistical
    confidence in an inferred *structure* is not otherwise quantified by CSSR.
    """
    generator = rng if isinstance(rng, np.random.Generator) else np.random.default_rng(rng)
    seq = list(sequence)
    n = len(seq)
    result = StructureStability(reference=topology_key(_reconstruct(seq, method, kwargs)))
    if resample == "subsample":
        if not 0.0 < fraction <= 1.0:
            raise ValueError("fraction must be in (0, 1]")
        m = max(2, int(round(fraction * n)))
        starts = generator.integers(0, n - m + 1, size=n_resamples)
        replicates: Iterable[Sequence[Any]] = (seq[s : s + m] for s in starts)
    elif resample == "block":
        import dit.inference

        stationary_bootstrap = getattr(dit.inference, "stationary_bootstrap", None)
        if stationary_bootstrap is None:  # pragma: no cover - depends on the installed dit
            raise ImportError('resample="block" requires a dit release with dit.inference.stationary_bootstrap')
        array = np.empty(n, dtype=object)
        array[:] = seq
        replicates = (list(r) for r in stationary_bootstrap(array, n_resamples, mean_block_length, generator))
    else:
        raise ValueError(f"unknown resample {resample!r}")
    for replicate in replicates:
        try:
            machine = _reconstruct(replicate, method, kwargs)
        except (StochasticValidationError, ValueError):
            result.failures += 1
            continue
        key = topology_key(machine)
        result.topologies[key] += 1
        result.state_counts[key[0]] += 1
    return result


def reconstruction_sweep(
    sequence: Sequence[Any],
    *,
    alphas: Sequence[float] = (0.05, 0.01, 0.001),
    max_histories: Sequence[int] = (1, 2, 3, 4),
    method: Literal["cssr"] | Callable[..., Any] = "cssr",
    **kwargs: Any,
) -> dict[tuple[float, int], tuple[int, tuple[tuple[int, str, int], ...]] | None]:
    """Reconstruct over a grid of significance levels and history lengths.

    Returns ``{(alpha, max_history): topology_key}``, with ``None`` where reconstruction
    failed. A structure that persists across a range of ``alpha`` and ``max_history`` is
    better supported than one that appears at a single setting. For a Markov
    source, it should be stable for every ``max_history`` at or above the source's order.
    """
    results: dict[tuple[float, int], tuple[int, tuple[tuple[int, str, int], ...]] | None] = {}
    for alpha in alphas:
        for history in max_histories:
            try:
                machine = _reconstruct(sequence, method, {**kwargs, "alpha": alpha, "max_history": history})
            except (StochasticValidationError, ValueError):
                results[alpha, history] = None
                continue
            results[alpha, history] = topology_key(machine)
    return results
