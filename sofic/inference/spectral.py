"""Spectral (method-of-moments) learning of stochastic processes.

This module learns a weighted finite automaton (WFA) / observable-operator
model :cite:`Jaeger2000` for a stationary, discrete-time, discrete-alphabet
process from sampled sequences, using the Hankel-matrix singular value
decomposition of Balle, Carreras, Luque & Quattoni :cite:`Balle2014`. The
approach is the automata-theoretic twin of the spectral HMM algorithm of Hsu,
Kakade & Zhang :cite:`Hsu2012`; unlike Baum-Welch it is a consistent one-shot
estimator with no local optima, and the model order is read off from the
singular-value spectrum rather than fixed in advance.

The learned model is returned as a :class:`~sofic.generators.quasi_realization.QuasiRealization`
-- sofic's native matrix representation of an observable-operator model -- whose
``word_probability`` implements the WFA recursion ``pi @ A_{x1} @ ... @ A_{xt} @ tau``
directly. Because the observable-operator representation is *signed*, this is
always well defined even when no non-negative (hidden Markov) realization of the
same rank exists; :func:`project_to_nmachine` and :func:`project_to_mealy`
provide a best-effort cleanup back to an :class:`~sofic.generators.nmachine.NMachine`
or a stochastic :class:`~sofic.generators.mealy.MealyHMM`.

Small-alphabet note
-------------------
The single-symbol spectral HMM of :cite:`Hsu2012` requires the observation
matrix to have full column rank, i.e. at least as many symbols as hidden states.
The Hankel formulation used here sidesteps that by indexing the moments with
multi-symbol *prefixes* and *suffixes*: increasing ``prefix_length`` /
``suffix_length`` is the discrete analogue of observation stacking and lets the
method recover processes whose alphabet is smaller than the number of states
(golden-mean, even process, ...).
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Hashable, Iterable, Sequence
from typing import Any

import numpy as np

from sofic.generators.quasi_realization import QuasiRealization

__all__ = [
    "SpectralInferenceError",
    "hankel_matrices",
    "learn_spectral_wfa",
    "project_to_epsilon_machine",
    "project_to_mealy",
    "project_to_nmachine",
    "spectral_singular_values",
]

_BELIEF_DECIMALS = 6
_MASS_ATOL = 1e-12


class SpectralInferenceError(ValueError):
    """Raised when spectral learning or projection cannot proceed."""


def _normalize_sequences(sequences: Iterable[Any]) -> list[tuple[Any, ...]]:
    """Normalize ``sequences`` to a list of observation tuples.

    Accepts either a single flat observation sequence (e.g. ``[0, 1, 0]``) or an
    iterable of sequences (e.g. ``[[0, 1], [1, 0]]``), mirroring
    :func:`sofic.generators.hmm_inference.baum_welch`.
    """
    seqs = list(sequences)
    if not seqs:
        return []
    first = seqs[0]
    if isinstance(first, (list, tuple)) and not isinstance(first, (str, bytes)):
        return [tuple(seq) for seq in seqs]
    return [tuple(seqs)]


def _sorted_alphabet(alphabet: Iterable[Any]) -> tuple[Any, ...]:
    return tuple(sorted(set(alphabet), key=repr))


def _words_up_to(alphabet: Sequence[Any], max_length: int) -> list[tuple[Any, ...]]:
    """Return every word over ``alphabet`` of length ``0 .. max_length``.

    The empty word is first, so index ``0`` always addresses the ``epsilon``
    row/column used for the initial and final WFA vectors.
    """
    words: list[tuple[Any, ...]] = [()]
    frontier: list[tuple[Any, ...]] = [()]
    for _ in range(max_length):
        extended = [word + (symbol,) for word in frontier for symbol in alphabet]
        words.extend(extended)
        frontier = extended
    return words


def _empirical_word_probability(
    sequences: Sequence[tuple[Any, ...]],
    max_length: int,
) -> Callable[[Sequence[Any]], float]:
    """Return a stationary block-probability estimator ``f(w)``.

    ``f(w)`` is the fraction of length-``|w|`` sliding windows (across all
    sequences) equal to ``w``, so ``sum_{|w|=L} f(w) = 1`` for each ``L`` and
    ``f(epsilon) = 1``. This is the standard substring estimator used for
    spectral learning of stochastic processes :cite:`Balle2014`.
    """
    counts: dict[tuple[Any, ...], int] = defaultdict(int)
    windows: dict[int, int] = defaultdict(int)
    for sequence in sequences:
        n = len(sequence)
        for length in range(1, max_length + 1):
            count = n - length + 1
            if count <= 0:
                break
            windows[length] += count
            for start in range(count):
                counts[sequence[start : start + length]] += 1

    def f(word: Sequence[Any]) -> float:
        word = tuple(word)
        length = len(word)
        if length == 0:
            return 1.0
        total = windows.get(length, 0)
        if total == 0:
            return 0.0
        return counts.get(word, 0) / total

    return f


def hankel_matrices(
    word_probability: Callable[[Sequence[Any]], float],
    alphabet: Sequence[Any],
    *,
    prefix_length: int,
    suffix_length: int,
) -> tuple[np.ndarray, dict[Any, np.ndarray], list[tuple[Any, ...]], list[tuple[Any, ...]]]:
    """Build the Hankel matrix ``H`` and per-symbol shifted matrices ``H_sigma``.

    ``H[u, v] = word_probability(u + v)`` over the prefix basis (words up to
    ``prefix_length``) and suffix basis (words up to ``suffix_length``), and
    ``H_sigma[u, v] = word_probability(u + (sigma,) + v)``.
    """
    prefixes = _words_up_to(alphabet, prefix_length)
    suffixes = _words_up_to(alphabet, suffix_length)
    n_prefixes, n_suffixes = len(prefixes), len(suffixes)

    hankel = np.zeros((n_prefixes, n_suffixes), dtype=float)
    for i, prefix in enumerate(prefixes):
        for j, suffix in enumerate(suffixes):
            hankel[i, j] = word_probability(prefix + suffix)

    shifted: dict[Any, np.ndarray] = {}
    for symbol in alphabet:
        matrix = np.zeros((n_prefixes, n_suffixes), dtype=float)
        for i, prefix in enumerate(prefixes):
            for j, suffix in enumerate(suffixes):
                matrix[i, j] = word_probability(prefix + (symbol,) + suffix)
        shifted[symbol] = matrix
    return hankel, shifted, prefixes, suffixes


def _select_rank(singular_values: np.ndarray, relative_threshold: float, min_singular_value: float) -> int:
    if singular_values.size == 0:
        return 1
    cutoff = max(min_singular_value, relative_threshold * float(singular_values[0]))
    keep = int(np.count_nonzero(singular_values > cutoff))
    return max(1, keep)


def _consistency_gauge(
    pi: np.ndarray,
    tau: np.ndarray,
    symbol_maps: dict[Any, np.ndarray],
) -> tuple[np.ndarray, dict[Any, np.ndarray]]:
    """Transform to the ``tau = 1`` gauge used by :class:`QuasiRealization`.

    Applies an invertible similarity ``C`` with ``C @ 1 = tau`` (rank-one
    Sherman-Morrison update) so that in the new basis the final vector is the
    all-ones vector, the initial vector sums to ``pi @ tau = f(epsilon) = 1``,
    and per-state outgoing masses sum to one. Word probabilities are preserved
    exactly.
    """
    n = tau.shape[0]
    ones = np.ones(n, dtype=float)
    k = int(np.argmax(np.abs(tau)))
    if abs(tau[k]) < 1e-12:
        raise SpectralInferenceError("degenerate final vector; cannot normalize spectral model")
    # C = I + (tau - 1) e_k^T  =>  C @ 1 = tau ; C^{-1} = I - (tau - 1) e_k^T / tau_k
    update = tau - ones
    gauge = np.eye(n) + np.outer(update, np.eye(n)[k])
    gauge_inv = np.eye(n) - np.outer(update, np.eye(n)[k]) / tau[k]

    pi_gauged = pi @ gauge
    maps_gauged = {symbol: gauge_inv @ matrix @ gauge for symbol, matrix in symbol_maps.items()}
    return pi_gauged, maps_gauged


def learn_spectral_wfa(
    sequences: Iterable[Any] | None = None,
    *,
    word_probability: Callable[[Sequence[Any]], float] | None = None,
    alphabet: Iterable[Any] | None = None,
    rank: int | None = None,
    prefix_length: int = 2,
    suffix_length: int | None = None,
    singular_value_threshold: float = 1e-3,
    min_singular_value: float = 1e-12,
) -> QuasiRealization:
    """Learn a WFA / observable-operator model by Hankel-matrix SVD.

    Parameters
    ----------
    sequences
        A single observed realization (e.g. ``[0, 1, 0, ...]``) or an iterable
        of realizations. Ignored when ``word_probability`` is given.
    word_probability
        Optional exact block-probability function ``f(word) -> float`` (e.g.
        :meth:`~sofic.generators.base.HiddenMarkovModel.word_probability`). When
        supplied the Hankel matrix is built exactly instead of from samples;
        ``alphabet`` is then required.
    alphabet
        Observation alphabet. Inferred from ``sequences`` when omitted.
    rank
        Number of latent states. When ``None`` the rank is chosen from the
        singular-value spectrum (values exceeding
        ``singular_value_threshold`` times the largest).
    prefix_length, suffix_length
        Maximum lengths of the prefix and suffix bases. ``suffix_length``
        defaults to ``prefix_length``. Larger values are the small-alphabet
        remedy (see module docstring).
    singular_value_threshold
        Relative cutoff for automatic rank selection.
    min_singular_value
        Absolute floor below which singular values are treated as zero.

    Returns
    -------
    QuasiRealization
        A consistent observable-operator model (``tau`` is the all-ones vector,
        ``pi`` sums to one) whose ``word_probability`` reproduces the learned
        statistics.
    """
    if suffix_length is None:
        suffix_length = prefix_length
    if prefix_length < 1 or suffix_length < 1:
        raise SpectralInferenceError("prefix_length and suffix_length must be at least 1")

    seqs: list[tuple[Any, ...]] = []
    if word_probability is None:
        if sequences is None:
            raise SpectralInferenceError("provide either sequences or word_probability")
        seqs = _normalize_sequences(sequences)
        if not any(seqs):
            raise SpectralInferenceError("sequences contain no symbols")

    if alphabet is not None:
        symbols = _sorted_alphabet(alphabet)
    elif seqs:
        symbols = _sorted_alphabet(symbol for sequence in seqs for symbol in sequence)
    else:
        raise SpectralInferenceError("alphabet is required when learning from word_probability")
    if not symbols:
        raise SpectralInferenceError("empty alphabet")

    f = word_probability
    if f is None:
        f = _empirical_word_probability(seqs, prefix_length + suffix_length + 1)

    hankel, shifted, prefixes, _suffixes = hankel_matrices(
        f, symbols, prefix_length=prefix_length, suffix_length=suffix_length
    )

    u_full, s_full, vt_full = np.linalg.svd(hankel, full_matrices=False)
    available = int(np.count_nonzero(s_full > min_singular_value))
    if available == 0:
        raise SpectralInferenceError("Hankel matrix is numerically zero; no signal to learn")
    if rank is None:
        rank = _select_rank(s_full, singular_value_threshold, min_singular_value)
    rank = max(1, min(int(rank), available))

    u_n = u_full[:, :rank]
    s_n = s_full[:rank]
    v_n = vt_full[:rank, :].T
    inv_s = 1.0 / s_n

    h_prefix = hankel[:, 0]  # empty-suffix column: f(u)
    h_suffix = hankel[0, :]  # empty-prefix row: f(v)

    pi = (h_suffix @ v_n) * inv_s
    tau = u_n.T @ h_prefix
    symbol_maps = {symbol: (u_n.T @ shifted[symbol] @ v_n) * inv_s[None, :] for symbol in symbols}

    pi, symbol_maps = _consistency_gauge(pi, tau, symbol_maps)
    tau = np.ones(rank, dtype=float)

    total = float(pi.sum())
    if abs(total) < 1e-12:
        raise SpectralInferenceError("degenerate initial vector; cannot normalize spectral model")
    pi = pi / total

    return QuasiRealization(pi=pi, tau=tau, symbol_maps=symbol_maps)


def spectral_singular_values(
    sequences: Iterable[Any] | None = None,
    *,
    word_probability: Callable[[Sequence[Any]], float] | None = None,
    alphabet: Iterable[Any] | None = None,
    prefix_length: int = 2,
    suffix_length: int | None = None,
) -> np.ndarray:
    """Return the Hankel singular-value spectrum used for model-order selection.

    A clear gap in the returned values indicates the effective number of latent
    states; pass the resulting count as ``rank`` to :func:`learn_spectral_wfa`.
    """
    if suffix_length is None:
        suffix_length = prefix_length
    if alphabet is not None:
        symbols = _sorted_alphabet(alphabet)
        f = word_probability
        if f is None:
            if sequences is None:
                raise SpectralInferenceError("provide sequences or word_probability")
            seqs = _normalize_sequences(sequences)
            f = _empirical_word_probability(seqs, prefix_length + suffix_length + 1)
    else:
        if sequences is None:
            raise SpectralInferenceError("alphabet is required when learning from word_probability")
        seqs = _normalize_sequences(sequences)
        symbols = _sorted_alphabet(symbol for sequence in seqs for symbol in sequence)
        f = _empirical_word_probability(seqs, prefix_length + suffix_length + 1)
    hankel, _shifted, _prefixes, _suffixes = hankel_matrices(
        f, symbols, prefix_length=prefix_length, suffix_length=suffix_length
    )
    return np.linalg.svd(hankel, compute_uv=False)


def _state_labels(n: int) -> list[Hashable]:
    from sofic.states import sequential_labels

    return list(sequential_labels(n))


def project_to_nmachine(qr: QuasiRealization, *, tol: float = 1e-9, validate: bool = True) -> Any:
    """Project a learned :class:`QuasiRealization` onto an :class:`NMachine`.

    Reads each operator entry ``A_sigma[i, j]`` as the signed joint
    quasiprobability ``P(state_j, sigma | state_i)`` and renormalizes per-state
    outgoing mass to one, producing an observable-operator generator with an
    explicit transition graph. Small-magnitude edges (``|w| <= tol``) are
    dropped. The result may carry signed weights (it is an n-machine, not
    necessarily a hidden Markov model); use :func:`project_to_mealy` when a
    non-negative realization is required.
    """
    from sofic.generators.nmachine import NMachine
    from sofic.graph import ATTR_EMISSION, ATTR_QUASIPROB

    symbol_maps = qr.symbol_maps
    n = qr.pi.shape[0]
    labels = _state_labels(n)
    alphabet = sorted(symbol_maps, key=repr)

    row_totals = np.zeros(n, dtype=float)
    for matrix in symbol_maps.values():
        row_totals += matrix.sum(axis=1)

    machine = NMachine(observation_alphabet=frozenset(alphabet))
    for label in labels:
        machine.graph.add_state(label)
    for i in range(n):
        scale = row_totals[i]
        if abs(scale) < 1e-12:
            continue
        for symbol in alphabet:
            matrix = symbol_maps[symbol]
            for j in range(n):
                weight = matrix[i, j] / scale
                if abs(weight) <= tol:
                    continue
                machine.graph.add_transition(
                    labels[i], labels[j], **{ATTR_QUASIPROB: float(weight), ATTR_EMISSION: symbol}
                )

    pi = np.asarray(qr.pi, dtype=float)
    pi_total = float(pi.sum())
    if abs(pi_total) < 1e-12:
        raise SpectralInferenceError("degenerate initial distribution; cannot build n-machine")
    machine.initial_quasidistribution = {labels[i]: float(pi[i] / pi_total) for i in range(n) if abs(pi[i]) > tol}
    if validate:
        machine.validate()
    return machine


def project_to_mealy(qr: QuasiRealization, *, tol: float = 1e-8, validate: bool = True) -> Any:
    """Project a learned :class:`QuasiRealization` onto a stochastic ``MealyHMM``.

    Succeeds only when the observable-operator model admits a non-negative
    realization in the current basis: any operator entry below ``-tol`` raises
    :class:`SpectralInferenceError`. Small negatives are clipped to zero and each
    state's outgoing mass is renormalized to one. This is a best-effort cleanup;
    a signed model should be kept as an :class:`~sofic.generators.nmachine.NMachine`
    via :func:`project_to_nmachine`.
    """
    from sofic.generators.mealy import MealyHMM

    symbol_maps = qr.symbol_maps
    n = qr.pi.shape[0]
    labels = _state_labels(n)
    alphabet = sorted(symbol_maps, key=repr)

    most_negative = min((float(matrix.min()) for matrix in symbol_maps.values()), default=0.0)
    if most_negative < -tol:
        raise SpectralInferenceError(
            f"no non-negative realization in this basis (min operator entry {most_negative:.3g}); "
            "use project_to_nmachine for the signed model"
        )

    clipped = {symbol: np.clip(matrix, 0.0, None) for symbol, matrix in symbol_maps.items()}
    row_totals = np.zeros(n, dtype=float)
    for matrix in clipped.values():
        row_totals += matrix.sum(axis=1)

    pi = np.clip(np.asarray(qr.pi, dtype=float), 0.0, None)
    pi_total = float(pi.sum())
    if pi_total < 1e-12:
        raise SpectralInferenceError("degenerate initial distribution; cannot build Mealy HMM")

    machine = MealyHMM(
        initial_distribution={labels[i]: float(pi[i] / pi_total) for i in range(n) if pi[i] > tol},
        observation_alphabet=frozenset(alphabet),
    )
    for label in labels:
        machine.graph.add_state(label)
    for i in range(n):
        scale = row_totals[i]
        if scale < 1e-12:
            continue
        for symbol in alphabet:
            matrix = clipped[symbol]
            for j in range(n):
                prob = matrix[i, j] / scale
                if prob <= tol:
                    continue
                machine.add_transition(labels[i], labels[j], symbol, float(prob))
    if validate:
        machine.validate()
    return machine


def project_to_epsilon_machine(
    qr: QuasiRealization,
    *,
    tol: float = 1e-8,
    max_states: int = 10_000,
) -> Any:
    """Extract an ε-machine from a learned spectral model.

    When the observable operators admit a non-negative realization in the
    learned basis, this is :func:`project_to_mealy` followed by
    :meth:`~sofic.generators.epsilon_machine.EpsilonMachine.from_hmm`. Signed
    operators are converted by enumerating mixed states of the observable
    operators (belief updates ``b A_x / (b A_x τ)``) and merging
    predictively equivalent recurrent states :cite:`Ellison2009`. This is the
    computational-mechanics extraction, not a clustering heuristic.

    Raises
    ------
    SpectralInferenceError
        If mixed-state enumeration exceeds ``max_states`` or the initial
        vector is degenerate.
    """
    from sofic.generators.epsilon_machine import EpsilonMachine

    try:
        mealy = project_to_mealy(qr, tol=tol, validate=True)
    except SpectralInferenceError:
        mealy = _mealy_from_operator_mixed_states(qr, max_states=max_states)
    return EpsilonMachine.from_hmm(mealy)


def _mealy_from_operator_mixed_states(
    qr: QuasiRealization,
    *,
    max_states: int = 10_000,
    decimals: int = _BELIEF_DECIMALS,
) -> Any:
    """Build a unifilar Mealy HMM whose states are mixed states of ``qr``."""
    from collections import deque

    from sofic.generators.mealy import MealyHMM
    from sofic.generators.mixed_state import MixedState
    from sofic.graph import ATTR_EMISSION, ATTR_PROB, TransitionGraph

    maps = qr.symbol_maps
    tau = np.asarray(qr.tau, dtype=float)
    symbols = tuple(sorted(maps, key=repr))
    eta0 = MixedState.from_vector(qr.pi, decimals=decimals)
    if eta0 is None:
        raise SpectralInferenceError("degenerate initial vector; cannot extract mixed states")

    graph = TransitionGraph()
    discovered: dict[MixedState, MixedState] = {}
    queue: deque[MixedState] = deque()

    def register(state: MixedState) -> MixedState:
        existing = discovered.get(state)
        if existing is not None:
            return existing
        atol = 10 ** (-decimals)
        for known in discovered:
            if all(np.isclose(a, b, rtol=0.0, atol=atol) for a, b in zip(known.belief, state.belief, strict=True)):
                discovered[state] = known
                return known
        if len(discovered) >= max_states:
            raise SpectralInferenceError(
                f"mixed-state extraction exceeded max_states={max_states}; "
                "use project_to_nmachine for the signed observable-operator model"
            )
        discovered[state] = state
        graph.add_state(state)
        queue.append(state)
        return state

    register(eta0)
    while queue:
        eta = queue.popleft()
        row = eta.as_array()
        emissions: list[tuple[Any, MixedState, float]] = []
        for symbol in symbols:
            nxt = row @ maps[symbol]
            prob = float(nxt @ tau)
            if prob <= _MASS_ATOL:
                continue
            successor = MixedState.from_vector(nxt, decimals=decimals)
            if successor is None:
                continue
            emissions.append((symbol, register(successor), prob))
        total = sum(prob for _symbol, _successor, prob in emissions)
        if total <= _MASS_ATOL:
            continue
        for symbol, successor, prob in emissions:
            graph.add_transition(
                eta,
                successor,
                **{ATTR_PROB: float(prob / total), ATTR_EMISSION: symbol},
            )

    keep = graph.terminal_recurrent_states()
    if not keep:
        keep = frozenset(discovered.values())
    recurrent = TransitionGraph()
    for state in keep:
        recurrent.add_state(state)
        for transition in graph.out_transitions(state):
            if transition.target in keep:
                recurrent.add_transition(transition.source, transition.target, **dict(transition.data))
    initial = {eta0: 1.0} if eta0 in keep else {}
    return MealyHMM(
        graph=recurrent,
        initial_distribution=initial,
        observation_alphabet=frozenset(symbols),
    )
