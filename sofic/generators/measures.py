"""Information measures for stochastic generators via dit."""

from __future__ import annotations

import warnings
from collections.abc import Iterator
from typing import Any, Literal, NamedTuple

import numpy as np

from sofic.exceptions import MixedStateExplosionError
from sofic.generators.base import HiddenMarkovModel, QuasiStochasticModel, StochasticModel
from sofic.generators.markov import MarkovChain
from sofic.graph import ATTR_EMISSION, ATTR_PROB

EntropyRateMethod = Literal["auto", "exact", "bounds", "blackwell"]


def require_dit(feature: str = "entropy measures") -> Any:
    """Import and return the :mod:`dit` package, or raise a helpful error.

    ``feature`` names the capability requiring dit and is interpolated into the
    error message when the optional dependency is missing.
    """
    try:
        import dit
    except ImportError as exc:
        raise ImportError(f"dit is required for {feature}; install with `pip install dit`") from exc
    return dit


# Backwards-compatible internal alias.
_require_dit = require_dit


def dit_state_label(state: Any) -> Any:
    """Return a dit-safe single-symbol label for a machine state.

    ``dit.Distribution`` treats each outcome as a sequence of random-variable
    values, so a tuple-valued state (e.g. an edge-machine state like
    ``("A", "0", "A")``) is misread as multi-dimensional coordinate data and
    raises ``MissingDimensionsError``. Tuple states are encoded to a lossless
    string via :func:`sofic.generators.edge_machine.edge_state_label` (invert
    with ``parse_edge_state_label``); mixed states, which dit cannot sort, are
    labeled by the ``repr`` of their belief; other scalar states pass through
    unchanged.
    """
    from sofic.generators.mixed_state import MixedState

    if isinstance(state, MixedState):
        return repr(state.belief)
    if isinstance(state, tuple):
        from sofic.generators.edge_machine import edge_state_label

        return edge_state_label(state)
    return state


def state_distribution(model: StochasticModel) -> Any:
    """Return the stationary state law as a ``dit.Distribution``.

    States are emitted as dit-safe labels (see :func:`dit_state_label`): scalar
    states are preserved verbatim, tuple states are encoded to a lossless string.
    """
    dit = _require_dit()
    idx = model.reindex()
    pi = model.stationary_distribution()
    outcomes = [(dit_state_label(idx.state(i)),) for i in range(len(idx))]
    from sofic.generators.prob import as_prob, has_symbolic, simplify_prob

    probs = [as_prob(pi[i]) for i in range(len(idx))]
    if has_symbolic(probs):
        from dit.symbolic import symbolic_distribution

        return symbolic_distribution(outcomes, [simplify_prob(p) for p in probs])
    return dit.Distribution(outcomes, [float(p) for p in probs])


def state_entropy(model: StochasticModel) -> Any:
    """Shannon entropy of the stationary state distribution in bits."""
    dit = _require_dit()
    dist = state_distribution(model)
    value = dit.shannon.entropy(dist)
    if hasattr(dist, "is_symbolic") and dist.is_symbolic():
        return value
    return float(value)


def joint_block_distribution(
    generator: HiddenMarkovModel,
    block_length: int = 2,
) -> Any:
    """Build a ``dit.Distribution`` over observed emission blocks of ``block_length`` symbols."""
    from sofic.generators.matrices import emission_tensors
    from sofic.generators.words import _enumerate_words, _matrix_step

    if block_length < 1:
        raise ValueError("block_length must be at least 1")
    dit = _require_dit()
    # Blocks of a stationary process are weighted by the stationary state law, not
    # the model's initial distribution (which may describe only the transient).
    pi, joint = emission_tensors(generator, policy="stationary")

    symbol_list = sorted(generator.observation_alphabet, key=repr)
    ones = np.ones(len(pi), dtype=float)
    outcomes = []
    probs = []
    for outcome, mass in _enumerate_words(
        symbol_list, block_length, pi.copy(), _matrix_step(joint, len(pi), prune=False)
    ):
        outcomes.append(outcome)
        probs.append(float(mass @ ones))

    total = sum(probs)
    if total > 0.0:
        probs = [p / total for p in probs]
    return dit.Distribution(outcomes, probs)


def _entropy_rate_from_transitions(
    model: StochasticModel,
    pi: np.ndarray,
    idx: Any,
) -> Any:
    """Entropy rate from edge probabilities when symbol-labeled joint mass is absent.

    Accepts any :class:`StochasticModel` (visible Markov chain or hidden Markov
    model); the target state stands in as the emitted symbol when no emission is set.
    """
    dit = _require_dit()
    from sofic.generators.prob import (
        as_prob,
        has_symbolic,
        is_positive_mass,
        is_symbolic,
        simplify_prob,
    )

    symbolic = pi.dtype == object or has_symbolic(pi.ravel())
    rate: Any = 0 if symbolic else 0.0
    for state in idx.states:
        i = idx.index(state)
        outgoing = list(model.graph.out_transitions(state))
        if not outgoing:
            continue
        targets: list[Any] = []
        probs: list[Any] = []
        for transition in outgoing:
            prob = as_prob(transition.data.get(ATTR_PROB, 0.0))
            if not is_positive_mass(prob):
                continue
            emission = transition.data.get(ATTR_EMISSION)
            target = (transition.target, emission) if emission is not None else transition.target
            targets.append(dit_state_label(target))
            probs.append(prob)
        if not probs:
            continue
        if has_symbolic(probs) or symbolic:
            from dit.symbolic import symbolic_distribution

            conditional = symbolic_distribution(targets, [simplify_prob(p) for p in probs])
            contrib = as_prob(pi[i]) * dit.shannon.entropy(conditional)
            rate = simplify_prob(as_prob(rate) + as_prob(contrib))
        else:
            conditional = dit.Distribution(targets, [float(p) for p in probs])
            rate += float(pi[i] * dit.shannon.entropy(conditional))
    if is_symbolic(rate):
        return simplify_prob(rate)
    return float(rate)


def entropy_rate_hmm(hmm: HiddenMarkovModel) -> Any:
    """Shannon entropy rate for unifilar hidden Markov presentations.

    Returns a sympy :class:`~sympy.Expr` when the stationary law or emission
    tensors are symbolic; otherwise a Python ``float``.
    """
    from sofic.generators.matrices import symbol_matrices
    from sofic.generators.prob import (
        array_sum,
        as_prob,
        has_symbolic,
        is_positive_mass,
        is_symbolic,
        simplify_prob,
        sum_probs,
    )

    is_unifilar = getattr(hmm, "is_unifilar", None)
    if is_unifilar is None or not is_unifilar():
        raise NotImplementedError("entropy_rate_hmm is only exact for unifilar HMM presentations")

    dit = _require_dit()
    idx = hmm.reindex()
    pi = hmm.stationary_distribution()
    joint = symbol_matrices(hmm.to_mealy())

    symbolic = pi.dtype == object or has_symbolic(pi.ravel())
    if not symbolic:
        symbolic = any(matrix.dtype == object or has_symbolic(matrix.ravel()) for matrix in joint.values())

    # Emit dit-safe state labels so tuple-valued states (e.g. edge-machine
    # states like ("A", "0", "A")) do not break dit.Distribution.
    outcomes: list[tuple[Any, Any]] = []
    probs: list[Any] = []
    for state in idx.states:
        i = idx.index(state)
        label = dit_state_label(state)
        for symbol, matrix in joint.items():
            row_mass = as_prob(pi[i]) * array_sum(matrix[i])
            row_mass = simplify_prob(row_mass) if symbolic or is_symbolic(row_mass) else float(row_mass)
            if not is_positive_mass(row_mass):
                continue
            outcomes.append((label, symbol))
            probs.append(row_mass)

    if not probs:
        return (
            entropy_rate_markov(hmm) if isinstance(hmm, MarkovChain) else _entropy_rate_from_transitions(hmm, pi, idx)
        )

    total = sum_probs(probs)
    state_dist = state_distribution(hmm)
    if symbolic or has_symbolic(probs) or is_symbolic(total):
        from dit.symbolic import symbolic_distribution

        joint_dist = symbolic_distribution(
            outcomes,
            [simplify_prob(as_prob(p) / as_prob(total)) for p in probs],
        )
        return simplify_prob(as_prob(dit.shannon.entropy(joint_dist)) - as_prob(dit.shannon.entropy(state_dist)))
    joint_dist = dit.Distribution(outcomes, [float(p) / float(total) for p in probs])
    return float(dit.shannon.entropy(joint_dist) - dit.shannon.entropy(state_dist))


def entropy_rate_markov(chain: MarkovChain) -> Any:
    """Shannon entropy rate of a visible Markov chain in bits.

    A visible Markov chain has no separate emissions, so its entropy rate is the
    conditional-transition entropy computed by :func:`_entropy_rate_from_transitions`
    (which treats the target state as the emitted symbol when no emission is set).
    """
    idx = chain.reindex()
    pi = chain.stationary_distribution()
    return _entropy_rate_from_transitions(chain, pi, idx)


class EntropyRateEstimate(NamedTuple):
    """Monte Carlo entropy-rate estimate in bits per symbol and its batch-means standard error."""

    estimate: float
    stderr: float


def entropy_rate(
    model: StochasticModel,
    method: EntropyRateMethod = "auto",
    *,
    tol: float = 1e-6,
    max_length: int = 30,
    max_words: int = 2**20,
    max_states: int = 1_000,
    n_samples: int = 100_000,
    burn_in: int = 1_000,
    seed: int | np.random.Generator | None = None,
) -> Any:
    r"""Shannon entropy rate :math:`h_\mu` in bits per symbol of any finite HMM presentation.

    ``method`` selects the algorithm:

    ``"exact"``
        Unifilar presentations use the closed form over their states
        (:func:`entropy_rate_hmm`); visible Markov chains use
        :func:`entropy_rate_markov`. Non-unifilar presentations are unifilarized
        by mixed-state construction :cite:`Ellison2009` with at most
        ``max_states`` beliefs, and the closed form is applied to its recurrent
        component. Raises :class:`~sofic.exceptions.MixedStateExplosionError`
        when the mixed states do not close, which is the generic case
        :cite:`blackwell1957entropy,jurgens2021shannon`.
    ``"bounds"``
        The midpoint of the sandwich
        :math:`H[X_n \mid X_{0:n}, S_0] \le h_\mu \le H[X_n \mid X_{0:n}]`
        (see :func:`entropy_rate_bounds`), with ``n`` increased until the gap is
        at most ``tol``. A :class:`RuntimeWarning` is issued when the gap still
        exceeds ``tol`` at ``n = max_length`` or once more than ``max_words``
        positive-probability words are enumerated.
    ``"blackwell"``
        The stochastic mixed-state estimate of :func:`entropy_rate_blackwell`
        (``n_samples``, ``burn_in``, ``seed``); call that function directly to
        also obtain its standard error.
    ``"auto"``
        ``"exact"`` when it succeeds, otherwise ``"bounds"``. The result is
        deterministic: the stochastic estimator is never chosen implicitly.

    Exact results are sympy expressions for symbolic presentations; bounds and
    Blackwell estimates are numeric only.
    """
    if method not in ("auto", "exact", "bounds", "blackwell"):
        raise ValueError(f"unknown entropy-rate method {method!r}; expected 'auto', 'exact', 'bounds', or 'blackwell'")
    if method == "blackwell":
        return entropy_rate_blackwell(model, n_samples=n_samples, burn_in=burn_in, seed=seed).estimate
    if method in ("auto", "exact"):
        try:
            return _entropy_rate_exact(model, max_states=max_states)
        except MixedStateExplosionError:
            if method == "exact":
                raise
    return _entropy_rate_from_bounds(model, tol=tol, max_length=max_length, max_words=max_words)


def _entropy_rate_exact(model: StochasticModel, *, max_states: int) -> Any:
    if isinstance(model, MarkovChain):
        return entropy_rate_markov(model)
    if not isinstance(model, HiddenMarkovModel):
        raise TypeError(f"entropy rate requires a HiddenMarkovModel or MarkovChain, not {type(model)!r}")
    is_unifilar = getattr(model, "is_unifilar", None)
    if is_unifilar is not None and is_unifilar():
        return entropy_rate_hmm(model)
    from sofic.generators.mixed_state_construction import build_mixed_state_presentation

    presentation = build_mixed_state_presentation(model.to_mealy(), max_states=max_states)
    return entropy_rate_hmm(presentation.to_recurrent())


def _entropy_rate_from_bounds(model: StochasticModel, *, tol: float, max_length: int, max_words: int) -> float:
    if max_length < 0:
        raise ValueError("max_length must be nonnegative")
    pi, tensors = _numeric_symbol_tensors(model)
    for n, (lower, upper, n_words) in enumerate(_iter_entropy_rate_bounds(pi, tensors)):
        if upper - lower <= tol:
            return 0.5 * (lower + upper)
        if n >= max_length or n_words > max_words:
            break
    warnings.warn(
        f"entropy-rate bounds did not converge to tol={tol:g} by n={n}: [{lower:.10g}, {upper:.10g}]; "
        "returning the midpoint",
        RuntimeWarning,
        stacklevel=3,
    )
    return 0.5 * (lower + upper)


def entropy_rate_bounds(model: StochasticModel, n: int) -> tuple[float, float]:
    r"""Return ``(lower, upper)`` bounds on the entropy rate in bits per symbol.

    For a stationary HMM with hidden states :math:`S_t` and symbols :math:`X_t`,

    .. math::

        H[X_n \mid X_{0:n}, S_0] \;\le\; h_\mu \;\le\; H[X_n \mid X_{0:n}],

    the lower bound is nondecreasing and the upper bound nonincreasing in
    ``n``, and both converge to :math:`h_\mu` (Cover & Thomas, Thm. 4.5.1, there
    stated for functions of Markov chains with 1-based indices
    :cite:`Cover2006`). For edge-emitting presentations :math:`S_0` is the
    state *before* :math:`X_0` is emitted; by the Markov property it screens
    the future off from the infinite past. The upper bound is the block-entropy
    difference :math:`H[X_{0:n+1}] - H[X_{0:n}]`, which for many HMMs converges
    exponentially fast :cite:`Travers2013`.

    Both bounds are exact: every positive-probability word of length ``n + 1``
    is enumerated from the stationary state law, so the cost grows with the
    number of such words (about :math:`2^{n h_\mu}`).
    """
    if n < 0:
        raise ValueError("n must be nonnegative")
    pi, tensors = _numeric_symbol_tensors(model)
    for index, (lower, upper, _n_words) in enumerate(_iter_entropy_rate_bounds(pi, tensors)):
        if index == n:
            return lower, upper
    raise AssertionError("unreachable")


def _iter_entropy_rate_bounds(pi: np.ndarray, tensors: np.ndarray) -> Iterator[tuple[float, float, int]]:
    """Yield ``(H[X_n | X_{0:n}, S_0], H[X_n | X_{0:n}], word count)`` for ``n = 0, 1, ...``.

    Rows of ``backward`` are :math:`T^{(w)} \\mathbf{1}` over positive-probability
    words ``w``; prepending a symbol ``x`` maps a row ``b`` to :math:`T^{(x)} b`.
    """
    n_states = len(pi)
    backward = np.ones((1, n_states))
    block_entropy = 0.0
    state_block_entropy = _entropy_bits(pi)
    while True:
        backward = np.einsum("xij,wj->xwi", tensors, backward).reshape(-1, n_states)
        joint = backward * pi
        words = joint.sum(axis=1)
        keep = words > 0.0
        backward, joint, words = backward[keep], joint[keep], words[keep]
        next_block = _entropy_bits(words)
        next_state_block = _entropy_bits(joint.ravel())
        yield next_state_block - state_block_entropy, next_block - block_entropy, len(words)
        block_entropy, state_block_entropy = next_block, next_state_block


def entropy_rate_blackwell(
    model: StochasticModel,
    *,
    n_samples: int = 100_000,
    burn_in: int = 1_000,
    seed: int | np.random.Generator | None = None,
    n_batches: int = 20,
) -> EntropyRateEstimate:
    r"""Estimate the entropy rate by a random walk on mixed states.

    Blackwell's formula :cite:`blackwell1957entropy` writes the entropy rate as
    the average next-symbol uncertainty over the Blackwell measure
    :math:`\mu_B` on beliefs (mixed states) :math:`\eta`,

    .. math::

        h_\mu = \int H[X \mid \eta] \, d\mu_B(\eta), \qquad
        \Pr(x \mid \eta) = \eta T^{(x)} \mathbf{1}, \qquad
        \eta \mapsto \frac{\eta T^{(x)}}{\eta T^{(x)} \mathbf{1}} .

    Following Jurgens & Crutchfield (:cite:`jurgens2021shannon`, Eqs. 16-17),
    the integral is replaced by a time average along a sampled mixed-state
    trajectory started from the stationary state law. The first ``burn_in``
    steps are discarded; the standard error comes from ``n_batches``
    non-overlapping batch means, which accounts for autocorrelation.

    ``seed`` is an integer seed or a :class:`numpy.random.Generator`.
    """
    if n_batches < 2:
        raise ValueError("n_batches must be at least 2")
    if n_samples < n_batches:
        raise ValueError("n_samples must be at least n_batches")
    pi, tensors = _numeric_symbol_tensors(model)
    rng = seed if isinstance(seed, np.random.Generator) else np.random.default_rng(seed)
    _beliefs, _symbols, probabilities = mixed_state_walk(pi, tensors, n_samples=n_samples, burn_in=burn_in, rng=rng)
    entropies = _row_entropies_bits(probabilities)
    return EntropyRateEstimate(float(entropies.mean()), batch_means_stderr(entropies, n_batches))


def mixed_state_walk(
    pi: np.ndarray,
    tensors: np.ndarray,
    *,
    n_samples: int,
    burn_in: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sample the mixed-state process started from belief ``pi``.

    ``tensors`` stacks the symbol-labeled matrices :math:`T^{(x)}` along axis 0.
    After discarding ``burn_in`` steps, returns the beliefs ``eta_t`` (one row
    per step), the symbol indices ``x_t`` emitted from them, and the
    next-symbol distributions ``Pr(x | eta_t)``.
    """
    if burn_in < 0:
        raise ValueError("burn_in must be nonnegative")
    if n_samples < 1:
        raise ValueError("n_samples must be positive")
    n_symbols, n_states, _ = tensors.shape
    beliefs = np.empty((n_samples, n_states))
    symbols = np.empty(n_samples, dtype=np.intp)
    probabilities = np.empty((n_samples, n_symbols))
    uniforms = rng.random(burn_in + n_samples)
    eta = np.asarray(pi, dtype=float) / float(np.sum(pi))
    for t, u in enumerate(uniforms):
        masses = eta @ tensors
        p = masses.sum(axis=1)
        total = float(p.sum())
        if total <= 0.0:
            raise ValueError("mixed-state walk reached a belief with no outgoing probability")
        p /= total
        x = min(int(np.searchsorted(np.cumsum(p), u, side="right")), n_symbols - 1)
        if t >= burn_in:
            row = t - burn_in
            beliefs[row] = eta
            symbols[row] = x
            probabilities[row] = p
        eta = masses[x] / float(masses[x].sum())
    return beliefs, symbols, probabilities


def batch_means_stderr(values: np.ndarray, n_batches: int) -> float:
    """Standard error of the mean of an autocorrelated stationary series by non-overlapping batch means."""
    size = len(values) // n_batches
    means = np.asarray(values[: size * n_batches], dtype=float).reshape(n_batches, size).mean(axis=1)
    return float(means.std(ddof=1) / np.sqrt(n_batches))


def _numeric_symbol_tensors(model: StochasticModel) -> tuple[np.ndarray, np.ndarray]:
    """Return the stationary state law and the stacked ``T^(x)`` matrices as float arrays.

    A visible Markov chain emits its next state, so ``T^(j)`` keeps column ``j``
    of the transition matrix.
    """
    if isinstance(model, MarkovChain):
        from sofic.properties import transition_matrix

        idx = model.reindex()
        transition, _states = transition_matrix(model, attr=ATTR_PROB, states=idx.states)
        transition = np.asarray(transition, dtype=float)
        pi = np.asarray(model.stationary_distribution(), dtype=float)
        n_states = len(pi)
        tensors = np.zeros((n_states, n_states, n_states))
        for j in range(n_states):
            tensors[j, :, j] = transition[:, j]
        return pi / pi.sum(), tensors

    from sofic.generators.matrices import emission_tensors

    pi, joint = emission_tensors(model, policy="stationary")
    if not joint:
        raise ValueError("model emits no symbols")
    try:
        pi_f = np.asarray(pi, dtype=float)
        tensors = np.stack([np.asarray(matrix, dtype=float) for matrix in joint.values()])
    except TypeError as exc:
        raise NotImplementedError("entropy-rate bounds and estimators need numeric probabilities") from exc
    return pi_f / pi_f.sum(), tensors


def _entropy_bits(probabilities: np.ndarray) -> float:
    positive = probabilities[probabilities > 0.0]
    return float(-np.sum(positive * np.log2(positive)))


def _row_entropies_bits(probabilities: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        terms = np.where(probabilities > 0.0, probabilities * np.log2(probabilities), 0.0)
    return -terms.sum(axis=1)


def collision_entropy(quasi_model: QuasiStochasticModel) -> float:
    """Second Renyi entropy rate (bits) from quasi transition matrices."""
    matrices = quasi_model.symbol_matrices()
    pi = quasi_model.stationary_quasidistribution()
    total = 0.0
    for matrix in matrices.values():
        total += float(pi @ (matrix @ matrix) @ np.ones(len(pi)))
    if total <= 0.0:
        return 0.0
    return float(-np.log2(total))


def process_negativity(quasi_model: QuasiStochasticModel) -> float:
    """Negativity proxy from the stationary quasidistribution."""
    pi = quasi_model.stationary_quasidistribution()
    positive = np.maximum(pi, 0.0)
    return float(np.sum(np.abs(pi - positive)))
