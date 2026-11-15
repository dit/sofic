r"""Finite-state rounded Bayesian filters for positive-transition Moore HMMs.

The log-odds grid and loss-dependent precision follow Feng & Meng,
:cite:`feng2026recursive`, Theorems 3.1 and 4.3 and Appendix A.4. Bounds are
analytic real-arithmetic bounds; this float64 implementation does not provide
an outward-rounded numerical certificate.
"""

from __future__ import annotations

import math
from collections.abc import Hashable, Iterable, Sequence
from dataclasses import dataclass, field
from numbers import Integral
from typing import Any, Literal

import numpy as np

from sofic.generators.matrices import StartSpec, start_vector
from sofic.generators.mealy import MealyHMM
from sofic.generators.moore import MooreHMM
from sofic.graph import ATTR_EMISSION_DIST
from sofic.properties import transition_matrix

__all__ = ["QuantizedFilter", "QuantizedFilterBounds", "quantized_filter"]

_LOG2 = math.log(2.0)
Loss = Literal["event", "word", "kl"]


def _positive_integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


@dataclass(frozen=True)
class QuantizedFilterBounds:
    r"""Theorem-derived error bounds for a rounded filter :cite:`feng2026recursive`.

    ``tolerance`` and all decoded losses are in **bits**. ``rounding_error``
    and ``hilbert_error`` are natural-log Hilbert distances; ``contraction``
    is Birkhoff's coefficient. ``max_rounding_error`` is the largest float64
    distance checked over the reachable transitions, not an interval bound.

    ``rigorous`` is always false: the theorems hold in real arithmetic, but
    grid rounding, normalization, and the transition checks use float64.
    This record is not a machine-verified certificate or an optimal-memory
    claim. Bounds hold along histories starting from the constructor's exact
    initial label, not after an arbitrary reset of the source alone.
    """

    loss: Loss
    tolerance: float
    horizon: int | None
    contraction: float
    rounding_error: float
    hilbert_error: float
    max_rounding_error: float
    rigorous: bool = field(default=False, init=False)

    @property
    def event_log_error_bound(self) -> float:
        """Absolute log2 error for any supplied positive-probability future event."""
        return self.hilbert_error / _LOG2

    def word_log_error_bound(self, horizon: int) -> float:
        """Uniform absolute log2 word error through ``horizon`` generated symbols."""
        return _positive_integer(horizon, "horizon") * self.event_log_error_bound

    def total_kl_bound(self, horizon: int) -> float:
        """Total forward KL in bits, uniformly over observed histories.

        Each conditional KL is at most both the log-ratio bound and the
        chi-square bound ``expm1(hilbert_error)**2`` (Theorem 4.3).
        """
        horizon = _positive_integer(horizon, "horizon")
        error = self.hilbert_error
        one_step = min(error, math.expm1(error) ** 2) if error < 1.0 else error
        return horizon * (one_step / _LOG2)

    @property
    def error_bound(self) -> float:
        """Bound for the loss and horizon requested from :func:`quantized_filter`."""
        if self.loss == "event":
            return self.event_log_error_bound
        assert self.horizon is not None
        if self.loss == "word":
            return self.word_log_error_bound(self.horizon)
        return self.total_kl_bound(self.horizon)


@dataclass(frozen=True)
class QuantizedFilter:
    r"""Reachable finite-state observer and its approximate beliefs.

    Construct with :func:`quantized_filter`, following :cite:`feng2026recursive`.
    Label 0 stores the exact initial belief. ``beliefs[z]`` is a distribution
    over ``basis_states``; ``successors[z, a]`` is the next label on
    ``symbols[a]`` or -1 for an impossible observation. Arrays are read-only.
    ``grid_states`` counts the full construction before reachable-state
    pruning, including the separate initial label.

    Event queries and private simulation use the original HMM from a retained
    belief. :meth:`to_generator` instead rounds again after each output, so its
    errors can accumulate. These are deliberately separate operations.
    """

    basis_states: tuple[Hashable, ...]
    symbols: tuple[Any, ...]
    beliefs: np.ndarray
    successors: np.ndarray
    output_probabilities: np.ndarray
    grid_states: int
    bounds: QuantizedFilterBounds
    _source: MooreHMM = field(repr=False, compare=False)

    @property
    def n_states(self) -> int:
        """Number of reachable labels, including the exact initial label."""
        return len(self.beliefs)

    @property
    def memory_bits(self) -> int:
        """Allocated label capacity ``ceil(log2(n_states))``, not state entropy."""
        return (self.n_states - 1).bit_length()

    def _state(self, state: int) -> int:
        if isinstance(state, bool) or not isinstance(state, Integral) or not 0 <= state < self.n_states:
            raise ValueError(f"state must be a label in [0, {self.n_states})")
        return int(state)

    def step(self, state: int, symbol: Any) -> int:
        """Update one retained label; reject unknown or impossible observations."""
        state = self._state(state)
        try:
            column = self.symbols.index(symbol)
        except ValueError as error:
            raise ValueError(f"unknown observation {symbol!r}") from error
        target = int(self.successors[state, column])
        if target < 0:
            raise ValueError(f"observation {symbol!r} is impossible from state {state}")
        return target

    def state_after(self, observations: Iterable[Any], *, state: int = 0) -> int:
        """Read observations from label 0, or continue a previous observation phase."""
        state = self._state(state)
        for symbol in observations:
            state = self.step(state, symbol)
        return state

    def event_probability(self, likelihood: Sequence[float], *, state: int = 0) -> float:
        r"""Decode ``beliefs[state] @ likelihood`` for a supplied future event.

        ``likelihood[i]`` must be the event's probability given hidden state
        ``basis_states[i]`` at the next emission. Theorem 3.1 of
        :cite:`feng2026recursive` bounds its relative/log error at every horizon;
        constructing the event-likelihood vector is the caller's responsibility.
        """
        state = self._state(state)
        vector = np.asarray(likelihood, dtype=float)
        if vector.shape != (len(self.basis_states),) or not np.all(np.isfinite(vector)):
            raise ValueError("likelihood must have one finite probability per hidden state")
        if np.any(vector < 0.0) or np.any(vector > 1.0):
            raise ValueError("likelihood values must lie in [0, 1]")
        return float(self.beliefs[state] @ vector)

    def to_generator(self) -> MealyHMM:
        r"""Return the rounded, fixed-start unifilar generator (Appendix A.4).

        This is an approximate generator, not an epsilon-machine or an exact
        mixed-state presentation :cite:`feng2026recursive`. Returned models
        can be sampled, queried, or serialized with the ordinary HMM APIs.
        """
        generator = MealyHMM(initial_distribution={0: 1.0}, observation_alphabet=frozenset(self.symbols))
        for state, row in enumerate(self.successors):
            generator.graph.add_state(state)
            for column, target in enumerate(row):
                if target >= 0:
                    generator.add_transition(
                        state, int(target), self.symbols[column], float(self.output_probabilities[state, column])
                    )
        return generator

    def private_generator(self, *, state: int = 0) -> MooreHMM:
        r"""Initialize a copy of the original hidden generator from a retained belief.

        Only initialization uses the approximation (Proposition 4.4 of
        :cite:`feng2026recursive`). Its event-law error is horizon-independent.
        The paper's ``K + m`` label accounting assumes observation and simulation
        phases reuse storage; keeping this observer live concurrently is not
        that memory model. Subsequent edits to the copy cannot alter the filter.
        """
        state = self._state(state)
        generator = self._source.copy()
        generator.initial_distribution = dict(zip(self.basis_states, self.beliefs[state].tolist(), strict=True))
        return generator


def _model_data(model: MooreHMM, start: StartSpec) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.validate()
    states = model.reindex().states
    if not states:
        raise ValueError("quantized_filter requires a nonempty MooreHMM")
    transition, _states = transition_matrix(model, states=states)
    transition = np.asarray(transition, dtype=float)
    if not np.all(np.isfinite(transition)) or np.any(transition <= 0.0):
        raise ValueError("quantized_filter requires strictly positive hidden transition probabilities")
    symbols = tuple(sorted(model.observation_alphabet, key=repr))
    emissions = np.array(
        [[model.graph.state_attrs(s).get(ATTR_EMISSION_DIST, {}).get(y, 0.0) for y in symbols] for s in states],
        dtype=float,
    )
    initial = np.asarray(start_vector(model, start), dtype=float)
    for name, array in (("transition", transition), ("emission", emissions), ("initial", initial[None, :])):
        if (
            not np.all(np.isfinite(array))
            or np.any(array < 0.0)
            or not np.allclose(array.sum(axis=1), 1.0, rtol=0.0, atol=1e-12)
        ):
            raise ValueError(f"{name} probabilities must be finite, nonnegative, and normalized")
    return transition, emissions, initial


def _precision(tolerance: float, loss: Loss, horizon: int | None) -> tuple[float, int | None]:
    if isinstance(tolerance, bool) or not math.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("tolerance must be finite and strictly positive (in bits)")
    if loss not in ("event", "word", "kl"):
        raise ValueError("loss must be 'event', 'word', or 'kl'")
    if loss == "event":
        if horizon is not None:
            raise ValueError("event loss is horizon-independent; omit horizon")
        return tolerance * _LOG2, None
    horizon = _positive_integer(horizon, "horizon")
    log_budget = math.log(tolerance) + math.log(_LOG2) - math.log(horizon)
    if loss == "word":
        return math.exp(log_budget), horizon
    # log(1 + sqrt(epsilon_nats / T)), stable for very large horizons.
    return float(np.logaddexp(0.0, log_budget / 2.0)), horizon


def quantized_filter(
    model: MooreHMM,
    tolerance: float,
    *,
    loss: Loss = "event",
    horizon: int | None = None,
    start: StartSpec = None,
    max_states: int = 10_000,
) -> QuantizedFilter:
    r"""Build a finite rounded filter for a known positive-transition Moore HMM.

    Implements the log-odds construction of :cite:`feng2026recursive`, Theorem
    3.1 (event queries), Appendix A.4 (uniform words), and Theorem 4.3 (total
    forward KL). No stationary initialization, positive emissions, or finite
    exact mixed-state presentation is required. Arbitrary Mealy HMMs are not
    accepted: positivity of their summed transition matrix is insufficient.

    Parameters
    ----------
    model
        Numeric state-emitting HMM with strictly positive hidden transitions.
        The departure-emission convention is that of ``MooreHMM.to_mealy``.
    tolerance
        Positive error budget in **bits** (log2 probability error or KL).
    loss
        ``'event'``: uniform future-event log error, independent of horizon.
        ``'word'``: uniform word log error for the rounded generator through
        ``horizon``. ``'kl'``: total forward KL for that generator through
        ``horizon``, uniformly over observed histories, not just on average.
    horizon
        Positive integer required for ``'word'`` and ``'kl'``; omitted for
        ``'event'``. A per-symbol budget ``a`` corresponds to total ``a*horizon``.
    start
        Initial law in the usual HMM start-vector convention. Defaults to the
        model's initial law (stationary only when the model has no explicit law).
        Label 0 stores it exactly, including boundary beliefs.
    max_states
        Safety cap on reachable labels. The full Cartesian grid is never
        allocated, but its size is exponential in hidden-belief dimension.

    Notes
    -----
    With ``L = log(p_max/p_min)`` and Birkhoff coefficient ``kappa``, round
    log-odds coordinates at spacing at most ``(1-kappa)*xi``. Here ``xi`` is
    ``epsilon``, ``epsilon/T``, or ``log(1+sqrt(epsilon/T))`` for the three
    losses, after converting the bit budget to nats. Hilbert contraction
    bounds repeated rounding, and mixture likelihood ratios control events.

    The returned bounds describe this construction in real arithmetic;
    ``bounds.rigorous`` is false because numerical operations are float64,
    not interval-certified. Unresolvable grids or nonpositive rounded beliefs
    are rejected rather than given a spurious bound. This is not an optimizer
    of allocated memory, statistical complexity, or the information bottleneck.

    Raises
    ------
    TypeError
        If ``model`` is not a Moore HMM.
    ValueError
        For invalid probabilities/options, a reachable-state cap, or precision
        too fine for float64.
    """
    if not isinstance(model, MooreHMM):
        raise TypeError("quantized_filter requires a MooreHMM, not a general edge-emitting HMM")
    precision, horizon = _precision(tolerance, loss, horizon)
    max_states = _positive_integer(max_states, "max_states")
    source = model.copy()
    transition, emissions, initial = _model_data(source, start)
    basis = source.reindex().states
    symbols = tuple(sorted(source.observation_alphabet, key=repr))
    constant = bool(np.all(transition == transition[0]))
    log_p = np.log(transition)
    diameter = max(float(np.ptp(log_p - row, axis=1).max()) for row in log_p)
    contraction = math.tanh(diameter / 4.0)
    gap = 1.0 - contraction
    width = float(log_p.max() - log_p.min())
    if constant:
        intervals = 0
        spacing = 0.0
        grid_states = 1 if np.array_equal(initial, transition[0]) else 2
    else:
        radius = gap * precision
        ratio = 2.0 * width / radius if radius > 0.0 else math.inf
        if not math.isfinite(ratio) or ratio > 2**48:
            raise ValueError("requested filter precision is too fine for float64")
        intervals = max(1, math.ceil(ratio))
        spacing = 2.0 * width / intervals
        grid_states = 1 + (intervals + 1) ** (len(basis) - 1)

    beliefs = [initial]
    labels: dict[tuple[int, ...], int] = {}
    if constant and np.array_equal(initial, transition[0]):
        labels[()] = 0
    successors: list[list[int]] = []
    probabilities: list[np.ndarray] = []
    max_rounding_error = 0.0
    cursor = 0
    while cursor < len(beliefs):
        belief = beliefs[cursor]
        output = belief @ emissions
        row = [-1] * len(symbols)
        for column, probability in enumerate(output):
            likelihood = emissions[:, column]
            possible = bool(np.any((belief > 0.0) & (likelihood > 0.0)))
            if not possible:
                continue
            if probability <= 0.0:
                raise ValueError("observation probability underflows in float64")
            # Scaling the emission law avoids underflow for globally rare symbols.
            posterior = belief * (likelihood / likelihood.max())
            posterior /= posterior.sum()
            updated = posterior @ transition
            updated /= updated.sum()
            if constant:
                key = ()
                rounded = transition[0]
            else:
                coordinates = np.log(updated[:-1]) - math.log(updated[-1])
                position = (coordinates + width) / (2.0 * width) * intervals
                key = tuple(int(i) for i in np.clip(np.floor(position + 0.5), 0, intervals))
                coordinates = np.r_[-width + (2.0 * width / intervals) * np.asarray(key), 0.0]
                rounded = np.exp(coordinates - coordinates.max())
                rounded /= rounded.sum()
                if np.any(rounded <= 0.0):
                    raise ValueError("rounded belief underflows in float64")
                error = float(np.ptp(np.log(rounded) - np.log(updated)))
                max_rounding_error = max(max_rounding_error, error)
                if error > spacing:
                    raise ValueError("grid rounding cannot resolve the requested precision in float64")
            target = labels.get(key)
            if target is None:
                if len(beliefs) >= max_states:
                    raise ValueError(f"quantized filter exceeds max_states={max_states}; increase the cap or tolerance")
                target = labels[key] = len(beliefs)
                beliefs.append(rounded)
            row[column] = target
        successors.append(row)
        probabilities.append(output)
        cursor += 1

    arrays = [np.array(beliefs), np.array(successors, dtype=int), np.array(probabilities)]
    for array in arrays:
        array.setflags(write=False)
    bounds = QuantizedFilterBounds(
        loss, float(tolerance), horizon, contraction, spacing, spacing / gap, max_rounding_error
    )
    return QuantizedFilter(basis, symbols, arrays[0], arrays[1], arrays[2], grid_states, bounds, source)
