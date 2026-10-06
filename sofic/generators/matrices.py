"""Symbol-labeled transition matrices and start vectors for stochastic generators.

For a Mealy-style hidden Markov model with states indexed by ``model.reindex()``,
the symbol-labeled joint transition matrices are

.. math:: T^{(x)}_{ij} = P(S_{t+1} = j, X_t = x \\mid S_t = i),

so the probability of a word ``w = x_0 ... x_{L-1}`` from a start vector
:math:`\\eta` is :math:`\\eta T^{(x_0)} \\cdots T^{(x_{L-1})} \\mathbf{1}`
(:cite:`Rabiner1989,Ellison2009`). Their sum
:math:`T = \\sum_x T^{(x)}` is the internal state-to-state transition matrix.

Start policies
--------------
Every helper that needs an initial state law takes ``policy``:

``"model"``
    The model's ``initial_distribution``, or its ``stationary_distribution()``
    when no initial distribution is given. This is the law of the generator *as
    specified* and is used for likelihoods, decoding, sampling, and finite-word
    probabilities.

``"stationary"``
    The stationary law of the process. Block and window statistics of a
    stationary process must weight the first state by this law rather than by a
    possibly transient ``initial_distribution``. On reducible chains the
    eigenvector stationary law is not unique, so the limit of the ``"model"``
    start vector under :math:`T` is preferred; the eigenvector solution is used
    when that limit cannot be formed (or the model is symbolic), and the
    ``"model"`` start vector as a last resort.

An explicit ``start`` (a state, a state-to-mass mapping, or a dense vector in
``reindex()`` order) overrides either policy.
"""

from __future__ import annotations

from collections.abc import Callable, Hashable, Iterable, Mapping, Sequence
from typing import Any, Literal

import numpy as np

from sofic.generators.prob import as_prob, has_symbolic, zeros
from sofic.graph import ATTR_EMISSION, ATTR_PROB

StartPolicy = Literal["model", "stationary"]
StartSpec = Hashable | Mapping[Hashable, Any] | Sequence[Any] | np.ndarray | None

SKIP = object()


def accumulate_matrices(
    model: Any,
    *,
    attr: str = ATTR_PROB,
    states: Iterable[Hashable] | None = None,
    label: Callable[[Any], Any] = lambda transition: None,
    labels: Iterable[Any] = (),
    symbolic: bool | None = None,
) -> tuple[dict[Any, np.ndarray], list[Hashable]]:
    """Accumulate edge ``attr`` weights into one state-to-state matrix per edge label.

    Rows and columns follow ``states`` when given (edges leaving the set are
    ignored), otherwise all model states in iteration order. ``label(transition)``
    keys the matrix an edge contributes to; returning ``SKIP`` drops the edge.
    Matrices for ``labels`` are allocated first, in the given order, so they exist
    even without edges; other labels follow in order of first appearance.
    ``symbolic`` selects object (sympy) storage and is inferred from the retained
    edge weights when omitted. Returns the matrices and the ordered state list.
    """
    ordered = list(states) if states is not None else list(model.states())
    index = {state: i for i, state in enumerate(ordered)}
    n = len(ordered)
    edges: list[tuple[int, int, Any, Any]] = []
    for state in ordered:
        for transition in model.graph.out_transitions(state):
            j = index.get(transition.target)
            key = label(transition)
            if j is None or key is SKIP:
                continue
            edges.append((index[state], j, key, transition.data.get(attr, 0.0)))
    if symbolic is None:
        symbolic = has_symbolic(value for *_, value in edges)

    matrices: dict[Any, np.ndarray] = {key: zeros((n, n), symbolic=symbolic) for key in labels}
    for i, j, key, value in edges:
        matrix = matrices.get(key)
        if matrix is None:
            matrix = matrices[key] = zeros((n, n), symbolic=symbolic)
        if symbolic:
            matrix[i, j] = as_prob(matrix[i, j]) + as_prob(value)
        else:
            matrix[i, j] += float(value)
    return matrices, ordered


def emission_label(transition: Any) -> Any:
    """:func:`accumulate_matrices` label keying an edge by its emitted symbol (unlabeled edges skipped)."""
    emission = transition.data.get(ATTR_EMISSION)
    return SKIP if emission is None else emission


def _as_mealy(model: Any) -> Any:
    to_mealy = getattr(model, "to_mealy", None)
    return to_mealy() if to_mealy is not None else model


def _is_symbolic(mealy: Any) -> bool:
    edge_probs = [transition.data.get(ATTR_PROB, 0.0) for transition in mealy.transitions()]
    return has_symbolic(edge_probs) or has_symbolic(mealy.initial_distribution.values())


def symbol_matrices(mealy: Any) -> dict[Any, np.ndarray]:
    """Return symbol -> joint transition matrix ``T^(x)`` of a Mealy-style model.

    Symbols (the observation alphabet plus every emitted symbol) are keyed in
    ``repr``-sorted order, so iteration and seeded sampling are reproducible.
    Matrices are object-dtype when any edge or initial probability is symbolic.
    """
    emissions = {transition.data.get(ATTR_EMISSION) for transition in mealy.transitions()} - {None}
    symbols = sorted(set(getattr(mealy, "observation_alphabet", ())) | emissions, key=repr)
    matrices, _states = accumulate_matrices(
        mealy,
        attr=ATTR_PROB,
        states=mealy.reindex().states,
        label=emission_label,
        labels=symbols,
        symbolic=_is_symbolic(mealy),
    )
    return matrices


def limit_distribution(pi_initial: np.ndarray, transition: np.ndarray) -> np.ndarray | None:
    """Return the limiting occupation law of ``pi_initial`` under ``transition``.

    Power-iterates the normalized law until it is invariant. Returns ``None`` when
    the mass vanishes; on periodic chains the result after the iteration cap need
    not be invariant, so callers should check it.
    """
    pi = np.asarray(pi_initial, dtype=float).copy()
    total = float(pi.sum())
    if total <= 0.0:
        return None
    pi /= total
    matrix = np.asarray(transition, dtype=float)
    n = len(pi)
    for _ in range(max(100, 20 * n)):
        nxt = pi @ matrix
        mass = float(nxt.sum())
        if mass <= 0.0:
            return None
        nxt /= mass
        if np.allclose(nxt, pi, rtol=1e-12, atol=1e-14):
            pi = nxt
            break
        pi = nxt
    pi[np.isclose(pi, 0.0, atol=1e-15)] = 0.0
    mass = float(pi.sum())
    if mass <= 0.0:
        return None
    return pi / mass


def _model_start(mealy: Any) -> np.ndarray:
    idx = mealy.reindex()
    n = len(idx)
    symbolic = _is_symbolic(mealy)
    if mealy.initial_distribution:
        pi = zeros((n,), symbolic=symbolic)
        for state, mass in mealy.initial_distribution.items():
            pi[idx.index(state)] = as_prob(mass)
        return pi
    if n:
        return np.asarray(mealy.stationary_distribution(), dtype=object if symbolic else float)
    return zeros((0,), symbolic=symbolic)


def _stationary_start(mealy: Any, joint: Mapping[Any, np.ndarray]) -> np.ndarray:
    from sofic.generators.stationary import stationary_distribution_from_transition

    pi_initial = _model_start(mealy)
    n = len(pi_initial)
    if n == 0:
        return pi_initial
    symbolic = pi_initial.dtype == object or any(matrix.dtype == object for matrix in joint.values())
    transition = zeros((n, n), symbolic=symbolic)
    for matrix in joint.values():
        transition = transition + matrix
    if not symbolic:
        limited = limit_distribution(pi_initial, transition)
        if limited is not None and np.allclose(limited @ transition, limited, rtol=1e-8, atol=1e-10):
            return limited
    try:
        return stationary_distribution_from_transition(transition)
    except (np.linalg.LinAlgError, ValueError):
        return pi_initial


def _explicit_start(mealy: Any, start: Hashable | Mapping[Hashable, Any] | Sequence[Any] | np.ndarray) -> np.ndarray:
    idx = mealy.reindex()
    n = len(idx)
    if isinstance(start, Mapping):
        vector = zeros((n,), symbolic=has_symbolic(start.values()))
        for state, mass in start.items():
            if not mealy.graph.has_state(state):
                raise ValueError(f"unknown start state {state!r}")
            vector[idx.index(state)] = as_prob(mass)
        return vector

    if isinstance(start, Hashable) and mealy.graph.has_state(start):
        vector = zeros((n,))
        vector[idx.index(start)] = 1.0
        return vector

    values = list(np.asarray(start, dtype=object).ravel())
    vector = np.asarray(values, dtype=object if has_symbolic(values) else float)
    if vector.shape != (n,):
        raise ValueError(f"start vector has length {vector.size}, expected {n}")
    return vector


def start_vector(model: Any, start: StartSpec = None, *, policy: StartPolicy = "model") -> np.ndarray:
    """Return the initial state vector of ``model`` in ``model.to_mealy().reindex()`` order.

    ``start`` may be ``None`` (use ``policy``; see the module docstring), a state
    (point mass), a state -> mass mapping, or a dense vector in state order.
    Explicit starts are not renormalized. The vector is object-dtype when any
    supplied or model probability is symbolic.
    """
    mealy = _as_mealy(model)
    if start is not None:
        return _explicit_start(mealy, start)
    if policy == "model":
        return _model_start(mealy)
    if policy == "stationary":
        return _stationary_start(mealy, symbol_matrices(mealy))
    raise ValueError(f"unknown start policy {policy!r}; expected 'model' or 'stationary'")


def emission_tensors(model: Any, *, policy: StartPolicy = "model") -> tuple[np.ndarray, dict[Any, np.ndarray]]:
    """Return ``(pi, T)``: the ``policy`` start vector and symbol -> ``T^(x)`` matrices.

    ``model`` is converted with ``to_mealy()`` when it provides one, and both
    outputs share that presentation's ``reindex()`` state order.
    """
    mealy = _as_mealy(model)
    joint = symbol_matrices(mealy)
    if policy == "model":
        return _model_start(mealy), joint
    if policy == "stationary":
        return _stationary_start(mealy, joint), joint
    raise ValueError(f"unknown start policy {policy!r}; expected 'model' or 'stationary'")
