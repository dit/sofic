"""Array form of a finite unifilar channel ``T(y, s' | s, x)`` with ``s' = f(s, x, y)``."""

from __future__ import annotations

from collections.abc import Hashable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

from sofic.exceptions import StochasticValidationError, UnifilarityError
from sofic.graph import ATTR_OUTPUT, ATTR_PROB, ATTR_SYMBOL

if TYPE_CHECKING:
    from sofic.automata.transducers import MealyMachine


@dataclass(frozen=True, eq=False)
class ChannelTensors:
    """Dense arrays for a unifilar finite-state channel.

    ``P[s, x, y]`` is the output law, ``F[s, x, y]`` the successor state index
    (``-1`` where ``P`` is zero), and ``allowed[s, x]`` marks inputs defined at
    ``s``; undefined inputs act as input constraints. ``initial`` is the
    initial state law.
    """

    states: tuple[Hashable, ...]
    inputs: tuple[Any, ...]
    outputs: tuple[Any, ...]
    P: np.ndarray
    F: np.ndarray
    allowed: np.ndarray
    initial: np.ndarray

    @property
    def n_states(self) -> int:
        return len(self.states)

    @property
    def n_inputs(self) -> int:
        return len(self.inputs)

    @property
    def n_outputs(self) -> int:
        return len(self.outputs)

    def trivial_upper_bound(self) -> float:
        """Return ``log2 min(|X|, |Y|)``, which bounds every capacity."""
        return float(np.log2(min(self.n_inputs, self.n_outputs)))

    def conditional_entropy(self) -> np.ndarray:
        """Return ``H[Y | X = x, S = s]`` in bits as an ``(S, X)`` array."""
        P = self.P
        with np.errstate(divide="ignore", invalid="ignore"):
            terms = np.where(P > 0, -P * np.log2(np.where(P > 0, P, 1.0)), 0.0)
        return terms.sum(axis=2)


def _sorted(values: set[Any]) -> tuple[Any, ...]:
    return tuple(sorted(values, key=repr))


def channel_tensors(channel: MealyMachine) -> ChannelTensors:
    """Return the :class:`ChannelTensors` of a unifilar stochastic Mealy machine."""
    transitions = list(channel.transitions())
    states = _sorted(set(channel.states()))
    inputs = _sorted({t.data[ATTR_SYMBOL] for t in transitions})
    outputs = _sorted({t.data[ATTR_OUTPUT] for t in transitions})
    s_index = {state: i for i, state in enumerate(states)}
    x_index = {symbol: i for i, symbol in enumerate(inputs)}
    y_index = {symbol: i for i, symbol in enumerate(outputs)}

    shape = (len(states), len(inputs), len(outputs))
    P = np.zeros(shape)
    F = np.full(shape, -1, dtype=int)
    for transition in transitions:
        s = s_index[transition.source]
        x = x_index[transition.data[ATTR_SYMBOL]]
        y = y_index[transition.data[ATTR_OUTPUT]]
        prob = float(transition.data.get(ATTR_PROB, 1.0))
        if prob <= 0.0:
            continue
        target = s_index[transition.target]
        if F[s, x, y] not in (-1, target):
            raise UnifilarityError(
                f"(state, input, output) = {(transition.source, inputs[x], outputs[y])} is ambiguous"
            )
        P[s, x, y] += prob
        F[s, x, y] = target

    totals = P.sum(axis=2)
    allowed = totals > 0.0
    if not np.allclose(totals[allowed], 1.0):
        raise StochasticValidationError("channel rows T(., . | s, x) must sum to one")
    if not allowed.any(axis=1).all():
        raise StochasticValidationError("every state needs at least one admissible input")

    return ChannelTensors(states, inputs, outputs, P, F, allowed, initial_law(channel, states))


def initial_law(channel: MealyMachine, states: tuple[Hashable, ...] | list[Hashable]) -> np.ndarray:
    """Return the initial state law of ``channel`` over ``states``.

    An explicit ``initial_distribution`` wins; otherwise the law is uniform on
    ``initial_states``, or on every state when there are none.
    """
    distribution = getattr(channel, "initial_distribution", None) or {}
    if distribution:
        initial = np.array([float(distribution.get(state, 0.0)) for state in states])
    else:
        initial = np.array([1.0 if state in channel.initial_states else 0.0 for state in states])
        if initial.sum() == 0.0:
            initial = np.ones(len(states))
    return initial / initial.sum()


def require_strongly_connected(tensors: ChannelTensors) -> None:
    """Raise unless some input policy can drive every state to every other state.

    The feedback-capacity theorems used here assume a strongly connected
    unifilar channel :cite:`Permuter2008,Sabag2017`, which makes the capacity
    independent of the initial state.
    """
    n = tensors.n_states
    reachable = np.zeros((n, n), dtype=bool)
    sources, _xs, _ys = np.nonzero(tensors.P > 0)
    reachable[sources, tensors.F[tensors.P > 0]] = True
    n_components, _labels = connected_components(csr_matrix(reachable), directed=True, connection="strong")
    if n_components != 1:
        raise StochasticValidationError(
            f"capacity bounds need a strongly connected channel; this one has {n_components} strong components"
        )
