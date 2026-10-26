"""Finite-state channels as channels for :mod:`dit.coding` channel codes.

A stochastic :class:`~sofic.automata.transducers.MealyMachine` with
transitions ``T(y, s' | s, x)`` is a finite-state channel. Wrapped as a
:class:`FiniteStateChannel` it implements the channel protocol of
:func:`dit.coding.as_channel`, so a dit code can be decoded and evaluated over
it. ``MealyMachine.__dit_channel__`` returns this wrapper, so the machine can be
passed to dit directly.
"""

from __future__ import annotations

from collections.abc import Hashable, Mapping, Sequence
from typing import TYPE_CHECKING, Any

import numpy as np

from sofic.generators.capacity._channel import initial_law

if TYPE_CHECKING:
    from sofic.automata.transducers import MealyMachine

__all__ = ["FiniteStateChannel"]


def _sorted(values: Any) -> tuple[Any, ...]:
    try:
        return tuple(sorted(values))
    except TypeError:
        return tuple(sorted(values, key=repr))


def _default_map(symbols: Sequence[Any]) -> dict[Any, Any]:
    """Map digit strings to ``int`` (``"0" -> 0``), leaving other symbols alone."""
    if all(isinstance(s, str) and s.isdigit() for s in symbols):
        mapping = {s: int(s) for s in symbols}
        if len(set(mapping.values())) == len(mapping):
            return mapping
    return {s: s for s in symbols}


class FiniteStateChannel:
    """A finite-state channel ``T(y, s' | s, x)`` in the :mod:`dit.coding` channel protocol.

    Parameters
    ----------
    machine : MealyMachine
        A stochastic Mealy machine whose transitions read ``x`` and emit ``y``
        with probability ``T(y, s' | s, x)``. Unifilarity is not required,
        except by :meth:`capacity`.
    input_map, output_map : Mapping, optional
        Maps from the machine's symbols to the symbols dit sees. By default,
        digit strings become integers, so ``"0"`` and ``"1"`` become the bits
        ``0`` and ``1`` the binary codes use.

    Notes
    -----
    :meth:`log_likelihood` is the forward algorithm, exact for any
    finite-state channel. :meth:`symbol_log_likelihoods` runs the
    forward-backward (BCJR) recursions :cite:`Bahl1974` with uniform i.i.d.
    inputs at the other positions, giving ``log p(y^n | x_i)``; the per-symbol
    decoders in dit use these as their soft information.
    """

    def __init__(
        self,
        machine: MealyMachine,
        input_map: Mapping[Any, Any] | None = None,
        output_map: Mapping[Any, Any] | None = None,
    ) -> None:
        matrices, states = machine.labeled_transition_matrices()
        raw_inputs = _sorted({x for x, _ in matrices})
        raw_outputs = _sorted({y for _, y in matrices})
        input_map = dict(input_map) if input_map is not None else _default_map(raw_inputs)
        output_map = dict(output_map) if output_map is not None else _default_map(raw_outputs)

        self.machine = machine
        self.states: tuple[Hashable, ...] = tuple(states)
        self.inputs = _sorted({input_map[x] for x in raw_inputs})
        self.outputs = _sorted({output_map[y] for y in raw_outputs})
        self._in_index = {x: i for i, x in enumerate(self.inputs)}
        self._out_index = {y: j for j, y in enumerate(self.outputs)}

        n = len(self.states)
        self.T = np.zeros((len(self.inputs), len(self.outputs), n, n))
        for (x, y), matrix in matrices.items():
            self.T[self._in_index[input_map[x]], self._out_index[output_map[y]]] += matrix
        self.initial = initial_law(machine, self.states)

    def log_likelihood(self, codeword: Sequence[Any], received: Sequence[Any]) -> float:
        """Return ``log p(received | codeword)`` by the scaled forward algorithm."""
        alpha = self.initial
        total = 0.0
        for x, y in zip(codeword, received, strict=True):
            alpha = alpha @ self.T[self._in_index[x], self._out_index[y]]
            scale = alpha.sum()
            if scale <= 0.0:
                return -np.inf
            total += np.log(scale)
            alpha = alpha / scale
        return float(total)

    def sample(self, codeword: Sequence[Any], prng: np.random.Generator) -> tuple[Any, ...]:
        """Pass ``codeword`` through the channel once, starting from the initial law."""
        n = len(self.states)
        state = prng.choice(n, p=self.initial)
        received = []
        for x in codeword:
            weights = self.T[self._in_index[x], :, state, :].ravel()
            total = weights.sum()
            if total <= 0.0:
                raise ValueError(f"input {x!r} is not admissible in state {self.states[state]!r}")
            y, state = divmod(prng.choice(weights.size, p=weights / total), n)
            received.append(self.outputs[y])
        return tuple(received)

    def symbol_log_likelihoods(self, received: Sequence[Any]) -> np.ndarray:
        """Return the ``(n, len(inputs))`` array of ``log p(y^n | x_i = a)``, up to row constants.

        The inputs at the other positions are taken to be uniform and i.i.d.
        """
        cols = [self._out_index[y] for y in received]
        n = len(cols)
        M = [self.T[:, j].mean(axis=0) for j in cols]

        alphas = [self.initial]
        for t in range(n):
            alpha = alphas[-1] @ M[t]
            scale = alpha.sum()
            alphas.append(alpha / scale if scale > 0.0 else alpha)
        betas = [np.ones(len(self.states))]
        for t in reversed(range(n)):
            beta = M[t] @ betas[-1]
            scale = beta.sum()
            betas.append(beta / scale if scale > 0.0 else beta)
        betas.reverse()

        S = np.empty((n, len(self.inputs)))
        with np.errstate(divide="ignore"):
            for t, j in enumerate(cols):
                S[t] = np.log(np.einsum("s,ast,t->a", alphas[t], self.T[:, j], betas[t + 1]))
        return S

    def capacity(self, **kwargs: Any) -> tuple[float, float]:
        """Bracket the capacity without feedback, as ``(lower, upper)`` bits per use.

        Delegates to :func:`sofic.generators.capacity.capacity`, which needs a
        strongly connected unifilar channel; keyword arguments are passed on.
        """
        from sofic.generators.capacity import capacity

        bounds = capacity(self.machine, **kwargs)
        return (bounds.lower, bounds.upper)
