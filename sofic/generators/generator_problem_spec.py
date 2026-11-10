r"""Edge-tensor specification for generator search without a full bidirectional graph.

A :class:`GeneratorProblemSpec` holds the data the generator equations need: the edge law
``Pe = Pr(S+_0, X_0, S-_1)``, the forward and reverse maps ``F`` and ``G``, and the causal-state pairs
with their masses. :meth:`~sofic.generators.coupled_generators._GeneratorProblem.from_spec` turns it
into a problem for the kernel LP and the generator caps of :mod:`~sofic.generators.generator_caps`.

**Future windows** (:func:`window_spec`). When the reverse ε-machine is infinite, replace the reverse
causal state by the window :math:`Y_t = X_{t:t+L}` of the next ``L`` symbols. The generator-cap
relaxation uses three properties of :math:`S^-`, and all hold for :math:`Y`: a generator state
:math:`W_t` separates :math:`S^+_t` from :math:`Y_t` (it separates past from future);
:math:`Y_0 = g(X_0, Y_1)` with :math:`g` keeping :math:`X_0` and dropping the last symbol of
:math:`Y_1`; and :math:`Y_1` is independent of :math:`(W_0, X_0)` given :math:`W_1`. So every generator
satisfies the window equations, and caps and coverage bounds computed from a window spec are lower
bounds on :math:`C_g` (Löhr & Ay :cite:`LohrAy2009`) that need only the forward ε-machine. The tensors
are exact products of the forward symbol matrices. A solution of the window equations need not
generate the process, so upper bounds still need word-level verification.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class GeneratorProblemSpec:
    """Minimal data for coupled generator equations (see module docstring)."""

    Pe: np.ndarray
    """Shape ``(n_plus, n_sym, n_minus)``: edge law ``Pr(S+_0, X_0, S-_1)``."""
    F: np.ndarray
    """Forward causal map indicators, shape ``(n_plus, n_sym, n_plus)``."""
    G: np.ndarray
    """Reverse causal map indicators, shape ``(n_sym, n_minus, n_minus)``: ``G[x, b', b] = 1`` when
    ``S-_0 = b`` precedes ``S-_1 = b'`` on ``x``."""
    p_pair: np.ndarray
    """Pair marginal on lifted joint indices (same layout as ``_LiftedProblem.p``)."""
    p_plus: np.ndarray
    """Marginal on forward causal states."""
    p_minus: np.ndarray
    """Marginal on reverse causal states."""
    symbols: tuple[Any, ...]
    pairs: tuple[tuple[int, int], ...] | None = None
    """``(plus index, minus index)`` of each entry of ``p_pair``."""
    minus_labels: tuple[Any, ...] | None = None
    """Names of the reverse states (future windows for :func:`window_spec`)."""
    pe_drift: float | None = None
    """Max entrywise change in ``Pe`` between reverse-MSP caps (survey ladder)."""
    edge_reproduction_error: float | None = None
    """Word-level error vs forward reference after truncation (optional)."""

    @property
    def n_plus(self) -> int:
        return int(self.Pe.shape[0])

    @property
    def n_minus(self) -> int:
        return int(self.Pe.shape[2])

    @property
    def n_sym(self) -> int:
        return int(self.Pe.shape[1])


def spec_from_generator_problem(problem: Any) -> GeneratorProblemSpec:
    """Snapshot tensors from an existing ``_GeneratorProblem``."""
    pairs = tuple(
        (int(a), int(b))
        for a, b in zip(np.argmax(problem.lifted.plus, axis=1), np.argmax(problem.lifted.minus, axis=1), strict=True)
    )
    return GeneratorProblemSpec(
        Pe=np.asarray(problem.Pe, float),
        F=np.asarray(problem.F, float),
        G=np.asarray(problem.G, float),
        p_pair=np.asarray(problem.p_pair, float),
        p_plus=np.asarray(problem.p_plus, float),
        p_minus=np.asarray(problem.p_minus, float),
        symbols=tuple(problem.symbols),
        pairs=pairs,
    )


def window_spec(forward: Any, length: int, *, atol: float = 1e-15) -> GeneratorProblemSpec:
    """Generator equations with the reverse causal state replaced by the next ``length`` symbols.

    ``forward`` is a (unifilar) forward ε-machine. See the module docstring for why the window
    equations relax the generator equations.
    """
    from sofic.generators.matrices import symbol_matrices

    if length < 1:
        raise ValueError("length must be at least 1")
    mats = {x: np.asarray(T, dtype=float) for x, T in symbol_matrices(forward).items()}
    symbols = tuple(mats)
    n = next(iter(mats.values())).shape[0]
    total = np.sum(list(mats.values()), axis=0)
    vals, vecs = np.linalg.eig(total.T)
    pi = np.real(vecs[:, int(np.argmin(np.abs(vals - 1)))])
    pi = pi / pi.sum()

    def tail(word: tuple[Any, ...]) -> np.ndarray:
        v = np.ones(n)
        for x in reversed(word):
            v = mats[x] @ v
        return v

    words = [w for w in itertools.product(symbols, repeat=length) if pi @ tail(w) > atol]
    index = {w: j for j, w in enumerate(words)}
    P = np.array([pi * tail(w) for w in words]).T
    Pe = np.zeros((n, len(symbols), len(words)))
    G = np.zeros((len(symbols), len(words), len(words)))
    for i, x in enumerate(symbols):
        for j, w in enumerate(words):
            Pe[:, i, j] = pi * (mats[x] @ tail(w))
            if Pe[:, i, j].sum() > atol:
                G[i, j, index[(x, *w[:-1])]] = 1.0
    F = np.stack([(mats[x] > 0).astype(float) for x in symbols], axis=1)
    cells = [(a, j) for a in range(n) for j in range(len(words)) if P[a, j] > atol]
    p_pair = np.array([P[a, j] for a, j in cells])
    return GeneratorProblemSpec(
        Pe=Pe / Pe.sum(),
        F=F,
        G=G,
        p_pair=p_pair / p_pair.sum(),
        p_plus=pi,
        p_minus=P.sum(axis=0),
        symbols=symbols,
        pairs=tuple(cells),
        minus_labels=tuple(words),
    )


__all__ = ["GeneratorProblemSpec", "spec_from_generator_problem", "window_spec"]
