"""Edge-tensor specification for generator search without a full bidirectional graph.

When the reverse ε-machine is infinite and truncation error must be tracked explicitly,
build a :class:`GeneratorProblemSpec` from ``(Pe, F, G, p_pair)`` instead of
:class:`~sofic.generators.coupled_generators._GeneratorProblem` from a
:class:`~sofic.generators.bidirectional_epsilon_machine.BidirectionalEpsilonMachine`.

**Rigorous lowers** on truncated presentations require bounding
``||Pe_approx - Pe_true||`` (or word-level reproduction error). **Upper bounds**
from a generator verified against the forward ε-machine remain valid when
``edge_reproduction_error`` is below tolerance.

Future work: ``_GeneratorProblem.from_edge_tensors(spec)`` sharing the same
``generator_kernel`` / ``coupled_search`` interface as the bidirectional-backed
problem.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class GeneratorProblemSpec:
    """Minimal data for coupled generator equations (see module docstring)."""

    Pe: np.ndarray
    """Shape ``(n_plus, n_sym, n_minus)``: edge law ``Pr(S+_0, X_0, S-_1)``."""
    F: np.ndarray
    """Forward causal map indicators, shape ``(n_plus, n_sym, n_plus)``."""
    G: np.ndarray
    """Reverse causal map indicators, shape ``(n_sym, n_minus, n_minus)``."""
    p_pair: np.ndarray
    """Pair marginal on lifted joint indices (same layout as ``_LiftedProblem.p``)."""
    p_plus: np.ndarray
    """Marginal on forward causal states."""
    p_minus: np.ndarray
    """Marginal on reverse causal states."""
    symbols: tuple[int, ...] | tuple[str, ...]
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


def spec_from_generator_problem(problem: object) -> GeneratorProblemSpec:
    """Snapshot tensors from an existing ``_GeneratorProblem``."""
    return GeneratorProblemSpec(
        Pe=np.asarray(problem.Pe, float),
        F=np.asarray(problem.F, float),
        G=np.asarray(problem.G, float),
        p_pair=np.asarray(problem.p_pair, float),
        p_plus=np.asarray(problem.p_plus, float),
        p_minus=np.asarray(problem.p_minus, float),
        symbols=tuple(problem.symbols),
    )


__all__ = ["GeneratorProblemSpec", "spec_from_generator_problem"]
