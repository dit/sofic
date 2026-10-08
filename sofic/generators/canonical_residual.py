"""Canonical residual HMM: the process analogue of the canonical RFSA.

The canonical residual finite-state automaton of a regular language has the
*prime* residuals as states, those that are not a union of other residuals
:cite:`Denis2002` (see also :cite:`MaarandTamm2022`). For a stationary process
with a finite ε-machine, the residual of a causal state ``s`` is its future
morph :math:`f_s(w) = P(w \\mid s)`. Replacing "union" by "nonnegative
combination", the prime morphs are the extreme rays of the cone spanned by the
morphs. Every morph satisfies :math:`f_s(\\lambda) = 1` on the empty word, so a
nonnegative combination of morphs is a convex one.

Writing each morph as :math:`f_t = \\sum_{r \\in R} c_{t r} f_r` over the extreme
set :math:`R`, the identity :math:`f_r(x w) = \\sum_t T^{(x)}_{r t} f_t(w)` gives
a generator on :math:`R` with joint transition matrices
:math:`T^{(x)} C` restricted to the rows in :math:`R`, and initial law
:math:`\\pi C`. By induction on word length state ``r`` of this generator emits
the futures :math:`f_r`, so it generates the same process with at most as many
states as the ε-machine.
"""

from __future__ import annotations

from collections.abc import Hashable
from typing import TYPE_CHECKING, Any

import numpy as np
from scipy.optimize import linprog

from sofic.generators.mealy import MealyHMM

if TYPE_CHECKING:
    from sofic.generators.base import HiddenMarkovModel

__all__ = ["canonical_residual_hmm"]


def _unifilar_support(
    model: HiddenMarkovModel, max_states: int
) -> tuple[list[Hashable], np.ndarray, dict[Any, np.ndarray]]:
    from sofic.generators.epsilon_machine import EpsilonMachine
    from sofic.generators.matrices import symbol_matrices
    from sofic.generators.renyi import _stationary_support

    mealy = model.to_mealy()
    if not mealy.is_unifilar():
        mealy = EpsilonMachine.from_hmm(mealy, max_states=max_states)
    states = list(mealy.reindex().states)
    pi = np.asarray(mealy.stationary_distribution(), dtype=float)
    raw = {symbol: np.asarray(m, dtype=float) for symbol, m in symbol_matrices(mealy).items()}
    keep = _stationary_support(pi, list(raw.values()))
    matrices = {symbol: m[np.ix_(keep, keep)] for symbol, m in raw.items()}
    return [s for s, k in zip(states, keep, strict=True) if k], pi[keep] / pi[keep].sum(), matrices


def _morph_coordinates(matrices: dict[Any, np.ndarray], size: int, tol: float) -> np.ndarray:
    """Return the state-by-test-word matrix ``F[s, j] = P(w_j | s)`` on a basis of test words.

    Test words are found breadth first from the empty word, keeping ``w`` when
    :math:`T^{(w)} \\mathbf{1}` is independent of those kept so far. Their span
    is the span of all :math:`T^{(w)} \\mathbf{1}`, so linear relations among
    rows of ``F`` are exactly linear relations among the morphs.
    """
    basis: list[np.ndarray] = []
    queue = [np.ones(size)]
    while queue and len(basis) < size:
        vector = queue.pop(0)
        candidate = np.column_stack([*basis, vector])
        if np.linalg.matrix_rank(candidate, tol=tol * max(1.0, np.abs(candidate).max())) > len(basis):
            basis.append(vector)
            queue.extend(m @ vector for m in matrices.values())
    return np.column_stack(basis)


def _conic_residual(target: np.ndarray, generators: np.ndarray) -> tuple[float, np.ndarray]:
    """Return ``min ||generators @ c - target||_1`` over ``c >= 0`` and a minimizer."""
    rows, cols = generators.shape
    if cols == 0:
        return float(np.abs(target).sum()), np.zeros(0)
    identity = np.eye(rows)
    result = linprog(
        np.concatenate([np.zeros(cols), np.ones(2 * rows)]),
        A_eq=np.hstack([generators, identity, -identity]),
        b_eq=target,
        bounds=(0, None),
        method="highs",
    )
    if not result.success:
        raise RuntimeError(f"linear program failed: {result.message}")
    return float(result.fun), result.x[:cols]


def canonical_residual_hmm(
    machine: HiddenMarkovModel,
    *,
    experimental: bool = False,
    tol: float = 1e-7,
    max_states: int = 10_000,
) -> MealyHMM:
    """Return the canonical residual HMM of a process with a finite ε-machine.

    .. warning::
       Experimental, and must be enabled with ``experimental=True``. The
       construction is a process analogue of the canonical RFSA
       :cite:`Denis2002,MaarandTamm2022` with no published reference; it is
       verified against brute-force word distributions in the tests only.

    The states are the causal states whose future morphs are extreme rays of
    the cone spanned by all morphs (see the module docstring). Morphs are
    compared through their probabilities on a basis of test words, and
    extremality and the combination coefficients are decided by
    :func:`scipy.optimize.linprog` with residual tolerance ``tol`` (floating
    point; no exact arithmetic). The default sits at the HiGHS feasibility
    tolerance; morphs closer than ``tol`` are treated as equal, so ε-machines
    whose states accumulate numerically (finite approximations of infinite
    ones) lose those states. Causal states with identical morphs are
    merged, so a unifilar but non-minimal presentation gives the same result
    as its ε-machine. The result is a possibly non-unifilar
    :class:`~sofic.generators.mealy.MealyHMM` on the extreme states, keeping
    their labels, that generates the same process with at most as many states
    as the ε-machine. When the morphs are affinely independent (for example
    when the ε-machine has two states) it is the ε-machine itself.

    Since it is a generator of the process, its state entropy is an upper
    bound on the minimal state entropy of any generating HMM, and its state
    count an upper bound on the minimal generator size studied by Löhr and Ay
    :cite:`LohrAy2009`, which can lie strictly below the ε-machine. No lower
    bound or minimality claim is made.

    Non-unifilar input is converted with
    :meth:`EpsilonMachine.from_hmm(machine, max_states=max_states)
    <sofic.generators.epsilon_machine.EpsilonMachine.from_hmm>`, which raises
    :class:`~sofic.exceptions.MixedStateExplosionError` when the ε-machine is
    not finite.

    Raises
    ------
    RuntimeError
        If ``experimental`` is not ``True``.
    """
    if experimental is not True:
        raise RuntimeError("canonical_residual_hmm is experimental; pass experimental=True to use it")
    states, pi, matrices = _unifilar_support(machine, max_states)
    size = len(states)
    morphs = _morph_coordinates(matrices, size, tol)
    scale = max(1.0, float(np.abs(morphs).max()))

    representatives: list[int] = []
    for i in range(size):
        if not any(np.abs(morphs[i] - morphs[j]).sum() <= tol * scale for j in representatives):
            representatives.append(i)
    extreme = [
        i
        for i in representatives
        if _conic_residual(morphs[i], morphs[[j for j in representatives if j != i]].T)[0] > tol * scale
    ]

    coefficients = np.zeros((size, len(extreme)))
    for t in range(size):
        if t in extreme:
            coefficients[t, extreme.index(t)] = 1.0
            continue
        _residual, weights = _conic_residual(morphs[t], morphs[extreme].T)
        coefficients[t] = weights / weights.sum()

    initial = pi @ coefficients
    hmm = MealyHMM(
        initial_distribution={states[r]: float(mass) for r, mass in zip(extreme, initial / initial.sum(), strict=True)},
        observation_alphabet=frozenset(matrices),
    )
    for r in extreme:
        hmm.graph.add_state(states[r])
    joint = {symbol: (m @ coefficients)[extreme] for symbol, m in matrices.items()}
    for a, r in enumerate(extreme):
        edges = [
            (symbol, b, block[a, b])
            for symbol, block in joint.items()
            for b in range(len(extreme))
            if block[a, b] > tol
        ]
        total = sum(prob for *_, prob in edges)
        for symbol, b, prob in edges:
            hmm.add_transition(states[r], states[extreme[b]], symbol, float(prob / total))
    return hmm
