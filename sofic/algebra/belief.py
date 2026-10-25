"""Semigroups of belief (mixed-state) maps of hidden Markov models.

An observer of an HMM with symbol matrices ``T^(x)`` updates a belief
``eta`` over hidden states by ``eta -> eta T^(x) / (eta T^(x) 1)``. The
matrices ``T^(w)`` of words, taken up to a positive scalar, form a semigroup
that contains every such update. If that semigroup is finite, every initial
belief has a finite orbit, so the mixed-state presentation is finite from any
start. The converse fails: distinct matrices can induce the same belief map
(two rank-one matrices with the same row space both give one constant map),
and the orbit of one particular belief can be finite when the semigroup is
not. Finiteness of matrix semigroups is decidable :cite:`MandelSimon1977`
:cite:`Jacob1977`; this module only enumerates with a cap, so it proves
finiteness when the enumeration closes and is otherwise inconclusive.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Hashable
from typing import Any

import numpy as np

from sofic.algebra.transformation import DEFAULT_MAX_SIZE, TransformationSemigroup
from sofic.exceptions import SemigroupTooLargeError

__all__ = ["mixed_state_semigroup", "projective_semigroup"]


def projective_semigroup(
    hmm: Any,
    *,
    max_size: int = 10_000,
    decimals: int = 10,
) -> TransformationSemigroup:
    """Return the monoid of word matrices ``T^(w)`` up to scaling, as a transformation monoid.

    Each element is a word matrix normalized to total mass one (or the zero
    matrix, for words no state can emit), rounded to ``decimals`` places to
    compare. The monoid acts on itself by right multiplication by each
    symbol, which is faithful, so the result is a
    :class:`~sofic.algebra.transformation.TransformationSemigroup` whose
    states are shortest representative words (``()`` is the identity) and
    whose generators are the symbols. All of :mod:`sofic.algebra` then
    applies.

    Raises :class:`~sofic.exceptions.SemigroupTooLargeError` when more than
    ``max_size`` distinct elements appear: the semigroup may be infinite.
    """
    from sofic.generators.matrices import symbol_matrices

    matrices = {symbol: np.asarray(m, dtype=float) for symbol, m in symbol_matrices(hmm).items()}
    symbols = list(matrices)
    n = next(iter(matrices.values())).shape[0]

    def normalize(matrix: np.ndarray) -> np.ndarray:
        total = matrix.sum()
        return matrix / total if total > 0 else np.zeros_like(matrix)

    def key(matrix: np.ndarray) -> tuple[float, ...]:
        return tuple(np.round(matrix, decimals).ravel() + 0.0)

    identity = np.eye(n) / n
    elements = [identity]
    words: list[tuple[Hashable, ...]] = [()]
    index = {key(identity): 0}
    right: list[list[int]] = []
    queue = deque([0])
    while queue:
        i = queue.popleft()
        row = []
        for symbol in symbols:
            product = normalize(elements[i] @ matrices[symbol])
            k = key(product)
            if k not in index:
                if len(elements) >= max_size:
                    raise SemigroupTooLargeError(
                        f"more than {max_size} belief-map matrices; the semigroup may be infinite"
                    )
                index[k] = len(elements)
                elements.append(product)
                words.append((*words[i], symbol))
                queue.append(index[k])
            row.append(index[k])
        right.append(row)
    generators = {symbol: tuple(right[i][s] for i in range(len(elements))) for s, symbol in enumerate(symbols)}
    return TransformationSemigroup(generators, states=words, max_size=max(DEFAULT_MAX_SIZE, len(elements)))


def mixed_state_semigroup(hmm: Any, **kwargs: Any) -> TransformationSemigroup:
    """Return the transition semigroup of ``hmm``'s mixed-state presentation.

    The mixed-state presentation is unifilar, so its symbols act as partial
    maps of the reachable beliefs; keyword arguments go to
    :meth:`~sofic.generators.mealy.MealyHMM.mixed_state_presentation`.
    """
    from sofic.algebra.adapters import transformation_semigroup

    return transformation_semigroup(hmm.mixed_state_presentation(**kwargs))
