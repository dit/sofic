"""Support languages of stationary processes and the decisions built on them.

The *stationary support* of an HMM is the set of finite words of positive
probability under its stationary law. It is a factorial regular language, so
comparisons between supports reduce to inclusion of finite automata, decided
here by antichain inclusion :cite:`DeWulf2006`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from sofic.graph import ATTR_SYMBOL

if TYPE_CHECKING:
    from sofic.automata.nfa import NFA
    from sofic.generators.base import HiddenMarkovModel

_TOL = 1e-12


def support_nfa(model: HiddenMarkovModel) -> NFA:
    """Return an NFA accepting exactly the words of positive stationary probability.

    States are the generator states of positive stationary probability, all of
    them initial and accepting; edges are the positive-probability transitions.
    The stationary support is closed under such transitions, so a word is
    accepted iff some stationary-positive state can emit it.

    This differs from :meth:`~sofic.generators.base.HiddenMarkovModel.to_support_nfa`,
    which starts from *every* state, transient ones included, and so accepts
    words that only a transient state can emit (words of stationary
    probability zero).
    """
    from sofic.automata.nfa import NFA
    from sofic.generators.matrices import symbol_matrices

    mealy = model.to_mealy()
    states = tuple(mealy.reindex().states)
    pi = np.asarray(mealy.stationary_distribution(), dtype=float)
    matrices = {symbol: np.asarray(matrix, dtype=float) for symbol, matrix in symbol_matrices(mealy).items()}
    live = [i for i, mass in enumerate(pi) if mass > _TOL]
    live_states = frozenset(states[i] for i in live)
    nfa = NFA(
        input_alphabet=frozenset(matrices),
        initial_states=live_states,
        accepting_states=live_states,
    )
    for state in live_states:
        nfa.graph.add_state(state)
    for symbol, matrix in matrices.items():
        for i in live:
            for j in np.flatnonzero(matrix[i] > _TOL):
                nfa.graph.add_transition(states[i], states[j], **{ATTR_SYMBOL: symbol})
    return nfa


def support_includes(p: HiddenMarkovModel, q: HiddenMarkovModel) -> bool:
    """Return whether every word with positive ``P`` probability has positive ``Q`` probability.

    Both processes are taken in their stationary laws. Decided by antichain
    inclusion of the :func:`support_nfa` automata :cite:`DeWulf2006`, without
    determinizing.
    """
    return support_nfa(q).includes(support_nfa(p))


def is_absolutely_continuous(p: HiddenMarkovModel, q: HiddenMarkovModel) -> bool:
    """Return whether ``P_{0:n} << Q_{0:n}`` for every block length ``n``.

    This is absolute continuity on the finite-dimensional cylinders, the
    condition under which every block divergence ``D(P_{0:n} || Q_{0:n})`` and
    the relative entropy rate can be finite; it is the same decision as
    :func:`support_includes`. It is *not* absolute continuity of the laws on
    infinite sequences: two distinct ergodic stationary measures are mutually
    singular there, even when their finite-word supports coincide.
    """
    return support_includes(p, q)


def support_equal(p: HiddenMarkovModel, q: HiddenMarkovModel) -> bool:
    """Return whether ``P`` and ``Q`` give positive probability to the same finite words.

    Equivalently, ``P_{0:n}`` and ``Q_{0:n}`` are mutually absolutely continuous
    for every ``n``: inclusion of the :func:`support_nfa` languages both ways
    :cite:`DeWulf2006`.
    """
    left, right = support_nfa(p), support_nfa(q)
    return right.includes(left) and left.includes(right)
