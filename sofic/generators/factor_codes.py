"""Sliding block codes applied to stochastic processes.

A sliding block code ``Phi`` with memory ``m``, anticipation ``a`` and window
``w = m + 1 + a`` maps a process ``X`` to the *image process* ``Y = Phi(X)``,

.. math::

   Y_t = \\Phi(X_{t:t+w}),

i.e. ``(Phi X)_{t+m}`` in the two-sided indexing of :cite:`LindMarcus1995`
(§1.5). For a stationary ``X`` the index shift by ``m`` is immaterial, and the
law of ``Y_{0:n}`` is the pushforward of the law of ``X_{0:n+w-1}`` under the
word map ``x_{0:n+w-1} -> Phi(x_{0:n+w-1})``. A stationary coding cannot
increase the entropy rate, so a conjugacy (an invertible code) preserves it
:cite:`Gray1990` (the topological analogue is :cite:`LindMarcus1995`, §4.1);
neither the excess entropy nor the statistical complexity is a conjugacy
invariant (see :func:`higher_block`).
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Hashable
from typing import TYPE_CHECKING, Any

from sofic.automata.transducer_operations import transduce_generator
from sofic.generators.base import HiddenMarkovModel
from sofic.generators.mealy import MealyHMM
from sofic.graph import ATTR_EMISSION, ATTR_PROB

if TYPE_CHECKING:
    from sofic.shifts.sliding_block_code import SlidingBlockCode


def image_process(hmm: HiddenMarkovModel, code: SlidingBlockCode) -> MealyHMM:
    """Return a Mealy HMM generating the image process ``Phi(X)`` of ``hmm``.

    The code's sliding-window transducer (:meth:`SlidingBlockCode.to_transducer
    <sofic.shifts.sliding_block_code.SlidingBlockCode.to_transducer>`, whose
    state is the last ``w - 1`` input symbols) is composed with ``hmm`` via
    :func:`~sofic.automata.transducer_operations.transduce_generator`. The
    composed state ``(q, x_{t-w+1:t})`` must start *synchronized*: its initial
    law is that of ``(Q_{w-1}, X_{0:w-1})`` under ``hmm``, so the first emitted
    symbol is ``Y_0 = Phi(X_{0:w})`` and the image of ``X_{0:n+w-1}`` is
    ``Y_{0:n}`` exactly :cite:`LindMarcus1995` (§1.5, §6.1). Only states
    reachable from that initial law are kept.

    Parameters
    ----------
    hmm
        Generator of the source process ``X``; its initial distribution is used
        as given (stationary input gives the stationary image).
    code
        The sliding block code ``Phi``.

    Returns
    -------
    MealyHMM
        States are ``(hmm_state, context)`` pairs with ``context`` a tuple of
        the last ``w - 1`` source symbols; the observation alphabet is
        ``code.output_alphabet``.

    Raises
    ------
    ValueError
        If ``hmm`` can emit a ``w``-block that is missing from
        ``code.block_map``.
    """
    gen = hmm.to_mealy()
    composed = transduce_generator(code.to_transducer(), gen, complete=False, normalize=False)

    initial: dict[tuple[Hashable, tuple[Any, ...]], Any] = {
        (state, ()): mass for state, mass in gen.initial_distribution.items() if mass
    }
    for _ in range(code.window - 1):
        advanced: dict[tuple[Hashable, tuple[Any, ...]], Any] = defaultdict(int)
        for (state, context), mass in initial.items():
            for edge in gen.graph.out_transitions(state):
                advanced[(edge.target, (*context, edge.data.get(ATTR_EMISSION)))] += mass * edge.data.get(ATTR_PROB)
        initial = {key: mass for key, mass in advanced.items() if mass}

    for key in initial:
        if not composed.graph.has_state(key):
            _state, context = key
            raise ValueError(f"context {context!r} uses symbols outside the code's input alphabet")
    reachable = composed.graph.forward_reachable(frozenset(initial))
    for state, context in reachable:
        for edge in gen.graph.out_transitions(state):
            block = (*context, edge.data.get(ATTR_EMISSION))
            if block not in code.block_map:
                raise ValueError(f"block {block!r} of the process is not in block_map")

    image = MealyHMM(initial_distribution=initial, observation_alphabet=frozenset(code.output_alphabet))
    for state in composed.states():
        if state in reachable:
            image.graph.add_state(state)
    for transition in composed.transitions():
        if transition.source in reachable:
            image.add_transition(
                transition.source,
                transition.target,
                transition.data[ATTR_EMISSION],
                transition.data[ATTR_PROB],
            )
    return image


def higher_block(hmm: HiddenMarkovModel, k: int) -> MealyHMM:
    """Return the ``k``-block presentation of ``hmm``'s process.

    The higher block code ``beta_k`` :cite:`LindMarcus1995` (§1.4, Example
    1.5.10) recodes ``X`` to the process of overlapping ``k``-tuples

    .. math::

       Y_t = X_{t:t+k} = (X_t, \\ldots, X_{t+k-1}),

    so ``P(Y_{0:n} = y_{0:n}) = P(X_{0:n+k-1} = x_{0:n+k-1})`` when the tuples
    overlap consistently (``y_{t+1}[:-1] == y_t[1:]``) and ``x_{0:n+k-1}`` is
    the word they spell, and is zero otherwise. ``beta_k`` is a conjugacy: the
    1-block code ``y -> y[0]`` inverts it. Hence the entropy rate is unchanged,
    but the excess entropy and statistical complexity grow:

    .. math::

       \\mathbf{E}(Y) = \\mathbf{E}(X) + (k - 1)\\, h_\\mu(X), \\qquad
       C_\\mu(Y) = H[\\mathcal{S}_0, X_{-(k-1):0}]
                 = C_\\mu(X) + H[X_{-(k-1):0} \\mid \\mathcal{S}_0],

    with ``S_0`` the causal state of ``X`` after ``X_{-1}``. The first follows
    from ``H[Y_{0:L}] = H[X_{0:L+k-1}]`` and ``E = lim (H[X_{0:L}] - L h_mu)``
    :cite:`CrutchfieldFeldman2003`; the second because a ``Y``-past fixes both the
    last ``k - 1`` symbols (read again by the next tuple) and the causal state
    of ``X``.

    Parameters
    ----------
    hmm
        Generator of ``X``.
    k
        Block length, ``k >= 1`` (``k = 1`` recodes symbols to 1-tuples).

    Returns
    -------
    MealyHMM
        Unifilar whenever ``hmm`` is.
    """
    from sofic.shifts.sliding_block_code import SlidingBlockCode

    if k < 1:
        raise ValueError("k must be at least 1")
    gen = hmm.to_mealy()
    frontier: set[tuple[Hashable, tuple[Any, ...]]] = {(state, ()) for state in gen.states()}
    for _ in range(k):
        frontier = {
            (edge.target, (*word, edge.data.get(ATTR_EMISSION)))
            for state, word in frontier
            for edge in gen.graph.out_transitions(state)
        }
    blocks = {word for _state, word in frontier}
    code = SlidingBlockCode(
        {block: block for block in blocks},
        memory=0,
        anticipation=k - 1,
        input_alphabet=gen.observation_alphabet,
    )
    return image_process(gen, code)


__all__ = ["higher_block", "image_process"]
