"""Topological information anatomy of a sofic shift: the MME split of ``h_top``.

Topological entropy ``h_top`` is the measure-independent counterpart of the metric
entropy ``h_mu``: by the variational principle ``h_top = sup_nu h_nu``, attained by
the **measure of maximal entropy** (MME) -- the Parry measure of the shift
(Parry 1964). This module evaluates the *information anatomy* (James et al. 2013)
of the observed symbol process at that MME, giving a topological analogue of the
``h_mu = b_mu + r_mu`` split:

    h_top = b_top + r_top,   b_top := b_{mu*},   r_top := r_{mu*},

where ``mu*`` is the MME. Concretely:

- ``h_top = h_{mu*} = log2(lambda)`` (Perron eigenvalue of the right-resolving
  presentation's adjacency matrix),
- ``b_top = I[X_0 : X_{1:} | X_{:0}]`` under ``mu*`` (bound information), and
- ``r_top = H[X_0 | X_{:0}, X_{1:}]`` under ``mu*`` -- the **erasure entropy rate**
  (Verdu & Weissman 2008) of the MME (ephemeral information).

Because the MME and the observed symbol process are intrinsic to the sofic shift,
``b_top`` and ``r_top`` are functions of the shift space (up to alphabet
relabeling / 1-block conjugacy), refining ``h_top`` the way ``b_mu`` / ``r_mu``
refine ``h_mu``. They are exact: the MME is a finite Markov chain on the
presentation, so the bidirectional-epsilon-machine anatomy is closed-form (no
Monte-Carlo, no finite-block truncation).

Pipeline (see :func:`topological_anatomy`):

1. Put the presentation in right-resolving (unifilar) form
   (:func:`_right_resolving`) -- required so ``h_top`` is exact rather than the
   path-overcount of a nondeterministic presentation.
2. Build the Parry MME as a labeled :class:`~sofic.generators.mealy.MealyHMM`
   (:func:`parry_measure_sofic`).
3. Minimize to the causal presentation
   (:meth:`~sofic.generators.epsilon_machine.EpsilonMachine.from_hmm`), pair it
   with its time reverse
   (:meth:`~sofic.generators.epsilon_machine.EpsilonMachine.to_bidirectional`),
   and read the anatomy
   (:meth:`~sofic.generators.bidirectional_epsilon_machine.BidirectionalEpsilonMachine.information_anatomy`).

References: Parry (1964), *Intrinsic Markov chains*; James, Ellison & Crutchfield
(2013), *Anatomy of a bit*; Verdu & Weissman (2008), *The information lost in
erasures*.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from sofic.exceptions import UnifilarityError
from sofic.graph import ATTR_SYMBOL

if TYPE_CHECKING:
    from sofic.generators.mealy import MealyHMM
    from sofic.shifts.sofic import SoficShift


def _dedup_symbol_edges(shift: SoficShift) -> SoficShift:
    """Return a copy of ``shift`` with duplicate ``(source, target, symbol)`` edges merged.

    The Fischer-cover construction maps every original transition onto a
    follower-class edge without deduplication, so merging classes can create
    several identical ``(source, target, symbol)`` edges. Those are the *same*
    labeled edge and must be counted once for the adjacency/Perron entropy to be
    correct; genuine right-resolving parallelism (same ``(source, target)``,
    *different* symbols) is preserved.
    """
    from sofic.shifts.sofic import SoficShift

    result = SoficShift(symbol_alphabet=shift.symbol_alphabet)
    for state in shift.states():
        result.graph.add_state(state)
    seen: set[tuple[object, object, object]] = set()
    for transition in shift.transitions():
        symbol = transition.data.get(ATTR_SYMBOL)
        key = (transition.source, transition.target, symbol)
        if key in seen:
            continue
        seen.add(key)
        result.add_transition(transition.source, transition.target, symbol)
    return result


def _right_resolving(shift: SoficShift) -> SoficShift:
    """Return a right-resolving (unifilar) presentation of ``shift``.

    If ``shift`` is already unifilar it is returned unchanged. Otherwise the right
    Fischer cover (:meth:`~sofic.shifts.covers.RightFischerCover.from_sofic`) is
    built and its duplicate labeled edges merged (:func:`_dedup_symbol_edges`).
    The cover construction uses a bounded follower language, so it is not
    guaranteed to determinize every presentation; if the result is still not
    unifilar a :class:`~sofic.exceptions.UnifilarityError` is raised asking for a
    right-resolving input.
    """
    if shift.is_unifilar():
        return shift
    from sofic.shifts.covers import RightFischerCover

    cover = _dedup_symbol_edges(RightFischerCover.from_sofic(shift))
    if not cover.is_unifilar():
        raise UnifilarityError(
            "could not derive a right-resolving presentation of the sofic shift "
            "automatically; supply a unifilar (right-resolving) SoficShift"
        )
    return cover


def parry_measure_sofic(shift: SoficShift) -> MealyHMM:
    """Return the measure of maximal entropy (Parry measure) as a labeled MealyHMM.

    Puts ``shift`` in right-resolving form (:func:`_right_resolving`) and builds
    the Parry chain ``P[i, j] = A[i, j] v_j / (lambda v_i)`` with stationary
    ``pi ~ u * v`` from the Perron data of the adjacency matrix ``A`` (Parry 1964),
    preserving each edge's emitted symbol. Its entropy rate is
    ``log2(lambda) = h_top`` and its observed process is the shift's MME symbol
    process.
    """
    from sofic.shifts.parry_construction import parry_measure

    return parry_measure(_right_resolving(shift))


def topological_anatomy(shift: SoficShift) -> dict[str, float]:
    """Return the topological information anatomy ``{h_top, b_top, r_top, excess_entropy}``.

    Evaluates the metric anatomy (James et al. 2013) of the observed symbol
    process at the measure of maximal entropy (:func:`parry_measure_sofic`):

    - ``h_top`` -- topological entropy ``= log2(lambda)`` (the MME entropy rate),
    - ``b_top`` -- MME bound information ``I[X_0 : X_{1:} | X_{:0}]``,
    - ``r_top`` -- MME ephemeral information ``H[X_0 | X_{:0}, X_{1:}]`` (the MME
      erasure entropy rate, Verdu & Weissman 2008),
    - ``excess_entropy`` -- MME excess entropy ``E = I[past : future]``,

    with ``h_top = b_top + r_top`` exactly. Requires ``dit`` for the anatomy
    entropies.
    """
    from sofic.generators.epsilon_machine import EpsilonMachine

    parry = parry_measure_sofic(shift)
    if not list(parry.states()):
        return {"h_top": 0.0, "b_top": 0.0, "r_top": 0.0, "excess_entropy": 0.0}

    anatomy = EpsilonMachine.from_hmm(parry).information_anatomy()
    return {
        "h_top": float(anatomy["entropy_rate"]),
        "b_top": float(anatomy["bound_mu"]),
        "r_top": float(anatomy["ephemeral_mu"]),
        "excess_entropy": float(anatomy["excess_entropy"]),
    }
