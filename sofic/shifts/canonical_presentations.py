r"""Canonical NFAs of a sofic shift's language, as presentations of the shift.

A sofic shift :math:`X` is determined by its factor language :math:`L(X)`, which is factorial and extendable
:cite:`LindMarcus1995`. Any automaton for :math:`L(X)` presents :math:`X` once trimmed (accessible and
co-accessible states) and restricted to the states on bi-infinite paths: every path between trim states reads a
word of :math:`L(X)` (it extends to an accepted word, and :math:`L(X)` is factorial), and every word of
:math:`L(X)` labels a path. The canonical NFAs of :math:`L(X)` -- the átomaton :cite:`BrzozowskiTamm2014`, the
canonical residual finite-state automaton :cite:`Denis2002`, the maximized prime átomaton
:cite:`MaarandTamm2022`, the distromaton :cite:`Myers2014` and the universal automaton
:cite:`LombardySakarovitch2008` -- therefore give canonical, generally non-resolving presentations, beside the
right-resolving Fischer and Krieger covers. On the residual side the states are follower sets that are not unions
of other follower sets; the distromaton's states are the intersections of follower sets that are not unions of
smaller intersections, and the universal automaton's are all intersections of follower sets.

The minimal xor automaton of :mod:`sofic.automata.canonical.closures` has no counterpart here: it accepts by
parity of paths, and a labeled graph presents the words with *some* path.

These are the topological shadows of the stochastic analogues in
:mod:`sofic.generators.canonical_generators`, which replace unions of residuals by mixtures of future morphs.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from sofic.graph import ATTR_SYMBOL, TransitionGraph

if TYPE_CHECKING:
    from sofic.automata.nfa import NFA
    from sofic.shifts.sofic import SoficShift

__all__ = [
    "atomaton_presentation",
    "canonical_rfsa_presentation",
    "distromaton_presentation",
    "factor_language_nfa",
    "follower_sets_intersection_closed",
    "maximized_prime_atomaton_presentation",
    "universal_presentation",
]


def factor_language_nfa(shift: SoficShift) -> NFA:
    """The trimmed presentation read as an NFA with every state initial and accepting: it accepts :math:`L(X)`."""
    from sofic.automata.nfa import NFA

    trimmed = shift.trim_transient()
    states = frozenset(trimmed.states())
    return NFA(
        graph=trimmed.graph.copy(),
        input_alphabet=frozenset(shift.symbol_alphabet),
        initial_states=states,
        accepting_states=states,
    )


def _presentation(shift: SoficShift, automaton: Any) -> SoficShift:
    from sofic.automata.algorithms import trim
    from sofic.shifts.sofic import SoficShift

    # Untrimmed states can read words outside L(X): the distromaton's inaccessible state Sigma*, and the universal
    # automaton's state for an empty residual, which is accessible but has a loop on every symbol.
    automaton = trim(automaton)
    graph = TransitionGraph()
    for state in automaton.states():
        graph.add_state(state)
    for transition in automaton.transitions():
        graph.add_transition(transition.source, transition.target, **{ATTR_SYMBOL: transition.data[ATTR_SYMBOL]})
    return SoficShift(graph=graph, symbol_alphabet=frozenset(shift.symbol_alphabet)).trim_transient()


def canonical_rfsa_presentation(shift: SoficShift) -> SoficShift:
    """Presentation of ``shift`` by the canonical RFSA of its factor language (prime follower sets)."""
    from sofic.automata.canonical.residual import canonical_rfsa_from_language

    return _presentation(shift, canonical_rfsa_from_language(factor_language_nfa(shift)))


def maximized_prime_atomaton_presentation(shift: SoficShift) -> SoficShift:
    """Presentation of ``shift`` by the maximized prime átomaton of its factor language (prime predecessor sets)."""
    from sofic.automata.canonical.residual import maximized_prime_atomaton_from_language

    return _presentation(shift, maximized_prime_atomaton_from_language(factor_language_nfa(shift)))


def atomaton_presentation(shift: SoficShift) -> SoficShift:
    """Presentation of ``shift`` by the átomaton of its factor language (reverse of the minimal DFA of the reverse)."""
    from sofic.automata.canonical.residual import atomaton_from_language

    return _presentation(shift, atomaton_from_language(factor_language_nfa(shift)))


def distromaton_presentation(shift: SoficShift) -> SoficShift:
    """Presentation of ``shift`` by the distromaton of its factor language (join-irreducible meets of follower sets)."""
    from sofic.automata.canonical.closures import distromaton_from_language

    return _presentation(shift, distromaton_from_language(factor_language_nfa(shift)))


def universal_presentation(shift: SoficShift) -> SoficShift:
    """Presentation of ``shift`` by the universal automaton of its factor language (all meets of follower sets).

    Every essential presentation of ``shift`` is an NFA for :math:`L(X)` with all states initial and accepting, so
    it maps into the universal automaton :cite:`LombardySakarovitch2008`. The number of states can be exponential
    in the number of follower sets.
    """
    from sofic.automata.canonical.closures import universal_automaton_from_language

    return _presentation(shift, universal_automaton_from_language(factor_language_nfa(shift)))


def follower_sets_intersection_closed(shift: SoficShift) -> bool:
    r"""Whether every intersection of two follower sets of ``shift`` is a union of follower sets.

    Follower sets are the residuals of the factor language, so this is
    :func:`~sofic.automata.canonical.closures.is_intersection_closed` applied to :math:`L(X)`. When it holds the
    canonical RFSA of :math:`L(X)` is a state-minimal NFA (:cite:`Myers2014`, Theorem 4.4). A presentation with
    the fewest states is essential, and an essential presentation with every state initial and accepting is an NFA
    for :math:`L(X)`, so :func:`canonical_rfsa_presentation` then has the fewest states of any presentation of
    ``shift``.
    """
    from sofic.automata.canonical.closures import is_intersection_closed

    return is_intersection_closed(factor_language_nfa(shift))
