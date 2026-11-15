r"""Canonical NFAs of a sofic shift's language, as presentations of the shift.

A sofic shift :math:`X` is determined by its factor language :math:`L(X)`, which is factorial and extendable
:cite:`LindMarcus1995`. Any automaton for :math:`L(X)` whose states are all accepting presents :math:`X` once
trimmed to the states on bi-infinite paths: every path reads a word of :math:`L(X)` (residuals of a factorial
language are factorial), and every word of :math:`L(X)` labels a path. The canonical NFAs of :math:`L(X)` --
the átomaton :cite:`BrzozowskiTamm2014`, the canonical residual finite-state automaton :cite:`Denis2002` and the
maximized prime átomaton :cite:`MaarandTamm2022` -- therefore give canonical, generally non-resolving
presentations, beside the right-resolving Fischer and Krieger covers. On the residual side the states are
follower sets that are not unions of other follower sets.

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
    "factor_language_nfa",
    "maximized_prime_atomaton_presentation",
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
    from sofic.shifts.sofic import SoficShift

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
