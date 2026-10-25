"""Syntactic monoids and star-freeness of regular languages.

The syntactic monoid of a regular language ``L`` is the quotient of ``A*``
by the syntactic congruence ``u ~ v iff xuy ∈ L ⇔ xvy ∈ L`` for all ``x, y``;
it is isomorphic to the transition monoid of the minimal complete DFA of
``L`` :cite:`Pin1986`. It is the smallest monoid recognizing ``L``, so its
properties are properties of the language rather than of a presentation.
"""

from __future__ import annotations

from collections.abc import Hashable, Iterable
from typing import Any

from sofic.algebra.adapters import semiautomaton
from sofic.algebra.transformation import DEFAULT_MAX_SIZE, SINK, TransformationSemigroup

__all__ = ["factor_monoid", "is_star_free", "syntactic_monoid", "transition_monoid"]


def transition_monoid(
    automaton: Any,
    *,
    alphabet: Iterable[Hashable] | None = None,
    max_size: int = DEFAULT_MAX_SIZE,
    monoid: bool = True,
) -> TransformationSemigroup:
    """Return the transition monoid of a deterministic automaton.

    Missing transitions go to an absorbing ``SINK``; symbols of ``alphabet``
    that never appear map every state to the sink. With ``monoid=False`` the
    result is the transition *semigroup* of nonempty words.
    """
    states, maps = semiautomaton(automaton, alphabet=alphabet)
    if not states:
        states = [SINK]
        sink_maps: dict[Hashable, dict[Hashable, Hashable]] = {symbol: {SINK: SINK} for symbol in maps}
        maps = sink_maps
    if not maps:
        return TransformationSemigroup({}, states=states, monoid=True, max_size=max_size)
    return TransformationSemigroup.from_maps(maps, states=states, monoid=monoid, max_size=max_size)


def syntactic_monoid(
    language: Any,
    *,
    alphabet: Iterable[Hashable] | None = None,
    max_size: int = DEFAULT_MAX_SIZE,
    monoid: bool = True,
) -> TransformationSemigroup:
    """Return the syntactic monoid of a regular language.

    ``language`` may be a labeled automaton (DFA, NFA, ...), an
    :class:`~sofic.automata.languages.base.AutomatonLanguage`, or a regular
    expression string accepted by :func:`~sofic.automata.regex.regex_to_nfa`.
    The alphabet defaults to the automaton's input alphabet; it matters,
    because a symbol that never occurs acts as a zero. With ``monoid=False``
    the result is the syntactic *semigroup* (images of nonempty words), which
    several variety tests need :cite:`Pin1986`.

    Examples
    --------
    >>> m = syntactic_monoid("(aa)*", alphabet="a")
    >>> len(m), m.is_aperiodic()
    (2, False)
    """
    from sofic.automata.algorithms import _to_nfa, _transition_alphabet, minimize
    from sofic.automata.base import LabeledAutomaton
    from sofic.automata.dfa import DFA
    from sofic.automata.languages.base import AutomatonLanguage
    from sofic.automata.nfa import NFA

    if isinstance(language, str):
        from sofic.automata.regex import regex_to_nfa

        language = regex_to_nfa(language, alphabet=alphabet)
    if isinstance(language, AutomatonLanguage):
        language = language.automaton
    if not isinstance(language, LabeledAutomaton):
        raise TypeError(f"cannot build a syntactic monoid from {type(language).__name__}")
    symbols = frozenset(alphabet) if alphabet is not None else _transition_alphabet(language)
    automaton = language if isinstance(language, (DFA, NFA)) else _to_nfa(language)
    minimal = minimize(automaton, alphabet=symbols)
    return transition_monoid(minimal, alphabet=_ordered(symbols), max_size=max_size, monoid=monoid)


def factor_monoid(
    machine: Any,
    *,
    alphabet: Iterable[Hashable] | None = None,
    max_size: int = DEFAULT_MAX_SIZE,
    monoid: bool = True,
) -> TransformationSemigroup:
    """Return the syntactic monoid of the words labeling paths in ``machine``.

    Every state is initial and accepting, so for a shift presentation this
    is the factor language ``B(X)`` and for a stochastic generator the
    support of its word distribution from any start state. Edges may carry
    ``symbol`` or ``emission`` labels; zero-probability edges are ignored, and
    the machine need not be unifilar.
    """
    from sofic.automata.nfa import NFA
    from sofic.graph import ATTR_EMISSION_EDGE, ATTR_PROB, ATTR_SYMBOL

    states = frozenset(machine.states())
    nfa = NFA(initial_states=states, accepting_states=states)
    for state in states:
        nfa.graph.add_state(state)
    symbols: set[Hashable] = set()
    for transition in machine.transitions():
        label = transition.data.get(ATTR_SYMBOL, transition.data.get(ATTR_EMISSION_EDGE))
        prob = transition.data.get(ATTR_PROB)
        if label is None or (prob is not None and prob == 0):
            continue
        nfa.add_transition(transition.source, transition.target, label)
        symbols.add(label)
    alphabet = frozenset(alphabet) if alphabet is not None else frozenset(symbols)
    nfa.input_alphabet = alphabet | symbols
    return syntactic_monoid(nfa, alphabet=alphabet, max_size=max_size, monoid=monoid)


def is_star_free(
    language: Any,
    *,
    alphabet: Iterable[Hashable] | None = None,
    max_size: int = DEFAULT_MAX_SIZE,
) -> bool:
    """Return whether a regular language is star-free.

    A regular language is star-free (expressible with union, concatenation
    and complement but no Kleene star) iff its syntactic monoid is aperiodic
    :cite:`Schutzenberger1965`, iff it is definable in first-order logic
    :cite:`McNaughtonPapert1971`, iff its Krohn-Rhodes complexity is zero.

    Examples
    --------
    >>> is_star_free("(ab)*", alphabet="ab"), is_star_free("(aa)*", alphabet="a")
    (True, False)
    """
    return syntactic_monoid(language, alphabet=alphabet, max_size=max_size).is_aperiodic()


def _ordered(symbols: Iterable[Hashable]) -> list[Hashable]:
    values = list(symbols)
    try:
        return sorted(values)  # type: ignore[type-var]
    except TypeError:
        return sorted(values, key=repr)
