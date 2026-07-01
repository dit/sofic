"""Automaton constructions for regular-language algebra."""

from __future__ import annotations

from collections.abc import Hashable, Sequence
from typing import Any

from pensive.automata.algorithms import complete, determinize, minimize, trim
from pensive.automata.base import LabeledAutomaton
from pensive.automata.dfa import DFA
from pensive.automata.nfa import NFA
from pensive.graph import ATTR_SYMBOL, EPSILON, TransitionGraph

_LEFT = object()
_RIGHT = object()
_START = object()
_STAR_START = object()


def _to_nfa(aut: LabeledAutomaton) -> NFA:
    if isinstance(aut, NFA):
        return trim(aut)
    nfa = NFA(
        input_alphabet=aut.input_alphabet,
        initial_states=aut.initial_states,
        accepting_states=aut.accepting_states,
        graph=aut.graph.copy(),
    )
    return trim(nfa)


def _to_dfa(aut: LabeledAutomaton, alphabet: frozenset[Any] | None = None) -> DFA:
    if isinstance(aut, DFA):
        return trim(aut)
    return trim(determinize(_to_nfa(aut), alphabet=alphabet))


def union_nfa(left: LabeledAutomaton, right: LabeledAutomaton) -> NFA:
    left_nfa = _to_nfa(left)
    right_nfa = _to_nfa(right)
    alphabet = left_nfa.input_alphabet | right_nfa.input_alphabet
    graph = TransitionGraph()
    start = (_START,)
    graph.add_state(start)
    for state in left_nfa.states():
        graph.add_state((_LEFT, state))
    for state in right_nfa.states():
        graph.add_state((_RIGHT, state))
    for initial in left_nfa.initial_states:
        graph.add_transition(start, (_LEFT, initial), **{ATTR_SYMBOL: EPSILON})
    for initial in right_nfa.initial_states:
        graph.add_transition(start, (_RIGHT, initial), **{ATTR_SYMBOL: EPSILON})
    for transition in left_nfa.transitions():
        graph.add_transition(
            (_LEFT, transition.source),
            (_LEFT, transition.target),
            **transition.data,
        )
    for transition in right_nfa.transitions():
        graph.add_transition(
            (_RIGHT, transition.source),
            (_RIGHT, transition.target),
            **transition.data,
        )
    accepting = {(_LEFT, s) for s in left_nfa.accepting_states} | {(_RIGHT, s) for s in right_nfa.accepting_states}
    return NFA(
        input_alphabet=alphabet,
        initial_states=frozenset({start}),
        accepting_states=frozenset(accepting),
        graph=graph,
    )


def intersection_dfa(left: LabeledAutomaton, right: LabeledAutomaton) -> DFA:
    left_dfa = _to_dfa(left)
    right_dfa = _to_dfa(right)
    alphabet = left_dfa.input_alphabet | right_dfa.input_alphabet
    graph = TransitionGraph()
    initial = (next(iter(left_dfa.initial_states)), next(iter(right_dfa.initial_states)))
    graph.add_state(initial)
    queue = [initial]
    seen = {initial}
    while queue:
        pair = queue.pop(0)
        for symbol in alphabet:
            left_next = left_dfa.delta(pair[0], symbol)
            right_next = right_dfa.delta(pair[1], symbol)
            if len(left_next) != 1 or len(right_next) != 1:
                continue
            target = (next(iter(left_next)), next(iter(right_next)))
            if target not in seen:
                seen.add(target)
                graph.add_state(target)
                queue.append(target)
            graph.add_transition(pair, target, **{ATTR_SYMBOL: symbol})
    accepting = {
        pair for pair in seen if pair[0] in left_dfa.accepting_states and pair[1] in right_dfa.accepting_states
    }
    return DFA(
        input_alphabet=alphabet,
        initial_states=frozenset({initial}),
        accepting_states=frozenset(accepting),
        graph=graph,
    )


def complement_dfa(dfa: LabeledAutomaton, alphabet: frozenset[Any] | None = None) -> DFA:
    symbols = _effective_alphabet(dfa) if alphabet is None else alphabet
    complete_dfa = complete(_to_dfa(dfa, alphabet=symbols), alphabet=symbols)
    all_states = set(complete_dfa.states())
    accepting = all_states - set(complete_dfa.accepting_states)
    result = complete_dfa.copy()
    result.accepting_states = frozenset(accepting)
    return result


def difference_dfa(
    left: LabeledAutomaton,
    right: LabeledAutomaton,
    alphabet: frozenset[Any] | None = None,
) -> DFA:
    """Return a DFA recognizing ``left`` minus ``right`` over ``alphabet``."""
    symbols = alphabet if alphabet is not None else _effective_alphabet(left) | _effective_alphabet(right)
    left_dfa = complete(_to_dfa(left, alphabet=symbols), alphabet=symbols)
    right_dfa = complete(_to_dfa(right, alphabet=symbols), alphabet=symbols)
    graph = TransitionGraph()
    initial = (next(iter(left_dfa.initial_states)), next(iter(right_dfa.initial_states)))
    graph.add_state(initial)
    queue = [initial]
    seen = {initial}
    while queue:
        pair = queue.pop(0)
        for symbol in symbols:
            target = (
                next(iter(left_dfa.delta(pair[0], symbol))),
                next(iter(right_dfa.delta(pair[1], symbol))),
            )
            if target not in seen:
                seen.add(target)
                graph.add_state(target)
                queue.append(target)
            graph.add_transition(pair, target, **{ATTR_SYMBOL: symbol})
    accepting = {
        pair for pair in seen if pair[0] in left_dfa.accepting_states and pair[1] not in right_dfa.accepting_states
    }
    return DFA(
        input_alphabet=symbols,
        initial_states=frozenset({initial}),
        accepting_states=frozenset(accepting),
        graph=graph,
    )


def concat_nfa(left: LabeledAutomaton, right: LabeledAutomaton) -> NFA:
    left_nfa = _to_nfa(left)
    right_nfa = _to_nfa(right)
    alphabet = left_nfa.input_alphabet | right_nfa.input_alphabet
    graph = left_nfa.graph.copy()
    for state in right_nfa.states():
        graph.add_state((_RIGHT, state))
    for transition in right_nfa.transitions():
        graph.add_transition(
            (_RIGHT, transition.source),
            (_RIGHT, transition.target),
            **transition.data,
        )
    for accept in left_nfa.accepting_states:
        for initial in right_nfa.initial_states:
            graph.add_transition(accept, (_RIGHT, initial), **{ATTR_SYMBOL: EPSILON})
    accepting = {(_RIGHT, s) for s in right_nfa.accepting_states}
    return NFA(
        input_alphabet=alphabet,
        initial_states=left_nfa.initial_states,
        accepting_states=frozenset(accepting),
        graph=graph,
    )


def kleene_star_nfa(aut: LabeledAutomaton) -> NFA:
    nfa = _to_nfa(aut)
    graph = nfa.graph.copy()
    start = (_STAR_START,)
    graph.add_state(start)
    for initial in nfa.initial_states:
        graph.add_transition(start, initial, **{ATTR_SYMBOL: EPSILON})
    for accept in nfa.accepting_states:
        for initial in nfa.initial_states:
            graph.add_transition(accept, initial, **{ATTR_SYMBOL: EPSILON})
    accepting = set(nfa.accepting_states) | {start}
    return NFA(
        input_alphabet=nfa.input_alphabet,
        initial_states=frozenset({start}),
        accepting_states=frozenset(accepting),
        graph=graph,
    )


def left_quotient_automaton(u: Sequence[Any], aut: LabeledAutomaton) -> NFA:
    nfa = _to_nfa(aut)
    current = nfa.epsilon_closure(set(nfa.initial_states))
    for symbol in u:
        next_states: set[Hashable] = set()
        for state in current:
            next_states.update(nfa.delta(state, symbol))
        current = nfa.epsilon_closure(next_states)
    return NFA(
        input_alphabet=nfa.input_alphabet,
        initial_states=frozenset(current),
        accepting_states=nfa.accepting_states,
        graph=nfa.graph.copy(),
    )


def right_quotient_automaton(aut: LabeledAutomaton, u: Sequence[Any]) -> NFA:
    reversed_u = tuple(reversed(u))
    rev = _to_nfa(aut).reverse()
    return left_quotient_automaton(reversed_u, rev).reverse()


def minimal_dfa_from_language(aut: LabeledAutomaton) -> DFA:
    return minimize(_to_dfa(aut))


def state_residual_languages(dfa: DFA) -> dict[Hashable, DFA]:
    """Map each state to the DFA for its left-quotient (right-language) residual."""
    residuals: dict[Hashable, DFA] = {}
    for state in dfa.states():
        sub = dfa.copy()
        sub.initial_states = frozenset({state})
        sub.accepting_states = dfa.accepting_states
        residuals[state] = minimize(_to_dfa(sub))
    return residuals


def _effective_alphabet(aut: LabeledAutomaton) -> frozenset[Any]:
    from pensive.automata.algorithms import _effective_alphabet as _shared

    return _shared(aut)
