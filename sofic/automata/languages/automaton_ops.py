"""Automaton constructions for regular-language algebra."""

from __future__ import annotations

from collections import deque
from collections.abc import Callable, Hashable, Sequence
from typing import Any

from sofic.automata.algorithms import complete, determinize, minimize, trim
from sofic.automata.base import LabeledAutomaton
from sofic.automata.dfa import DFA
from sofic.automata.nfa import NFA
from sofic.graph import ATTR_SYMBOL, EPSILON, TransitionGraph

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


def _product_dfa(
    left_dfa: DFA,
    right_dfa: DFA,
    alphabet: frozenset[Any],
    accept_pred: Callable[[Hashable, Hashable], bool],
    *,
    complete_inputs: bool,
) -> DFA:
    """Product-construction BFS over ``left_dfa`` x ``right_dfa``.

    When ``complete_inputs`` is ``False`` (intersection semantics) transitions
    with a missing successor in either factor are skipped, yielding a sparse
    product. When ``True`` (difference semantics) both factors are assumed total
    so every symbol produces a successor and traps contribute to the language.
    ``accept_pred`` decides acceptance from the two component states.
    """
    graph = TransitionGraph()
    initial = (next(iter(left_dfa.initial_states)), next(iter(right_dfa.initial_states)))
    graph.add_state(initial)
    queue: deque[tuple[Hashable, Hashable]] = deque([initial])
    seen = {initial}
    while queue:
        pair = queue.popleft()
        for symbol in alphabet:
            left_next = left_dfa.delta(pair[0], symbol)
            right_next = right_dfa.delta(pair[1], symbol)
            if not complete_inputs and (len(left_next) != 1 or len(right_next) != 1):
                continue
            target = (next(iter(left_next)), next(iter(right_next)))
            if target not in seen:
                seen.add(target)
                graph.add_state(target)
                queue.append(target)
            graph.add_transition(pair, target, **{ATTR_SYMBOL: symbol})
    accepting = {pair for pair in seen if accept_pred(pair[0], pair[1])}
    return DFA(
        input_alphabet=alphabet,
        initial_states=frozenset({initial}),
        accepting_states=frozenset(accepting),
        graph=graph,
    )


def intersection_dfa(left: LabeledAutomaton, right: LabeledAutomaton) -> DFA:
    left_dfa = _to_dfa(left)
    right_dfa = _to_dfa(right)
    alphabet = left_dfa.input_alphabet | right_dfa.input_alphabet
    return _product_dfa(
        left_dfa,
        right_dfa,
        alphabet,
        lambda ls, rs: ls in left_dfa.accepting_states and rs in right_dfa.accepting_states,
        complete_inputs=False,
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
    return _product_dfa(
        left_dfa,
        right_dfa,
        symbols,
        lambda ls, rs: ls in left_dfa.accepting_states and rs not in right_dfa.accepting_states,
        complete_inputs=True,
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
    from sofic.automata.algorithms import _effective_alphabet as _shared

    return _shared(aut)
