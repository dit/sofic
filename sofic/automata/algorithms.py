"""Automata constructions: reverse, determinize, and DFA minimization."""

from __future__ import annotations

from collections import deque
from collections.abc import Hashable, Sequence
from typing import Any, Literal, TypeVar

from sofic.automata.base import LabeledAutomaton
from sofic.automata.dfa import DFA
from sofic.automata.nfa import NFA
from sofic.graph import ATTR_SYMBOL, EPSILON

MinimizationAlgorithm = Literal["hopcroft", "moore", "brzozowski"]

_TRAP = object()

L = TypeVar("L", bound=LabeledAutomaton)


def trim(aut: L) -> L:  # noqa: UP047 - keep Python 3.11 compatibility.
    """Remove states not reachable from initials or not coaccessible to acceptors."""
    reachable = _forward_reachable(aut)
    coaccessible = _backward_coaccessible(aut)
    keep = reachable & coaccessible

    result = aut.copy()
    for state in list(result.states()):
        if state not in keep:
            result.graph.nx.remove_node(state)

    result.initial_states = frozenset(s for s in aut.initial_states if s in keep)
    result.accepting_states = frozenset(s for s in aut.accepting_states if s in keep)
    return result


def complete(dfa: DFA, alphabet: frozenset[Any] | None = None) -> DFA:
    """Add a trap state so every state has one outgoing transition per symbol."""
    symbols = alphabet if alphabet is not None else _effective_alphabet(dfa)
    result = dfa.copy()
    trap = _TRAP
    if trap not in result.graph.nx:
        result.graph.add_state(trap)

    for state in list(result.states()):
        if state == trap:
            continue
        for symbol in symbols:
            if not result.delta(state, symbol):
                result.add_transition(state, trap, symbol)

    for symbol in symbols:
        result.add_transition(trap, trap, symbol)

    if symbols:
        result.input_alphabet = frozenset(symbols) | result.input_alphabet
    return result


def reverse(aut: NFA | DFA) -> NFA:
    """Return an NFA recognizing the reversed language.

    For a generic :class:`~sofic.base.StateMachine`, use
    :func:`~sofic.operations.reverse` instead.
    """
    return aut.reverse()


def determinize(nfa: NFA, *, alphabet: frozenset[Any] | None = None) -> DFA:
    """Subset construction with epsilon closure."""
    symbols = alphabet if alphabet is not None else _effective_alphabet(nfa)
    start = frozenset(nfa.epsilon_closure(set(nfa.initial_states)))

    subsets: dict[frozenset[Hashable], frozenset[Hashable]] = {start: start}
    queue: deque[frozenset[Hashable]] = deque([start])
    edges: list[tuple[frozenset[Hashable], frozenset[Hashable], Any]] = []

    while queue:
        current = queue.popleft()
        for symbol in symbols:
            next_raw: set[Hashable] = set()
            for state in current:
                next_raw.update(nfa.delta(state, symbol))
            target = frozenset(nfa.epsilon_closure(next_raw))
            edges.append((current, target, symbol))
            if target not in subsets:
                subsets[target] = target
                queue.append(target)

    dfa = DFA(
        input_alphabet=symbols,
        initial_states=frozenset({start}),
        accepting_states=frozenset(s for s in subsets if s & nfa.accepting_states),
    )
    for subset in subsets:
        dfa.graph.add_state(subset)
    for source, target, symbol in edges:
        dfa.add_transition(source, target, symbol)
    return dfa


def minimize(
    aut: DFA | NFA,
    *,
    algorithm: MinimizationAlgorithm = "hopcroft",
    alphabet: frozenset[Any] | None = None,
) -> DFA:
    """Return a minimal DFA equivalent to ``aut``."""
    if algorithm == "brzozowski":
        return minimize_brzozowski(aut, alphabet=alphabet)
    dfa = aut if isinstance(aut, DFA) else determinize(aut, alphabet=alphabet)
    if algorithm == "hopcroft":
        return minimize_hopcroft(dfa, alphabet=alphabet)
    if algorithm == "moore":
        return minimize_moore(dfa, alphabet=alphabet)
    raise ValueError(f"unknown minimization algorithm {algorithm!r}")


def minimize_brzozowski(
    aut: DFA | NFA,
    *,
    alphabet: frozenset[Any] | None = None,
) -> DFA:
    """Minimize via Brzozowski double reversal: det(rev(det(rev(A))))."""
    nfa = aut if isinstance(aut, NFA) else _dfa_as_nfa(aut)
    return determinize(trim(nfa).reverse().determinize(alphabet=alphabet).reverse().determinize(alphabet=alphabet))


def minimize_moore(dfa: DFA, *, alphabet: frozenset[Any] | None = None) -> DFA:
    """Minimize a DFA using Moore (1961) partition refinement."""
    symbols = alphabet if alphabet is not None else _effective_alphabet(dfa)
    work = complete(trim(dfa), symbols)
    states = sorted(work.states(), key=repr)
    if not states:
        return work

    partition = _initial_partition(states, work.accepting_states)
    changed = True
    while changed:
        changed = False
        new_partition: list[set[Hashable]] = []
        for block in partition:
            refined: list[set[Hashable]] = [set(block)]
            for symbol in sorted(symbols, key=repr):
                next_refined: list[set[Hashable]] = []
                for piece in refined:
                    groups: dict[int, set[Hashable]] = {}
                    for state in piece:
                        successor = _dfa_successor(work, state, symbol)
                        index = -1 if successor is None else _block_index(partition, successor)
                        groups.setdefault(index, set()).add(state)
                    next_refined.extend(groups.values())
                refined = next_refined
            if len(refined) > 1:
                changed = True
            new_partition.extend(refined)
        partition = new_partition

    return _quotient_from_partition(work, partition, symbols)


def minimize_hopcroft(dfa: DFA, *, alphabet: frozenset[Any] | None = None) -> DFA:
    """Minimize a DFA using Hopcroft's algorithm."""
    symbols = alphabet if alphabet is not None else _effective_alphabet(dfa)
    work = complete(trim(dfa), symbols)
    states = sorted(work.states(), key=repr)
    if not states:
        return work

    accepting = set(work.accepting_states)
    partition: list[set[Hashable]] = []
    if accepting:
        partition.append(accepting & set(states))
    non_accepting = set(states) - accepting
    if non_accepting:
        partition.append(non_accepting)

    pred = _inverse_transitions(work, states, symbols)
    worklist: list[set[Hashable]] = [block.copy() for block in partition]

    while worklist:
        focus = worklist.pop()
        for symbol in symbols:
            predecessors: set[Hashable] = set()
            for state in focus:
                predecessors.update(pred[state][symbol])
            refined_partition: list[set[Hashable]] = []
            for block in partition:
                intersection = block & predecessors
                difference = block - predecessors
                if intersection and difference:
                    refined_partition.append(intersection)
                    refined_partition.append(difference)
                    if block in worklist:
                        worklist.remove(block)
                        worklist.append(intersection)
                        worklist.append(difference)
                    else:
                        if len(intersection) <= len(difference):
                            worklist.append(intersection)
                        else:
                            worklist.append(difference)
                else:
                    refined_partition.append(block)
            partition = refined_partition

    return _quotient_from_partition(work, partition, symbols)


def equivalent(
    aut1: LabeledAutomaton,
    aut2: LabeledAutomaton,
    alphabet: frozenset[Any],
) -> bool:
    """Return whether two automata recognize the same language over ``alphabet``."""
    d1 = minimize(_to_nfa(aut1), alphabet=alphabet, algorithm="hopcroft")
    d2 = minimize(_to_nfa(aut2), alphabet=alphabet, algorithm="hopcroft")
    return _isomorphic_minimal_dfa(d1, d2, alphabet)


def _to_nfa(aut: LabeledAutomaton) -> NFA:
    if isinstance(aut, NFA):
        return aut
    if isinstance(aut, DFA):
        return _dfa_as_nfa(aut)
    raise TypeError(f"unsupported automaton type {type(aut)!r}")


def _dfa_as_nfa(dfa: DFA) -> NFA:
    nfa = NFA(
        input_alphabet=dfa.input_alphabet,
        initial_states=dfa.initial_states,
        accepting_states=dfa.accepting_states,
        graph=dfa.graph.copy(),
    )
    return nfa


def _effective_alphabet(aut: LabeledAutomaton) -> frozenset[Any]:
    symbols = {symbol for symbol in aut.input_alphabet if symbol is not EPSILON}
    if symbols:
        return frozenset(symbols)
    for transition in aut.transitions():
        symbol = transition.data.get(ATTR_SYMBOL)
        if symbol is not None and symbol is not EPSILON:
            symbols.add(symbol)
    return frozenset(symbols)


def _forward_reachable(aut: LabeledAutomaton) -> set[Hashable]:
    if not aut.initial_states:
        return set()
    seed = set(aut.epsilon_closure(set(aut.initial_states)))
    return set(aut.graph.forward_reachable(seed))


def _backward_coaccessible(aut: LabeledAutomaton) -> set[Hashable]:
    predecessors: dict[Hashable, set[Hashable]] = {state: set() for state in aut.states()}
    for transition in aut.transitions():
        predecessors.setdefault(transition.target, set()).add(transition.source)

    coaccessible = set(aut.accepting_states)
    queue = deque(coaccessible)
    while queue:
        state = queue.popleft()
        for predecessor in predecessors.get(state, ()):
            if predecessor not in coaccessible:
                coaccessible.add(predecessor)
                queue.append(predecessor)
    return coaccessible


def _dfa_successor(dfa: DFA, state: Hashable, symbol: Any) -> Hashable | None:
    successors = dfa.delta(state, symbol)
    if not successors:
        return None
    return next(iter(successors))


def _initial_partition(states: Sequence[Hashable], accepting: frozenset[Hashable]) -> list[set[Hashable]]:
    accepting_block = set(accepting) & set(states)
    non_accepting = set(states) - accepting_block
    partition: list[set[Hashable]] = []
    if accepting_block:
        partition.append(accepting_block)
    if non_accepting:
        partition.append(non_accepting)
    return partition


def _block_index(partition: list[set[Hashable]], state: Hashable) -> int:
    for index, block in enumerate(partition):
        if state in block:
            return index
    raise KeyError(state)


def _inverse_transitions(
    dfa: DFA,
    states: Sequence[Hashable],
    symbols: frozenset[Any],
) -> dict[Hashable, dict[Any, set[Hashable]]]:
    pred: dict[Hashable, dict[Any, set[Hashable]]] = {state: {symbol: set() for symbol in symbols} for state in states}
    for state in states:
        for symbol in symbols:
            target = _dfa_successor(dfa, state, symbol)
            if target is not None and target in pred:
                pred[target][symbol].add(state)
    return pred


def _quotient_from_partition(dfa: DFA, partition: list[set[Hashable]], symbols: frozenset[Any]) -> DFA:
    blocks = [block for block in partition if block]
    block_of = {state: index for index, block in enumerate(blocks) for state in block}
    representatives = [min(block, key=repr) for block in blocks]
    rep_for_block = {index: frozenset({representatives[index]}) for index in range(len(blocks))}

    initial_block = block_of[next(iter(dfa.initial_states))] if dfa.initial_states else 0
    accepting_blocks = frozenset(
        rep_for_block[index] for index, block in enumerate(blocks) if block & dfa.accepting_states
    )

    result = DFA(
        input_alphabet=symbols,
        initial_states=frozenset({rep_for_block[initial_block]}),
        accepting_states=accepting_blocks,
    )
    for index, _rep in enumerate(representatives):
        result.graph.add_state(rep_for_block[index])

    for index, rep in enumerate(representatives):
        source = rep_for_block[index]
        for symbol in symbols:
            target_state = _dfa_successor(dfa, rep, symbol)
            if target_state is None:
                continue
            target_block = block_of[target_state]
            result.add_transition(source, rep_for_block[target_block], symbol)

    return trim(result)


def _isomorphic_minimal_dfa(d1: DFA, d2: DFA, alphabet: frozenset[Any]) -> bool:
    states1 = list(d1.states())
    states2 = list(d2.states())
    if len(states1) != len(states2):
        return False
    if not d1.initial_states or not d2.initial_states:
        return not d1.initial_states and not d2.initial_states

    start1 = next(iter(d1.initial_states))
    start2 = next(iter(d2.initial_states))
    if (start1 in d1.accepting_states) != (start2 in d2.accepting_states):
        return False

    mapping: dict[Hashable, Hashable] = {start1: start2}
    queue = deque([start1])
    while queue:
        state1 = queue.popleft()
        state2 = mapping[state1]
        for symbol in sorted(alphabet, key=repr):
            succ1 = _dfa_successor(d1, state1, symbol)
            succ2 = _dfa_successor(d2, state2, symbol)
            if succ1 is None and succ2 is None:
                continue
            if succ1 is None or succ2 is None:
                return False
            if succ1 in mapping:
                if mapping[succ1] != succ2:
                    return False
            else:
                if (succ1 in d1.accepting_states) != (succ2 in d2.accepting_states):
                    return False
                mapping[succ1] = succ2
                queue.append(succ1)
    return len(mapping) == len(states1)
